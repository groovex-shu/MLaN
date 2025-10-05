#!/usr/bin/env python3
"""
Continuous ROS process memory monitoring without pandas.
Each process gets its own column, new measurements appended as rows.
"""

import re
import subprocess
import sys
import time
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict, OrderedDict


def get_all_processes():
    """Get ROS processes, run_slam, and tf processes separately."""
    try:
        result = subprocess.run([
            'ps', 'axo', 'pid,%mem,rss,comm,args', '--no-headers'
        ], capture_output=True, text=True, check=True)
        
        processes = {}  # Dict for easier handling
        
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            
            pid, mem_percent, rss_kb, comm, full_args = parts
            
            # Check what type of process this is
            process_type = classify_process(comm, full_args)
            
            if process_type:
                try:
                    friendly_name = get_friendly_name(comm, full_args, process_type)
                    # Use process_name_pid as unique key
                    process_key = f"{friendly_name}_{pid}"
                    processes[process_key] = float(rss_kb) / 1024  # Memory in MB
                except ValueError:
                    continue
        
        return processes
        
    except subprocess.CalledProcessError as e:
        print(f"Error running ps: {e}", file=sys.stderr)
        return {}


def classify_process(comm, full_args):
    """Classify process type: 'ros', 'run_slam', 'lovot_localization', 'tf', or None."""
    args_lower = full_args.lower()
    
    # Check for run_slam first (lovot_slam Python process)
    if 'run_slam' in args_lower:
        return 'run_slam'
    
    # Check for lovot-localization services (systemd services or Python processes)
    if any(loc_key in args_lower for loc_key in ['lovot-localization-localizer', 'lovot-localization-builder']):
        return 'lovot_localization'
    
    # Check for lovot_slam Python processes
    if 'lovot_slam' in args_lower or ('python' in comm and 'lovot_slam' in args_lower):
        return 'lovot_localization'
    
    # Check for tf processes (lovot transform processes)  
    if any(tf_key in args_lower for tf_key in ['lovot-tf-', 'lovot_tf_', 'tf_bridge', 'twist_publisher']):
        return 'tf'
    
    # Check for ROS processes (pure ROS nodes)
    ros_indicators = [
        '/opt/ros/', 'roslaunch', 'rosmaster', 'roscore', 'rosout',
        '_node', 'amcl', 'localizer', 'map_server', 'static_transform_publisher'
    ]
    
    # ROS paths and executables
    if any(ros_key in args_lower for ros_key in ros_indicators):
        return 'ros'
    
    # ROS Python scripts in /opt/ros/
    if 'python' in comm and '/opt/ros/' in args_lower:
        return 'ros'
    
    return None


def get_friendly_name(comm, full_args, process_type=None):
    """Generate friendly name based on process type."""
    
    if process_type == 'run_slam':
        if 'spike' in full_args:
            return 'RUN_SLAM_spike'
        elif 'tom' in full_args:
            return 'RUN_SLAM_tom'
        elif 'shaun' in full_args:
            return 'RUN_SLAM_shaun'
        else:
            return 'RUN_SLAM'
    
    elif process_type == 'lovot_localization':
        if 'lovot-localization-localizer' in full_args:
            return 'LOVOT_SLAM_localizer'
        elif 'lovot-localization-builder' in full_args:
            return 'LOVOT_SLAM_builder'
        elif 'lovot_slam_manager' in full_args:
            return 'LOVOT_SLAM_manager'
        elif 'main.py' in full_args:
            return 'LOVOT_SLAM_main'
        else:
            return 'LOVOT_SLAM_process'
    
    elif process_type == 'tf':
        if 'tf_bridge' in full_args:
            return 'TF_bridge'
        elif 'twist_publisher' in full_args:
            return 'TF_twist_publisher'
        else:
            return f'TF_{comm}'
    
    elif process_type == 'ros':
        # ROS process naming
        if 'roslaunch' in full_args:
            # Extract launch file name
            for arg in full_args.split():
                if arg.endswith('.launch'):
                    launch_name = arg.split('/')[-1][:-7]  # Remove .launch
                    return f'ROS_roslaunch_{launch_name}'
            return 'ROS_roslaunch'
        
        elif 'rosmaster' in full_args:
            return 'ROS_rosmaster'
        elif 'rosout' in full_args:
            return 'ROS_rosout'
        elif 'localizer' in full_args:
            return 'ROS_localizer'
        elif 'amcl' in full_args:
            return 'ROS_amcl'
        elif 'map_server' in full_args:
            return 'ROS_map_server'
        elif '_node' in comm:
            return f'ROS_{comm}'
        elif 'python' in comm and '/opt/ros/' in full_args:
            # Extract Python script name
            args = full_args.split()
            for arg in args:
                if arg.endswith('.py'):
                    script_name = arg.split('/')[-1][:-3]
                    return f'ROS_{script_name}'
        
        # Fallback for ROS processes
        return f'ROS_{comm}'
    
    return comm


class MemoryMonitor:
    """Memory monitoring class without pandas."""
    
    def __init__(self):
        self.data = []  # List of dict records
        self.all_processes = set()  # Track all processes we've seen
    
    def add_measurement(self, timestamp, processes):
        """Add a measurement row."""
        record = {'timestamp': timestamp}
        record.update(processes)
        self.data.append(record)
        self.all_processes.update(processes.keys())
    
    def save_to_csv(self, filename):
        """Save data to CSV file."""
        if not self.data:
            return
        
        # Create ordered fieldnames
        fieldnames = ['timestamp'] + sorted(self.all_processes)
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in self.data:
                # Fill missing processes with 0
                row = {'timestamp': record['timestamp']}
                for proc in self.all_processes:
                    row[proc] = record.get(proc, 0)
                writer.writerow(row)
    
    def plot_memory_trends(self, output_file='ros_memory_plot.png'):
        """Generate memory usage trend plots."""
        if len(self.data) < 2:
            print("Not enough data for plotting.")
            return
        
        # Convert timestamps to datetime objects
        timestamps = [datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S') 
                     for record in self.data]
        
        # Get active processes (those that had memory > 0)
        active_processes = []
        for proc in self.all_processes:
            max_mem = max(record.get(proc, 0) for record in self.data)
            if max_mem > 0:
                active_processes.append(proc)
        
        if not active_processes:
            print("No active processes found for plotting.")
            return
        
        # Create plot
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot: Individual process memory usage (top 10)
        for proc in active_processes[:10]:
            memory_values = [record.get(proc, 0) for record in self.data]
            ax1.plot(timestamps, memory_values, label=proc, marker='o', markersize=3)
        
        ax1.set_title('ROS Process Memory Usage Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Memory (MB)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Calculate total for summary stats only
        total_memory = []
        for record in self.data:
            total = sum(record.get(proc, 0) for proc in active_processes)
            total_memory.append(total)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Memory trend plot saved to: {output_file}")
        print("No GUI display - plot saved as PNG file only")
        
        # Print summary statistics
        print("\nMemory Usage Summary:")
        print("-" * 50)
        for proc in active_processes:
            memory_values = [record.get(proc, 0) for record in self.data]
            max_mem = max(memory_values)
            avg_mem = sum(memory_values) / len(memory_values)
            if max_mem > 0:
                print(f"{proc:<30}: Max: {max_mem:6.1f} MB, Avg: {avg_mem:6.1f} MB")
        
        max_total = max(total_memory)
        avg_total = sum(total_memory) / len(total_memory)
        print(f"\nTotal Memory - Max: {max_total:.1f} MB, Avg: {avg_total:.1f} MB")
        print(f"Monitoring Duration: {len(self.data)} measurements over {len(self.data) * 15 / 60:.1f} minutes")


def continuous_monitoring(interval=15, output_file='ros_memory_timeseries.csv'):
    """Continuously monitor and save data."""
    monitor = MemoryMonitor()
    
    print(f"Starting continuous monitoring (every {interval} seconds)")
    print(f"Output file: {output_file}")
    print("Press Ctrl+C to stop\n")
    
    try:
        iteration = 0
        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            processes = get_all_processes()
            
            if processes:
                monitor.add_measurement(timestamp, processes)
                monitor.save_to_csv(output_file)
                
                iteration += 1
                total_memory = sum(processes.values())
                print(f"[{timestamp}] Iteration {iteration}: Found {len(processes)} processes, Total: {total_memory:.1f} MB")
                
                # Show current processes
                for proc_name, memory in sorted(processes.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {proc_name}: {memory:.1f} MB")
                print()
            else:
                print(f"[{timestamp}] No ROS processes found")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\nStopping monitoring. Total measurements: {len(monitor.data)}")
        print(f"Data saved to: {output_file}")
        
        # Generate plots before exit
        if monitor.data:
            print("\nGenerating memory usage plots...")
            monitor.plot_memory_trends()
        
        return monitor


if __name__ == '__main__':
    monitor = continuous_monitoring(interval=15)