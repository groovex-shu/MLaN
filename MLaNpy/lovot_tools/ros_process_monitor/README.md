# ROS Monitoring Tools

This directory contains standalone ROS monitoring utilities.

## 

A continuous ROS process memory monitoring tool that tracks memory usage.

### Usage

Run directly as a standalone script:

```sh
cp single_node_memory_monitor.py /home/dev/ # at lovot

python single_node_memory_monitor.py
```

### Features

- Monitors ROS processes, run_slam, lovot_localization services, and tf processes
- Saves memory usage data to CSV files
- Generates memory trend plots (PNG format)
- Runs continuously until stopped with Ctrl+C

### Output Files

- `ros_memory_timeseries.csv` - Time series data with memory usage
- `ros_memory_plot.png` - Memory usage trend visualization

### Dependencies

```bash
# activate appropriate env
```
include:
- matplotlib
- subprocess (built-in)
- csv (built-in) 
- datetime (built-in)