"""
Feature Map Vertices Movement Analysis Tool

Analyzes how vertices moved between two feature maps and generates visualization plots.
"""

import argparse
import sys
from pathlib import Path

# Import the analysis functions
from lovot_slam.feature_map.feature_map_vertices import (analyze_vertex_movements, 
                                                         plot_vertex_movements)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze vertex movements between two feature maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "original_map", 
        type=str,
        help="Path to the original (older) map directory"
    )
    
    parser.add_argument(
        "destination_map", 
        type=str,
        help="Path to the destination (newer) map directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="vertex_movements.csv",
        help="Output CSV file path for movement data"
    )
    
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )
    
    parser.add_argument(
        "--top-movers", "-t",
        type=int,
        default=10,
        help="Number of top movers to display"
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects and validate
    original_path = Path(args.original_map)
    destination_path = Path(args.destination_map)
    
    if not original_path.exists():
        print(f"Error: Original map path does not exist: {original_path}")
        sys.exit(1)
        
    if not destination_path.exists():
        print(f"Error: Destination map path does not exist: {destination_path}")
        sys.exit(1)
    
    print(f"Analyzing vertex movements...")
    print(f"Original map: {original_path}")
    print(f"Destination map: {destination_path}")
    print("-" * 60)
    
    # Analyze movements
    movements_df = analyze_vertex_movements(str(original_path), str(destination_path))
    
    if movements_df is None:
        print("Error: Could not analyze vertex movements")
        print("Check that both map directories contain valid feature map data")
        sys.exit(1)
    
    if len(movements_df) == 0:
        print("Warning: No matching vertices found between maps")
        print("This could mean:")
        print("- No common missions between maps")
        print("- No vertices could be matched by timestamp")
        sys.exit(1)
    
    # Save to CSV
    output_path = Path(args.output)
    if output_path.parent.exists():
        with open(output_path, "w") as f:
            movements_df.to_csv(f, index=False)
        print(f"Saved movement data to: {output_path}")
    else:
        print(f"Error: Output directory does not exist: {output_path.parent}")
        sys.exit(1)
        
    
    # Display summary
    print("\n=== VERTEX MOVEMENT SUMMARY ===")
    print(f"Total vertices analyzed: {len(movements_df)}")
    print(f"Average translation distance: {movements_df['distance'].mean():.4f} m")
    print(f"Max translation distance: {movements_df['distance'].max():.4f} m")
    print(f"Average rotation change: {movements_df['rotation_change'].mean():.2f}°")
    print(f"Max rotation change: {movements_df['rotation_change'].max():.2f}°")
    
    # Show top movers
    if args.top_movers > 0:
        print(f"\n=== TOP {args.top_movers} LARGEST MOVEMENTS ===")
        top_movements = movements_df.nlargest(args.top_movers, 'distance')
        print(top_movements[['mission_id', 'timestamp', 'distance', 'rotation_change']].to_string(index=False))
    
    # Generate plots
    if not args.no_plot:
        print(f"\nGenerating visualization plots...")
        try:
            plot_vertex_movements(movements_df)
            print("Plots displayed successfully")
        except Exception as e:
            print(f"Error generating plots: {e}")
            print("You may need to install matplotlib: pip install matplotlib")
    
    print(f"\nAnalysis complete!")


if __name__ == "__main__":
    main()