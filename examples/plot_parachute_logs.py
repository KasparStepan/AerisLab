"""
Standalone plotting script for existing simulation logs.

Useful for re-generating plots or creating custom visualizations
after simulation has completed.
"""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aerislab.visualization.plotting import (
    plot_trajectory_3d,
    plot_velocity_and_acceleration,
    plot_forces,
    compare_trajectories
)


def main():
    """Plot simulation results from CSV log."""
    parser = argparse.ArgumentParser(
        description="Generate plots from AerisLab simulation logs"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to simulation CSV file"
    )
    parser.add_argument(
        "--bodies",
        type=str,
        nargs="+",
        default=None,
        help="Body names to plot (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as CSV)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively"
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        plots_dir = Path(args.output_dir)
    else:
        plots_dir = csv_path.parent.parent / "plots"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get body names from CSV if not specified
    if args.bodies is None:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=1)
        # Extract unique body names from column headers
        bodies = set()
        for col in df.columns:
            if '.' in col:
                body_name = col.split('.')[0]
                bodies.add(body_name)
        args.bodies = sorted(bodies)
    
    print(f"Generating plots for: {', '.join(args.bodies)}")
    print(f"Output directory: {plots_dir}")
    
    # Generate plots for each body
    for body_name in args.bodies:
        print(f"\nProcessing {body_name}...")
        
        try:
            plot_trajectory_3d(
                str(csv_path),
                body_name,
                save_path=str(plots_dir / f"{body_name}_trajectory_3d.png"),
                show=args.show
            )
            print(f"  ✓ Trajectory 3D")
            
            plot_velocity_and_acceleration(
                str(csv_path),
                body_name,
                save_path=str(plots_dir / f"{body_name}_velocity_acceleration.png"),
                show=args.show,
                magnitude=False
            )
            print(f"  ✓ Velocity & Acceleration")
            
            plot_forces(
                str(csv_path),
                body_name,
                save_path=str(plots_dir / f"{body_name}_forces.png"),
                show=args.show,
                magnitude=False
            )
            print(f"  ✓ Forces")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nPlots saved to: {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
