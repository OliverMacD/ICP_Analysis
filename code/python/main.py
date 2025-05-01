import argparse
import subprocess
import sys
from pathlib import Path

def run_simulation(args):
    sim_path = Path(__file__).parent / "simulation" / "simulation_runner.py"
    cmd = [sys.executable, str(sim_path)]

    # Pass relevant flags
    if args.show_walls:
        cmd.append("--show-walls")
    if args.show_obstacles:
        cmd.append("--show-obstacles")
    if args.icp:
        cmd.extend(["--icp", args.icp])

    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="ICP Analysis Entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run the Lidar simulation")
    sim_parser.add_argument("--show-walls", action="store_true", help="Show wall map")
    sim_parser.add_argument("--show-obstacles", action="store_true", help="Show obstacle map")
    sim_parser.add_argument("--icp", choices=["point_to_point", "point_to_plane", "generalized_icp"],
                            help="Enable ICP with specified method")

    args = parser.parse_args()

    if args.command == "simulate":
        run_simulation(args)

if __name__ == "__main__":
    main()

