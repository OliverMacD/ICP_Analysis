import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import sys
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import importlib

from environment import Environment
from robot import Robot
from point_cloud import PointCloudLog
from utils.transform import apply_transform

# Config
LIDAR_NUM_BEAMS = 720
LIDAR_RANGE = 30.0
MOVE_STEP = 0.5
TURN_STEP = np.pi / 16

# Runtime toggles
show_walls = True
show_obstacles = True
icp_enabled = False
scan_map = None

# CLI args
parser = argparse.ArgumentParser(description="2D Lidar Simulation Runner")
parser.add_argument("--show-walls", action="store_true", help="Display environment walls")
parser.add_argument("--show-obstacles", action="store_true", help="Display circular obstacles")
parser.add_argument("--icp", type=str, default=None,
                    choices=["point_to_point", "point_to_plane", "generalized_icp"],
                    help="Enable ICP mapping with specified algorithm")
args = parser.parse_args()
show_walls = args.show_walls
show_obstacles = args.show_obstacles

# Dynamically import ICP function
ICP_MODULES = {
    "point_to_point": "icp.point_to_point",
    "point_to_plane": "icp.point_to_plane",
    "generalized_icp": "icp.generalized_icp"
}
icp_fn = None
if args.icp:
    icp_mod = importlib.import_module(ICP_MODULES[args.icp])
    icp_fn = icp_mod.icp
    print(f"Using ICP algorithm: {args.icp}")


def draw_simulation(env, robot, lidar_points, goal=None,
                    show_walls=True, show_obstacles=True,
                    scan_map=None, icp_enabled=False):
    plt.clf()
    ax = plt.gca()

    if show_walls or show_obstacles:
        env.plot(ax=ax, show=False, draw_walls=show_walls, draw_obstacles=show_obstacles)

    # Draw robot
    x, y = robot.position
    ax.add_patch(patches.Circle((x, y), 0.3, color="green"))

    # Draw orientation
    dx = 0.5 * np.cos(robot.orientation)
    dy = 0.5 * np.sin(robot.orientation)
    ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, color='black')

    # Draw lidar points
    if lidar_points:
        xs, ys = zip(*[(
            x + np.cos(robot.orientation) * px - np.sin(robot.orientation) * py,
            y + np.sin(robot.orientation) * px + np.cos(robot.orientation) * py
        ) for px, py in lidar_points])
        ax.scatter(xs, ys, c='blue', s=10)

    # Draw ICP map
    if icp_enabled and scan_map is not None and len(scan_map) > 0:
        mx, my = zip(*scan_map)
        ax.scatter(mx, my, c='purple', s=5, alpha=0.5)

    if goal:
        ax.add_patch(patches.Circle(goal, 0.3, color="purple", alpha=0.5))

    ax.set_xlim(0, env.size[0])
    ax.set_ylim(0, env.size[1])
    ax.set_title("2D Lidar Simulation")
    plt.pause(0.01)


def move_toward_goal(robot, goal, env):
    path = []
    pos = np.array(robot.position)
    goal = np.array(goal)

    for _ in range(100):
        delta = goal - pos
        if np.linalg.norm(delta) < 0.5:
            break
        step = MOVE_STEP * delta / np.linalg.norm(delta)
        candidate = pos + step
        if not env.check_collision(tuple(candidate), robot.size):
            path.append(tuple(candidate))
            pos = candidate
        else:
            perp = np.array([-step[1], step[0]])
            candidate = pos + perp
            if not env.check_collision(tuple(candidate), robot.size):
                path.append(tuple(candidate))
                pos = candidate
            else:
                break
    return path


def manual_mode(env, robot, pclog):
    global show_walls, show_obstacles, icp_enabled, scan_map
    print("Manual Mode: use W/A/S/D to move, Q/E to rotate, Z/X to toggle walls/obstacles, M to toggle ICP map, C to exit.")

    if sys.platform == 'win32':
        import msvcrt
        def get_key():
            return msvcrt.getch().decode('utf-8').upper() if msvcrt.kbhit() else None
    else:
        import termios, tty, select
        def get_key():
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if dr:
                return sys.stdin.read(1).upper()
            return None

        old_settings = None
        if sys.platform != 'win32':
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            key = get_key()

            if key == "W":
                robot.move(MOVE_STEP * np.cos(robot.orientation),
                           MOVE_STEP * np.sin(robot.orientation), 0)
            elif key == "S":
                robot.move(-MOVE_STEP * np.cos(robot.orientation),
                           -MOVE_STEP * np.sin(robot.orientation), 0)
            elif key == "A":
                robot.move(-MOVE_STEP * np.sin(robot.orientation),
                            MOVE_STEP * np.cos(robot.orientation), 0)
            elif key == "D":
                robot.move(MOVE_STEP * np.sin(robot.orientation),
                           -MOVE_STEP * np.cos(robot.orientation), 0)
            elif key == "Q":
                robot.move(0, 0, TURN_STEP)
            elif key == "E":
                robot.move(0, 0, -TURN_STEP)
            elif key == "Z":
                show_walls = not show_walls
                print(f"[toggle] show_walls = {show_walls}")
            elif key == "X":
                show_obstacles = not show_obstacles
                print(f"[toggle] show_obstacles = {show_obstacles}")
            elif key == "M":
                icp_enabled = not icp_enabled
                print(f"[toggle] icp_enabled = {icp_enabled}")
            elif key == "C":
                print("Exiting...")
                break

            # Lidar + ICP
            lidar = robot.simulate_lidar_scan(num_beams=LIDAR_NUM_BEAMS, max_range=LIDAR_RANGE)
            pclog.add_scan(robot.position + (robot.orientation,), lidar)

            if icp_fn:
                if scan_map is None:
                    scan_map = np.array(lidar)
                else:
                    scan_np = np.array(lidar)
                    aligned_scan, updated_map = icp_fn(scan_np, scan_map)
                    scan_map = updated_map

            draw_simulation(env, robot, lidar,
                            show_walls=show_walls,
                            show_obstacles=show_obstacles,
                            scan_map=scan_map,
                            icp_enabled=icp_enabled)

            time.sleep(0.01)
    finally:
        if sys.platform != 'win32' and old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def goal_mode(env, robot, pclog):
    global scan_map
    gx, gy = map(float, input("Enter goal coordinates (x y): ").strip().split())
    path = move_toward_goal(robot, (gx, gy), env)

    for pos in path:
        robot.set_pose(pos, robot.orientation)
        lidar = robot.simulate_lidar_scan(num_beams=LIDAR_NUM_BEAMS, max_range=LIDAR_RANGE)
        pclog.add_scan(robot.position + (robot.orientation,), lidar)

        if icp_fn:
            if scan_map is None:
                scan_map = np.array(lidar)
            else:
                scan_np = np.array(lidar)
                aligned_scan, updated_map = icp_fn(scan_np, scan_map)
                scan_map = updated_map

        draw_simulation(env, robot, lidar,
                        goal=(gx, gy),
                        show_walls=show_walls,
                        show_obstacles=show_obstacles,
                        scan_map=scan_map,
                        icp_enabled=icp_enabled)
        time.sleep(0.2)


def main():
    env = Environment()
    with open("robot.json", "r") as f:
        robot_config = json.load(f)
    robot = Robot(robot_config, env)
    pclog = PointCloudLog()

    print("Simulation Started.")
    print("Choose Mode:")
    print("1. Manual (WASD)")
    print("2. Goal [x y]")

    mode = input("Enter 1 or 2: ").strip()
    plt.ion()
    plt.figure(figsize=(8, 8))

    if mode == "1":
        manual_mode(env, robot, pclog)
    elif mode == "2":
        goal_mode(env, robot, pclog)

    pclog.save_to_file("scans.json")
    print("Simulation complete. Point cloud saved to 'scans.json'.")


if __name__ == "__main__":
    main()
