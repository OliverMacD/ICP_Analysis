import json
import math
import matplotlib.pyplot as plt
from typing import Tuple, List

class Environment:
    def __init__(self, env_path: str = "environment.json"):
        with open(env_path, "r") as f:
            data = json.load(f)
        
        self.size = tuple(data["environment"]["size"])
        self.origin = data["environment"]["origin"]
        self.walls = data["walls"]
        self.obstacles = data["obstacles"]

    def check_collision(self, pos: Tuple[float, float], size: List[float]) -> bool:
        x, y = pos

        # Check boundary
        if x < 0 or x > self.size[0] or y < 0 or y > self.size[1]:
            return True

        # Check circular obstacles
        for obs in self.obstacles:
            ox, oy = obs["position"]
            ox += 0.5
            oy += 0.5
            r = obs.get("size", [1])[0] * 0.4
            if math.hypot(x - ox, y - oy) < r + size[0] * 0.5:
                return True

        # Check wall "hitboxes" (simplified as padded lines)
        for wall in self.walls:
            (x1, y1), (x2, y2) = wall["start"], wall["end"]
            dist = self.point_line_distance((x, y), (x1, y1), (x2, y2))
            if dist < size[0] * 0.4:  # padding around robot radius
                return True

        return False

    @staticmethod
    def point_line_distance(p, a, b):
        # Distance from point p to line segment ab
        px, py = p
        ax, ay = a
        bx, by = b

        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab_len_sq = abx**2 + aby**2

        if ab_len_sq == 0:
            return math.hypot(px - ax, py - ay)

        t = max(0, min(1, (apx * abx + apy * aby) / ab_len_sq))
        closest = (ax + t * abx, ay + t * aby)
        return math.hypot(px - closest[0], py - closest[1])


    def plot(self, ax=None, show=True, draw_walls=True, draw_obstacles=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')

        if draw_walls:
            for wall in self.walls:
                (x1, y1), (x2, y2) = wall["start"], wall["end"]
                ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)

        if draw_obstacles:
            for obs in self.obstacles:
                ox, oy = obs["position"]
                r = obs.get("size", [1])[0] * 0.4
                ax.add_patch(plt.Circle((ox + 0.5, oy + 0.5), radius=r, color="red"))

        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])
        ax.grid(True)
        if show:
            plt.show()



if __name__ == "__main__":
    env = Environment()
    env.plot()
