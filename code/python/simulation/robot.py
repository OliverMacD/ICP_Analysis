import math
from typing import Tuple, List, Dict
import numpy as np

class Robot:
    def __init__(self, config: Dict, environment):
        self.shape = config["shape"]
        self.size = config["size"]
        self.position = tuple(config["starting_position"])
        self.orientation = config["starting_orientation"]  # radians
        self.environment = environment

    def set_pose(self, position: Tuple[float, float], orientation: float):
        self.position = position
        self.orientation = orientation

    def move(self, dx: float, dy: float, dtheta: float):
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        new_theta = self.orientation + dtheta
        if not self.environment.check_collision((new_x, new_y), self.size):
            self.position = (new_x, new_y)
            self.orientation = new_theta

    def simulate_lidar_scan(
        self,
        angle_range: Tuple[float, float] = (-math.pi, math.pi),
        num_beams: int = 360,
        max_range: float = 30.0
    ) -> List[Tuple[float, float]]:
        """
        Simulate 2D Lidar by casting rays from the robot within the given angle range.
        Returns list of points in robot-local frame.
        """
        points = []
        x, y = self.position
        start_angle = angle_range[0]
        end_angle = angle_range[1]

        for i in range(num_beams):
            angle = self.orientation + start_angle + (end_angle - start_angle) * i / (num_beams - 1)
            hit = self.cast_ray(x, y, angle, max_range)
            if hit:
                # Convert to robot-local coordinates
                dx = hit[0] - x
                dy = hit[1] - y
                local_x = math.cos(-self.orientation) * dx - math.sin(-self.orientation) * dy
                local_y = math.sin(-self.orientation) * dx + math.cos(-self.orientation) * dy
                points.append((local_x, local_y))
        return points

    def cast_ray(self, x: float, y: float, angle: float, max_range: float) -> Tuple[float, float]:
        """
        Cast a ray and find intersection with walls or circular obstacles.
        Returns the point of first collision or endpoint at max range.
        """
        ray_end = (
            x + max_range * math.cos(angle),
            y + max_range * math.sin(angle)
        )

        closest_intersection = None
        min_dist = float("inf")

        # Check wall intersections
        for wall in self.environment.walls:
            intersect = self.ray_segment_intersect((x, y), ray_end, wall["start"], wall["end"])
            if intersect:
                dist = math.hypot(intersect[0] - x, intersect[1] - y)
                if dist < min_dist:
                    closest_intersection = intersect
                    min_dist = dist

        # Check circular obstacle intersections
        for obs in self.environment.obstacles:
            ox, oy = obs["position"]
            ox += 0.5  # Treat center of grid cell as obstacle center
            oy += 0.5
            r = obs.get("size", [1])[0] * 0.4
            hit = self.ray_circle_intersect((x, y), (math.cos(angle), math.sin(angle)), (ox, oy), r)
            if hit:
                dist = math.hypot(hit[0] - x, hit[1] - y)
                if dist < min_dist:
                    closest_intersection = hit
                    min_dist = dist

        return closest_intersection if closest_intersection else ray_end

    @staticmethod
    def ray_segment_intersect(p1, p2, q1, q2):
        """
        Compute intersection of ray (p1 to p2) and segment (q1 to q2).
        Returns intersection point or None.
        """
        def to_vec(a, b):
            return b[0] - a[0], b[1] - a[1]

        def cross(v, w):
            return v[0] * w[1] - v[1] * w[0]

        r = to_vec(p1, p2)
        s = to_vec(q1, q2)
        denom = cross(r, s)

        if denom == 0:
            return None  # Parallel

        qp = to_vec(p1, q1)
        t = cross(qp, s) / denom
        u = cross(qp, r) / denom

        if t >= 0 and 0 <= u <= 1:
            return p1[0] + t * r[0], p1[1] + t * r[1]
        else:
            return None

    @staticmethod
    def ray_circle_intersect(ray_origin, ray_dir, circle_center, radius):
        """
        Returns the point of intersection between a ray and a circle, or None if no hit.
        """
        ox, oy = ray_origin
        dx, dy = ray_dir
        cx, cy = circle_center

        fx, fy = ox - cx, oy - cy

        a = dx ** 2 + dy ** 2
        b = 2 * (fx * dx + fy * dy)
        c = fx ** 2 + fy ** 2 - radius ** 2

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # We want the smallest positive t
        for t in sorted([t1, t2]):
            if t >= 0:
                return ox + dx * t, oy + dy * t
        return None
