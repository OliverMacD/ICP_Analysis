import numpy as np
import math
import json
from typing import List, Tuple

def transform_points(points: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
    """
    Transform points from robot frame to world frame using pose (x, y, theta).
    """
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    transformed = np.dot(points, rot.T) + np.array([x, y])
    return transformed

class LidarScan:
    def __init__(self, pose: Tuple[float, float, float], points: List[Tuple[float, float]]):
        self.pose = pose  # (x, y, theta)
        self.points = np.array(points)  # In robot frame

    def get_world_points(self) -> np.ndarray:
        """
        Get the scan points transformed into the world frame.
        """
        return transform_points(self.points, self.pose)

    def to_dict(self) -> dict:
        return {
            "pose": self.pose,
            "points": self.points.tolist()
        }

    @staticmethod
    def from_dict(data: dict):
        return LidarScan(tuple(data["pose"]), data["points"])


class PointCloudLog:
    def __init__(self):
        self.scans: List[LidarScan] = []

    def add_scan(self, pose: Tuple[float, float, float], points: List[Tuple[float, float]]):
        self.scans.append(LidarScan(pose, points))

    def get_all_world_points(self) -> np.ndarray:
        """
        Get all scan points transformed to world coordinates.
        """
        all_points = []
        for scan in self.scans:
            all_points.append(scan.get_world_points())
        return np.vstack(all_points) if all_points else np.empty((0, 2))

    def save_to_file(self, path: str):
        with open(path, 'w') as f:
            json.dump([scan.to_dict() for scan in self.scans], f, indent=2)

    def load_from_file(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.scans = [LidarScan.from_dict(scan_data) for scan_data in data]

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        return self.scans[idx]
