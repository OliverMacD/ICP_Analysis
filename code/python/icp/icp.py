from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
from icp.config import ICPConfig


class ICPAlgorithm(ABC):
    def __init__(self, config: ICPConfig):
        self.config = config
        self.prev_map = None

    def _apply_gaussian_filter(self, points: np.ndarray) -> np.ndarray:
        pct = self.config.gaussian_filter_pct
        if pct <= 0.0 or pct >= 100.0:
            return points
        keep_count = int(len(points) * (1 - pct / 100))
        indices = np.random.choice(len(points), keep_count, replace=False)
        return points[indices]

    def _get_reference_map(self, full_map: np.ndarray) -> np.ndarray:
        mode = self.config.memory_mode
        if mode == "previous" and self.prev_map is not None:
            return self.prev_map
        elif mode == "sliding":
            return full_map[-len(full_map) // 2:]
        return full_map

    def _segment_map(self, map_points: np.ndarray) -> list[np.ndarray]:
        strategy = self.config.segmentation_strategy
        if strategy == "none" or map_points.shape[1] < 3:
            return [map_points]

        if strategy == "even_points":
            return np.array_split(map_points, self.config.num_segments)

        elif strategy == "even_range":
            z_min, z_max = np.min(map_points[:, 2]), np.max(map_points[:, 2])
            bins = np.linspace(z_min, z_max, self.config.num_segments + 1)
            segments = [map_points[(map_points[:, 2] >= bins[i]) & (map_points[:, 2] < bins[i + 1])]
                        for i in range(len(bins) - 1)]
            return [seg for seg in segments if len(seg) > 0]

        elif strategy == "uneven_points":
            counts = np.array_split(map_points, self.config.num_segments)
            return [seg for seg in counts if len(seg) > 0]

        return [map_points]

    def _find_neighbors(self, source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.config.neighborhood_mode == "fixed":
            nbrs = NearestNeighbors(radius=self.config.search_radius).fit(target)
            indices = nbrs.radius_neighbors(source, return_distance=False)
        else:
            k1 = NearestNeighbors(n_neighbors=2).fit(target)
            dists, _ = k1.kneighbors(target)
            avg_density = np.mean(dists[:, 1])
            radius = self.config.density_factor * avg_density
            nbrs = NearestNeighbors(radius=radius).fit(target)
            indices = nbrs.radius_neighbors(source, return_distance=False)

        matched_src = []
        matched_tgt = []

        for i, neighbors in enumerate(indices):
            if len(neighbors) > 0:
                for j in neighbors:
                    matched_src.append(source[i])
                    matched_tgt.append(target[j])

        return np.array(matched_src), np.array(matched_tgt)

    def _compute_normals(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Estimate normals via PCA on the k-nearest neighbors.
        Returns Nx2 array of unit normals.
        """
        if len(points) < k:
            raise ValueError("Not enough points for normal estimation")

        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        _, indices = nbrs.kneighbors(points)

        normals = np.zeros_like(points)
        for i, neighbors in enumerate(indices):
            neighbors = points[neighbors]
            cov = np.cov(neighbors.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]  # Smallest eigenvector
            normal /= np.linalg.norm(normal)
            normals[i] = normal

        return normals

    def _reject_pairs(self, src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rejection = self.config.rejection or {}
        keep = np.ones(len(src), dtype=bool)

        if rejection.get("max_distance") is not None:
            dists = np.linalg.norm(src - tgt, axis=1)
            keep &= dists < rejection["max_distance"]

        if rejection.get("percentile_clip") is not None:
            dists = np.linalg.norm(src - tgt, axis=1)
            high, low = rejection["percentile_clip"]
            lo_thres, hi_thres = np.percentile(dists, [low, high])
            keep &= (dists >= lo_thres) & (dists <= hi_thres)

        if rejection.get("max_normal_diff") is not None:
            normals_src = self._compute_normals(src)
            normals_tgt = self._compute_normals(tgt)
            cos_angles = np.sum(normals_src * normals_tgt, axis=1)
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            angle_diff = np.arccos(cos_angles)
            keep &= angle_diff < rejection["max_normal_diff"]

        return src[keep], tgt[keep]

    @abstractmethod
    def register(self, new_scan: np.ndarray, map_so_far: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass
