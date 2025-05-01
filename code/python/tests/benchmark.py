import os
import json
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from itertools import product

@dataclass
class ICPConfig:
    segmentation_strategy: Literal["none", "even_points", "even_range", "uneven_points"] = "even_points"
    gaussian_filter_pct: float = 0.0
    memory_mode: Literal["full", "previous", "sliding"] = "full"
    rejection: dict = field(default_factory=lambda: {
        "max_normal_diff": None,
        "max_distance": None,
        "percentile_clip": None
    })
    neighborhood_mode: Literal["fixed", "density_adaptive"] = "fixed"
    search_radius: float = 0.5
    density_factor: float = 1.0
    num_segments: Optional[int] = 4

# Grid parameters
segmentation_strategies = ["none", "even_points", "even_range", "uneven_points"]
gaussian_filter_pcts = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
memory_modes = ["full", "previous", "sliding"]
neighborhood_modes = ["fixed", "density_adaptive"]
search_radii = [0.1, 0.3, 0.5, 0.75, 1.0]
density_factors = [0.5, 1.0, 2.0, 3.0, 4.0]
num_segments = [2, 4, 6, 8]
rejection_configs = [
    {"max_normal_diff": None, "max_distance": None, "percentile_clip": None},
    {"max_normal_diff": 0.1, "max_distance": 0.3, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.25, "max_distance": 0.3, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.5, "max_distance": 0.3, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.75, "max_distance": 0.3, "percentile_clip": [95, 5]},
    {"max_normal_diff": 1.0, "max_distance": 0.3, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.1, "max_distance": 0.5, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.25, "max_distance": 0.5, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.5, "max_distance": 0.5, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.75, "max_distance": 0.5, "percentile_clip": [95, 5]},
    {"max_normal_diff": 1.0, "max_distance": 0.5, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.1, "max_distance": 0.75, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.25, "max_distance": 0.75, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.5, "max_distance": 0.75, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.75, "max_distance": 0.75, "percentile_clip": [95, 5]},
    {"max_normal_diff": 1.0, "max_distance": 0.75, "percentile_clip": [95, 5]},
    {"max_normal_diff": 0.1, "max_distance": 0.3, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.25, "max_distance": 0.3, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.5, "max_distance": 0.3, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.75, "max_distance": 0.3, "percentile_clip": [90, 10]},
    {"max_normal_diff": 1.0, "max_distance": 0.3, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.1, "max_distance": 0.5, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.25, "max_distance": 0.5, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.5, "max_distance": 0.5, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.75, "max_distance": 0.5, "percentile_clip": [90, 10]},
    {"max_normal_diff": 1.0, "max_distance": 0.5, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.1, "max_distance": 0.75, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.25, "max_distance": 0.75, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.5, "max_distance": 0.75, "percentile_clip": [90, 10]},
    {"max_normal_diff": 0.75, "max_distance": 0.75, "percentile_clip": [90, 10]},
    {"max_normal_diff": 1.0, "max_distance": 0.75, "percentile_clip": [90, 10]},
    
]

def generate_all_configs():
    configs = []
    for seg, gf, mem, neigh, sr, df, rej, ns in product(
        segmentation_strategies,
        gaussian_filter_pcts,
        memory_modes,
        neighborhood_modes,
        search_radii,
        density_factors,
        rejection_configs,
        num_segments,
    ):
        if neigh != "density_adaptive" and df != 1.0:
            continue
        if seg == "none" and ns != 4:
            continue

        config = ICPConfig(
            segmentation_strategy=seg,
            gaussian_filter_pct=gf,
            memory_mode=mem,
            rejection=rej,
            neighborhood_mode=neigh,
            search_radius=sr,
            density_factor=df,
            num_segments=ns
        )
        configs.append(config)
    return configs

if __name__ == "__main__":
    all_configs = generate_all_configs()
    os.makedirs("config_sets", exist_ok=True)
    with open("config_sets/icp_config_grid.json", "w") as f:
        json.dump([asdict(cfg) for cfg in all_configs], f, indent=2)
    print(f"Saved {len(all_configs)} configs to config_sets/icp_config_grid.json")
