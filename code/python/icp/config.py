from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class ICPConfig:
    # 1. Map segmentation strategy
    segmentation_strategy: Literal["none", "even_points", "even_range", "uneven_points"] = "even_points"

    # 2. Gaussian filtering (percentage of points to drop)
    gaussian_filter_pct: float = 0.0  # e.g., 5.0 = 5% randomly removed

    # 3. Memory mode (reference map selection)
    memory_mode: Literal["full", "previous", "sliding"] = "full"

    # 4. Pair rejection criteria
    rejection: dict = field(default_factory=lambda: {
        "max_normal_diff": None,   # e.g., 0.5 (in radians or cosθ)
        "max_distance": None,      # e.g., 0.2
        "percentile_clip": None    # e.g., [95, 5] to keep 5–95% distances
    })

    # 5. Neighborhood search mode
    neighborhood_mode: Literal["fixed", "density_adaptive"] = "fixed"

    # 6. Fixed-radius search (used in 'fixed' mode)
    search_radius: float = 0.5

    # 7. Density scaling (used in 'density_adaptive' mode)
    density_factor: float = 1.0

    # Optional: number of segments for 3D map if segmentation applies
    num_segments: Optional[int] = 4
