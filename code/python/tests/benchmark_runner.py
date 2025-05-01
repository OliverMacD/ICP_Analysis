import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from tqdm import tqdm

from icp.point_to_point import icp as icp_ptp
from icp.point_to_plane import icp as icp_ptpl
from icp.generalized_icp import icp as icp_gicp
from icp.config import ICPConfig
from tests.common import make_square_cloud, make_wireframe_cube, apply_known_transform

# === Load prebuilt grid of configs ===
with open("config_sets/icp_config_grid.json", "r") as f:
    config_dicts = json.load(f)

# === Setup output folder ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_base = f"benchmark_results_{timestamp}"
os.makedirs(output_base, exist_ok=True)

# === Define ICP methods ===
methods = {
    "point_to_point": icp_ptp,
    "point_to_plane": icp_ptpl,
    "generalized_icp": icp_gicp,
}

def generate_clouds(dim):
    if dim == "2d":
        map_ref = make_square_cloud(center=(0, 0), size=2.0)
        scan = apply_known_transform(map_ref, dx=0.3, dy=0.2, theta=np.pi / 18)
    else:
        map_ref = make_wireframe_cube(center=(0, 0, 0), size=2.0)
        scan = apply_known_transform(map_ref, dx=0.3, dy=0.2, dz=0.1, rz=np.pi / 18)
    return scan, map_ref

def save_plot_2d(scan, map_ref, aligned, path):
    plt.figure(figsize=(6, 6))
    plt.scatter(map_ref[:, 0], map_ref[:, 1], c='black', label='Map', s=10)
    plt.scatter(scan[:, 0], scan[:, 1], c='red', alpha=0.5, label='Scan', s=10)
    plt.scatter(aligned[:, 0], aligned[:, 1], c='green', alpha=0.7, label='Aligned', s=10)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.title("ICP Result (2D)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_plot_3d(scan, map_ref, aligned, path):
    fig = go.Figure()
    def add_cloud(points, color, name):
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color=color),
            name=name
        ))
    add_cloud(map_ref, 'black', 'Map')
    add_cloud(scan, 'red', 'Scan')
    add_cloud(aligned, 'green', 'Aligned')
    fig.update_layout(
        title="ICP Result (3D)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.write_image(path)

# === Run benchmark ===
print("Starting benchmark...")
total_tests = len(methods) * len(config_dicts) * 2
results_log = []

with tqdm(total=total_tests, desc="Running ICP Benchmarks") as pbar:
    for method_name, icp_func in methods.items():
        for i, cfg_dict in enumerate(config_dicts):
            config = ICPConfig(**cfg_dict)
            for dim in ["2d", "3d"]:
                start = time.time()
                result = {
                    "method": method_name,
                    "dimension": dim,
                    "test_id": f"test_{i}",
                    "rmse_error": None,
                    "duration_sec": None,
                    "success": False,
                    "error_message": None,
                    "segmentation_strategy": cfg_dict["segmentation_strategy"],
                    "gaussian_filter_pct": cfg_dict["gaussian_filter_pct"],
                    "memory_mode": cfg_dict["memory_mode"],
                    "neighborhood_mode": cfg_dict["neighborhood_mode"],
                    "search_radius": cfg_dict["search_radius"],
                    "density_factor": cfg_dict["density_factor"],
                    "num_segments": cfg_dict["num_segments"],
                    "rejection.max_distance": cfg_dict["rejection"].get("max_distance"),
                    "rejection.max_normal_diff": cfg_dict["rejection"].get("max_normal_diff"),
                    "rejection.percentile_clip": str(cfg_dict["rejection"].get("percentile_clip")),
                }

                outdir = os.path.join(output_base, method_name, dim, f"test_{i}")
                os.makedirs(outdir, exist_ok=True)
                try:
                    scan, map_ref = generate_clouds(dim)
                    aligned, _ = icp_func(scan, map_ref, config=config)
                    duration = time.time() - start
                    error = float(np.mean(np.linalg.norm(aligned - scan[:len(aligned)], axis=1)))

                    result["rmse_error"] = error
                    result["duration_sec"] = round(duration, 4)
                    result["success"] = True

                    with open(os.path.join(outdir, "config.json"), "w") as f:
                        json.dump(cfg_dict, f, indent=2)

                    with open(os.path.join(outdir, "metrics.json"), "w") as f:
                        json.dump({"rmse_error": error, "duration_sec": duration}, f, indent=2)

                    plot_path = os.path.join(outdir, "plot.png")
                    if dim == "2d":
                        save_plot_2d(scan, map_ref, aligned, plot_path)
                    # else:
                    #     save_plot_3d(scan, map_ref, aligned, plot_path)

                    pbar.set_postfix_str(f"{method_name} {dim} #{i} ✓ err={error:.4f} time={duration:.2f}s")
                except Exception as e:
                    result["error_message"] = str(e)
                    with open(os.path.join(outdir, "error.txt"), "w") as f:
                        f.write(str(e))
                    pbar.set_postfix_str(f"{method_name} {dim} #{i} ✗ ERROR")

                results_log.append(result)
                pbar.update(1)

# === Save summary CSV ===
summary_csv_path = os.path.join(output_base, "summary.csv")
with open(summary_csv_path, mode="w", newline="") as csvfile:
    fieldnames = list(results_log[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results_log:
        writer.writerow(row)

print(f"\nBenchmark complete. Summary saved to: {summary_csv_path}")
