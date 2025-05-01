import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def make_square_cloud(center=(0, 0), size=1.0, noise=0.01, num_points=100):
    """
    Create a square-shaped point cloud in 2D (for 2D ICP testing).
    """
    cx, cy = center
    points = []
    for _ in range(num_points // 4):
        points.append([cx - size / 2 + np.random.randn() * noise, cy + size / 2 + np.random.randn() * noise])
        points.append([cx + size / 2 + np.random.randn() * noise, cy - size / 2 + np.random.randn() * noise])
        points.append([cx - size / 2 + np.random.randn() * noise, cy - size / 2 + np.random.randn() * noise])
        points.append([cx + size / 2 + np.random.randn() * noise, cy + size / 2 + np.random.randn() * noise])
    return np.array(points)


def make_wireframe_cube(center=(0, 0, 0), size=1.0, noise=0.005):
    """
    Create a wireframe cube point cloud (only edges), centered at `center`.
    Returns Nx3 array.
    """
    cx, cy, cz = center
    s = size / 2

    # 8 corners of the cube
    corners = np.array([
        [cx - s, cy - s, cz - s],
        [cx - s, cy - s, cz + s],
        [cx - s, cy + s, cz - s],
        [cx - s, cy + s, cz + s],
        [cx + s, cy - s, cz - s],
        [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz - s],
        [cx + s, cy + s, cz + s]
    ])

    # edges = pairs of corner indices
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    points = []
    for start, end in edges:
        line = np.linspace(corners[start], corners[end], 20)
        if noise > 0:
            line += np.random.normal(scale=noise, size=line.shape)
        points.append(line)

    return np.vstack(points)

def apply_known_transform(points, dx=0.5, dy=0.3, dz=0.0, theta=None, rx=0.0, ry=0.0, rz=np.pi/12):
    """
    Apply a known rigid transform to 2D or 3D points.
    - If `theta` is provided, it's treated as a 2D rotation.
    - Otherwise, a 3D rotation is built from rx, ry, rz.
    """
    dim = points.shape[1]

    if dim == 2:
        if theta is None:
            theta = np.pi / 12
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        return (R @ points.T).T + np.array([dx, dy])

    elif dim == 3:
        def rot_z(t): return np.array([[np.cos(t), -np.sin(t), 0],
                                       [np.sin(t),  np.cos(t), 0],
                                       [0, 0, 1]])
        def rot_y(t): return np.array([[np.cos(t), 0, np.sin(t)],
                                       [0, 1, 0],
                                       [-np.sin(t), 0, np.cos(t)]])
        def rot_x(t): return np.array([[1, 0, 0],
                                       [0, np.cos(t), -np.sin(t)],
                                       [0, np.sin(t),  np.cos(t)]])

        R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
        return (R @ points.T).T + np.array([dx, dy, dz])

    else:
        raise ValueError("apply_known_transform only supports 2D or 3D points")

def visualize_icp_result(map_ref, scan, aligned):
    dim = map_ref.shape[1]

    if dim == 2:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.scatter(map_ref[:, 0], map_ref[:, 1], c='black', label='Map', s=10)
        plt.scatter(scan[:, 0], scan[:, 1], c='red', alpha=0.5, label='Original', s=10)
        plt.scatter(aligned[:, 0], aligned[:, 1], c='green', alpha=0.7, label='Aligned', s=10)
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.title("ICP Result (2D)")
        plt.tight_layout()
        plt.show()

    elif dim == 3:
        import plotly.graph_objects as go
        fig = go.Figure()

        def add_cloud(points, color, name):
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(size=3, color=color),
                name=name
            ))

        add_cloud(map_ref, 'black', 'Map')
        add_cloud(scan, 'red', 'Original')
        add_cloud(aligned, 'green', 'Aligned')

        fig.update_layout(
            title="ICP Result (3D)",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show()
    else:
        raise ValueError("Unsupported point dimensionality for visualization.")

