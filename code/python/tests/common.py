import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

def apply_known_transform(points, dx=0.5, dy=0.3, dz=0.2, rx=0.0, ry=0.0, rz=np.pi/12):
    """
    Apply a known rigid 3D transform (translation + rotation).
    """
    def rot_z(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [0, 0, 1]])

    def rot_y(theta):
        return np.array([[ np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])

    def rot_x(theta):
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta),  np.cos(theta)]])

    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    return (R @ points.T).T + np.array([dx, dy, dz])

def visualize_icp_result(map_ref, scan, aligned):
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
