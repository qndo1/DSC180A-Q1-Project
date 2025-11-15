import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import ot
import scipy as sp
import matplotlib.pyplot as plt

def plot_3d_points_and_connections(points1, points2, G):
    # Ensure numpy arrays
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    G = np.asarray(G)

    fig = go.Figure()

    # Plot first set of 3D points
    fig.add_trace(go.Scatter3d(
        x=points1[:, 0], y=points1[:, 1], z=points1[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Points 1'
    ))

    # Plot second set of 3D points
    fig.add_trace(go.Scatter3d(
        x=points2[:, 0], y=points2[:, 1], z=points2[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Points 2'
    ))

    # Draw connections for nonzero G[i, j]
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[i, j] != 0:
                p1 = points1[i]
                p2 = points2[j]
                fig.add_trace(go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ))

    # Layout styling
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title='3D Points with Connections',
        template='plotly_white'
    )

    return fig

def compute_gw_and_plot(xs, xt):
    """
    Computes the GW plan between two points clouds and plots the results.
    """
    p = np.ones(xs.shape[0]) / xs.shape[0]
    q = np.ones(xt.shape[0]) / xt.shape[0]

    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    G0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, log = True, verbose = True)

    fig = plot_3d_points_and_connections(xt, xs, G0)
    print("hi")
    return fig, G0


# ----------------------------------------------------
# Takafumi's work
# one txt file -> 2 point clouds -> distance profiles
# example code
# lcp = LoadCloudPoint()
# source_pc, target_pc = lcp.get_two_random_point_cloud()
# dp = DistanceProfile(source_pc, target_pc)
# distance_matrix = dp.compute_L2_matrix()
# print(distance_matrix)
# ----------------------------------------------------

class LoadCloudPoint:
    def __init__(self, filepath=None):
        if filepath == None:
            csv_dir = Path("datasets/csv_files")
            csv_list = sorted(csv_dir.glob("*.csv"))
            filepath = np.random.choice(csv_list)
        else:
            pass

        self.filepath = Path(filepath)
        self.point_cloud = pd.read_csv(filepath).to_numpy()

    def get_two_random_point_cloud(self):
        idx_1 = np.random.choice(self.point_cloud.shape[0]//2)
        idx_2 = np.random.choice(self.point_cloud.shape[0]//2) + self.point_cloud.shape[0]//2
        source = self.point_cloud[idx_1].reshape(-1,3)
        target = self.point_cloud[idx_2].reshape(-1,3)
        return source, target

class DistanceProfile:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def compute_L2_matrix(self):
        n_source = self.source.shape[0]
        n_target = self.target.shape[0]
        distance_matrix = np.zeros((n_source, n_target))
        for i in range(n_source):
            for j in range(n_target):
                distance_matrix[i, j] = np.linalg.norm(self.source[i] - self.target[j], ord=2)
        return distance_matrix
