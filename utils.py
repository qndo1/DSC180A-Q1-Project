import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import ot
import scipy as sp
import matplotlib.pyplot as plt

def plot_3d_points_and_connections(points1, points2, G, switch_xz = True):
    """
    Given points1, points2, and G, plot the points and lines between matching points. If switch_xz is true then this will switch the x and z coordinates before plotting (since by default in the mocap data the x is the vertical axis).
    points1, points2: Nx3 arrays
    G: NxN array
    switch_xz: Boolean
    """
    if points1.shape[0] != points2.shape[0]:
        raise ValueError("Point clouds are not the same length")

    if G.shape[0] != G.shape[1]:
        raise ValueError("Matching matrix is not square")

    if G.shape[0] != points1.shape[0]:
        raise ValueError("Matching matrix dimensions don't match point cloud dimensions")

    if np.count_nonzero(G) > points1.shape[0]:
        raise ValueError("Matching has too many nonzero entries")

    if np.count_nonzero(G) < points1.shape[0]:
        raise ValueError("Matching has too few nonzero entries")

    if switch_xz:
        x_ind = 2
        z_ind = 0
    else:
        x_ind = 0
        z_ind = 2

    # Ensure numpy arrays
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    G = np.asarray(G)

    fig = go.Figure()

    # Plot first set of 3D points
    fig.add_trace(go.Scatter3d(
        x=points1[:, x_ind], y=points1[:, 1], z=points1[:, z_ind],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Points 1'
    ))

    # Plot second set of 3D points
    fig.add_trace(go.Scatter3d(
        x=points2[:, x_ind], y=points2[:, 1], z=points2[:, z_ind],
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
                    x=[p1[x_ind], p2[x_ind]],
                    y=[p1[1], p2[1]],
                    z=[p1[z_ind], p2[z_ind]],
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
        distance_matrix = np.array([np.zeros((n_source, n_source)), np.zeros((n_target, n_target))])
        count = -1
        for cp in [self.source, self.target]:
            count += 1
            n = cp.shape[0]
            for i in range(n):
                for j in range(n):
                    distance_matrix[count][i, j] = np.linalg.norm(cp[i] - cp[j])
        return distance_matrix[0], distance_matrix[1]


#Quy-Dzu work

"""
Computes W(i,j) as defined in the equation, given precomputed distance matrices.

We assume p = inf so all mappings are allowed.
"""

import numpy as np
from ot import wasserstein_1d, emd


def compute_W_matrix_distance_matrix_input(X_dists, Y_dists):
    """
    Computes W(i,j) for all i ∈ [n], j ∈ [m] as defined in the equation.

    X: array of shape (n, d)
    Y: array of shape (m, d)

    Returns: W matrix of shape (n, m)
    """
    n, _ = X_dists.shape
    m, _ = Y_dists.shape

    W = np.zeros((n, m))

    # Calculate D(i, j) which is the Wasserstein distance between the distributions of distances

    for i in range(n):
        Xi_distances = X_dists[i]  # vector of length n
        for j in range(m):
            Yj_distances = Y_dists[j]  # vector of length m

            # Wasserstein between the two empirical distributions
            W[i, j] = wasserstein_1d(Xi_distances, Yj_distances)

    map_matrix = emd(np.ones(n) / n, np.ones(m) / m, W)

    return W, map_matrix

def compute_W_matrix(X, Y):
    """
    Computes W(i,j) for all i ∈ [n], j ∈ [m] as defined in the equation.

    X: array of shape (n, d)
    Y: array of shape (m, d)

    Returns: W matrix of shape (n, m)
    """
    n, _ = X.shape
    m, _ = Y.shape

    # Precompute all intra-set distances ||X_i - X_ℓ|| and ||Y_j - Y_ℓ||
    X_dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)  # shape (n, n)
    Y_dists = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)  # shape (m, m)

    W = np.zeros((n, m))

    # Calculate D(i, j) which is the Wasserstein distance between the distributions of distances

    for i in range(n):
        Xi_distances = X_dists[i]  # vector of length n
        for j in range(m):
            Yj_distances = Y_dists[j]  # vector of length m

            # Wasserstein between the two empirical distributions
            W[i, j] = wasserstein_1d(Xi_distances, Yj_distances)

    map_matrix = emd(np.ones(n) / n, np.ones(m) / m, W)

    return W, map_matrix

