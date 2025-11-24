import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import ot
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import heapq
import numpy as np
from ot import wasserstein_1d, emd

# Matt work

def plot_3d_points_and_connections(points1, points2, G, switch_xz = True, color_incorrect = False):
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
                c = "gray"
                if color_incorrect and i != j:
                    c = "red"
                p1 = points1[i]
                p2 = points2[j]
                fig.add_trace(go.Scatter3d(
                    x=[p1[x_ind], p2[x_ind]],
                    y=[p1[1], p2[1]],
                    z=[p1[z_ind], p2[z_ind]],
                    mode='lines',
                    line=dict(color=c, width=2),
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

def animate_point_cloud_matches(points1_list, points2_list, G_list, switch_xz=True, color_incorrect=False):
    """
    Create a Plotly animation where each frame shows two point clouds and
    the matchings between them.

    points1_list, points2_list: lists of length N, each element is an Mx3 array
    G_list: list of length N, each element is an MxM array
    switch_xz: swap x,z axes for visualization
    color_incorrect: highlight incorrect matches (i != j) in red
    """
    N = len(points1_list)
    if not (len(points2_list) == len(G_list) == N):
        raise ValueError("points1_list, points2_list, and G_list must have same length")

    # Axis swapping logic
    if switch_xz:
        x_ind, z_ind = 2, 0
    else:
        x_ind, z_ind = 0, 2

    # Prepare base figure
    fig = go.Figure()

    # --- INITIAL FRAME (frame 0) ---
    p1 = points1_list[0]
    p2 = points2_list[0]
    G = G_list[0]

    # Scatter traces for points (these remain and are updated in frames)
    fig.add_trace(go.Scatter3d(
        x=p1[:, x_ind], y=p1[:, 1], z=p1[:, z_ind],
        mode="markers", marker=dict(size=5, color="blue"), name="Points 1"
    ))
    fig.add_trace(go.Scatter3d(
        x=p2[:, x_ind], y=p2[:, 1], z=p2[:, z_ind],
        mode="markers", marker=dict(size=5, color="red"), name="Points 2"
    ))

    # Create line traces for the *maximum possible* number of matches (M)
    # We update their coordinates in each frame.
    M = p1.shape[0]
    line_traces = []
    for _ in range(M):
        line_traces.append(go.Scatter3d(
            x=[None, None],
            y=[None, None],
            z=[None, None],
            mode="lines",
            line=dict(color="gray", width=2),
            showlegend=False
        ))
        fig.add_trace(line_traces[-1])

    # --- BUILD FRAMES ---
    frames = []
    for k in range(N):
        p1 = points1_list[k]
        p2 = points2_list[k]
        G = G_list[k]

        # Extract edges for this frame
        xs, ys, zs, colors = [], [], [], []
        for i in range(M):
            for j in range(M):
                if G[i, j] != 0:
                    p1i = p1[i]
                    p2j = p2[j]
                    xs.append([p1i[x_ind], p2j[x_ind]])
                    ys.append([p1i[1],    p2j[1]])
                    zs.append([p1i[z_ind], p2j[z_ind]])
                    colors.append("red" if color_incorrect and i != j else "gray")

        # Ensure the number of stored line slots = M
        # We fill missing lines with dummy points
        while len(xs) < M:
            xs.append([None, None])
            ys.append([None, None])
            zs.append([None, None])
            colors.append("gray")

        # Build frame data list
        frame_data = []

        # Updated points
        frame_data.append(go.Scatter3d(
            x=p1[:, x_ind], y=p1[:, 1], z=p1[:, z_ind],
            mode="markers", marker=dict(size=5, color="blue")
        ))
        frame_data.append(go.Scatter3d(
            x=p2[:, x_ind], y=p2[:, 1], z=p2[:, z_ind],
            mode="markers", marker=dict(size=5, color="red")
        ))

        # Updated line connections
        for idx in range(M):
            frame_data.append(go.Scatter3d(
                x=xs[idx],
                y=ys[idx],
                z=zs[idx],
                mode="lines",
                line=dict(color=colors[idx], width=2),
                showlegend=False
            ))

        frames.append(go.Frame(data=frame_data, name=f"frame{k}"))

    fig.frames = frames

    # --- LAYOUT / ANIMATION CONTROLS ---
    fig.update_layout(
        title="Point Cloud Matching Animation",
        scene=dict(aspectmode="data"),
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 150, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ]
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method="animate",
                         args=[[f"frame{k}"], {"frame": {"duration": 0, "redraw": True}}],
                         label=str(k))
                    for k in range(N)
                ],
                currentvalue={"prefix": "Frame "}
            )
        ]
    )

    return fig


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
        """
        Load point cloud data from a CSV file. If no filepath is provided, randomly select one from the datasets/csv_files directory.
        """
        if filepath == None:
            csv_dir = Path("datasets/csv_files")
            csv_list = sorted(csv_dir.glob("*.csv"))
            filepath = np.random.choice(csv_list)
        else:
            pass

        self.filepath = Path(filepath)
        self.point_cloud = pd.read_csv(filepath).to_numpy()
        print(f"Loaded point cloud data from {self.filepath}, number of frames: {self.point_cloud.shape[0]}")

    def get_entire_point_cloud(self):
        """
        Return the entire loaded point cloud data.
        """
        return self.point_cloud.reshape(self.point_cloud.shape[0], -1, 3)

    def get_two_random_point_cloud(self):
        """
        Randomly select two point clouds from the first and second halves of the loaded data.
        """
        idx_1 = np.random.choice(self.point_cloud.shape[0]//2)
        idx_2 = np.random.choice(self.point_cloud.shape[0]//2) + self.point_cloud.shape[0]//2
        source = self.point_cloud[idx_1].reshape(-1,3)
        target = self.point_cloud[idx_2].reshape(-1,3)
        return source, target

    def get_pointclouds_fixed_timestep(self, timestep, fixed_beginning_idx = None):
        if fixed_beginning_idx == None:
            idx_1 = np.random.choice(self.point_cloud.shape[0] - timestep)
        else:
            idx_1 = fixed_beginning_idx
        idx_2 = idx_1 + timestep
        source = self.point_cloud[idx_1].reshape(-1,3)
        target = self.point_cloud[idx_2].reshape(-1,3)
        return source, target

    def get_pointclouds_range(self, indices):
        return self.point_cloud[indices]

    def get_t_distant_point_cloud(self, t=500):
        """
        Select two point clouds that are t frames apart.
        """
        if t >= self.point_cloud.shape[0]:
            raise ValueError(f"t is too large. There are {self.point_cloud.shape[0]} frames in this file.")
        idx_1 = np.random.choice(self.point_cloud.shape[0]-t)
        idx_2 = idx_1 + t
        source = self.point_cloud[idx_1].reshape(-1,3)
        target = self.point_cloud[idx_2].reshape(-1,3)
        return source, target

    def get_point_cloud_at_index(self, index):
        """
        Get the point cloud at a specific index.
        """
        if index < 0 or index >= self.point_cloud.shape[0]:
            raise ValueError(f"Index out of bounds. There are {self.point_cloud.shape[0]} frames in this file.")
        pc = self.point_cloud[index].reshape(-1,3)
        return pc

class DistanceProfile:
    def __init__(self, source, target):
        """
        Initialize the DistanceProfile with source and target point clouds.
        """
        self.source = source
        self.target = target

    def compute_L2_matrix(self):
        """
        Compute the L2 distance matrix for the source and target point clouds.
        """
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

    def compute_L1_matrix(self):
        """
        Compute the L1 distance matrix for the source and target point clouds.
        """
        n_source = self.source.shape[0]
        n_target = self.target.shape[0]
        distance_matrix = np.array([np.zeros((n_source, n_source)), np.zeros((n_target, n_target))])
        count = -1
        for cp in [self.source, self.target]:
            count += 1
            n = cp.shape[0]
            for i in range(n):
                for j in range(n):
                    distance_matrix[count][i, j] = np.linalg.norm(cp[i] - cp[j], ord=1)
        return distance_matrix[0], distance_matrix[1]

    def compute_LN_matrix(self,n=1):
        """
        Compute the L-norm distance matrix for the source and target point clouds.
        """
        n_source = self.source.shape[0]
        n_target = self.target.shape[0]
        distance_matrix = np.array([np.zeros((n_source, n_source)), np.zeros((n_target, n_target))])
        count = -1
        for cp in [self.source, self.target]:
            count += 1
            n = cp.shape[0]
            for i in range(n):
                for j in range(n):
                    distance_matrix[count][i, j] = np.linalg.norm(cp[i] - cp[j], ord=n)
        return distance_matrix[0], distance_matrix[1]

    def knn_geodesic_distances(points, k):
        """
        Compute geodesic (shortest-path) distances over the kNN graph for a 3D point cloud.

        Parameters:
            points (np.ndarray): shape (N, 3) point cloud
            k (int): number of nearest neighbors

        Returns:
            np.ndarray: shape (N, N) geodesic distance matrix
        """
        points = np.asarray(points)
        N = points.shape[0]

        # ==== Build full Euclidean distance matrix ====
        diff = points[:, None, :] - points[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))

        # ==== Determine k nearest neighbors for each point ====
        knn_indices = np.argsort(dist_matrix, axis=1)[:, 1:k+1]  # skip self (index 0)

        # ==== Build adjacency list for the kNN graph ====
        graph = [[] for _ in range(N)]
        for i in range(N):
            for j in knn_indices[i]:
                w = dist_matrix[i, j]
                graph[i].append((j, w))
                graph[j].append((i, w))  # make graph symmetric

        # ==== Dijkstra's algorithm for a single source ====
        def dijkstra(start):
            dist = np.full(N, np.inf)
            dist[start] = 0.0
            pq = [(0.0, start)]

            while pq:
                current_dist, u = heapq.heappop(pq)
                if current_dist > dist[u]:
                    continue

                for v, w in graph[u]:
                    new_dist = current_dist + w
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        heapq.heappush(pq, (new_dist, v))

            return dist

        # ==== Compute all-pairs geodesic distances ====
        geodesic_matrix = np.vstack([dijkstra(i) for i in range(N)])

        return geodesic_matrix

    def compute_knn_geodesic_distance_matrix(self, k):
        """
        Compute the kNN geodesic distance matrices for both source and target point clouds.
        """
        source_geodesic = DistanceProfile.knn_geodesic_distances(self.source, k)
        target_geodesic = DistanceProfile.knn_geodesic_distances(self.target, k)
        return source_geodesic, target_geodesic


#Quy-Dzu work

"""
Computes W(i,j) as defined in the equation, given precomputed distance matrices.

We assume p = inf so all mappings are allowed.
"""


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
