import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import ot
import os
import random
import scipy as sp
import matplotlib.pyplot as plt


# Matt work

def plot_3d_points_and_connections(points1, points2, G, switch_yz = True, color_incorrect = False):
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

    x_ind = 0
    if switch_yz:
        y_ind = 2
        z_ind = 1
    else:
        y_ind = 1
        z_ind = 2

    # Ensure numpy arrays
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    G = np.asarray(G)

    fig = go.Figure()

    # Plot first set of 3D points
    fig.add_trace(go.Scatter3d(
        x=points1[:, x_ind], y=points1[:, y_ind], z=points1[:, z_ind],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Points 1'
    ))

    # Plot second set of 3D points
    fig.add_trace(go.Scatter3d(
        x=points2[:, x_ind], y=points2[:, y_ind], z=points2[:, z_ind],
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
                    y=[p1[y_ind], p2[y_ind]],
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

def animate_point_cloud_matches(points1_list, points2_list, G_list, switch_yz=True, color_incorrect=False):
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
    x_ind = 0
    if switch_yz:
        y_ind = 2
        z_ind = 1
    else:
        y_ind = 1
        z_ind = 2

    # Prepare base figure
    fig = go.Figure()

    # --- INITIAL FRAME (frame 0) ---
    p1 = points1_list[0]
    p2 = points2_list[0]
    G = G_list[0]

    # Scatter traces for points (these remain and are updated in frames)
    fig.add_trace(go.Scatter3d(
        x=p1[:, x_ind], y=p1[:, y_ind], z=p1[:, z_ind],
        mode="markers", marker=dict(size=5, color="blue"), name="Points 1"
    ))
    fig.add_trace(go.Scatter3d(
        x=p2[:, x_ind], y=p2[:, y_ind], z=p2[:, z_ind],
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
                    ys.append([p1i[y_ind],    p2j[y_ind]])
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
            x=p1[:, x_ind], y=p1[:, y_ind], z=p1[:, z_ind],
            mode="markers", marker=dict(size=5, color="blue")
        ))
        frame_data.append(go.Scatter3d(
            x=p2[:, x_ind], y=p2[:, y_ind], z=p2[:, z_ind],
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

def remove_points_then_match(source, target, alpha = 0, matchtype = "FGW", p = float("inf")):
    """
    Returns:
    G (matching)
    source_points_removed (source good points only)
    target_points_removed (target good points only)
    source_indices_removed (indices of removed source points)
    target_indices_removed (indices of removed target points)
    source_indices (indices of good source points)
    target_indices (indices of good target points)
    """
    
    source_points_removed = []
    target_points_removed = []
    source_indices_removed = []
    target_indices_removed = []
    source_indices = []
    target_indices = []

    for i in range(source.shape[0]):
        if source[i, 1] == 0 or source[i, 2] == 0:
            source_indices_removed.append(i)
            continue
        source_points_removed.append(source[i])
        source_indices.append(i)

    for i in range(target.shape[0]):
        if target[i, 1] == 0 or target[i, 2] == 0:
            target_indices_removed.append(i)
            continue
        target_points_removed.append(target[i])
        target_indices.append(i)

    source_points_removed = np.array(source_points_removed)
    target_points_removed = np.array(target_points_removed)

    M = ot.dist(source_points_removed, target_points_removed)
    a = np.ones(source_points_removed.shape[0]) / source_points_removed.shape[0]
    b = np.ones(target_points_removed.shape[0]) / target_points_removed.shape[0]

    C1 = sp.spatial.distance.cdist(source_points_removed, source_points_removed)
    C2 = sp.spatial.distance.cdist(target_points_removed, target_points_removed)

    if matchtype == "FGW":
        G = ot.fused_gromov_wasserstein(M, C1, C2, alpha=alpha)
    elif matchtype == "pGW":
        G = ot.gromov.partial_gromov_wasserstein(C1, C2, a, b)
    elif matchtype == "DPM":
        C1 = sp.spatial.distance.cdist(source, source)
        C2 = sp.spatial.distance.cdist(target, target)
        G = dpm_finite_p(C1, C2, p = p)
    else:
        G = ot.solve(M, a, b).plan

    output = (
        G,
        source_points_removed,
        target_points_removed,
        source_indices_removed,
        target_indices_removed,
        source_indices,
        target_indices
    )

    return output

def construct_index_match(G, source, source_points_removed, source_indices_removed, target_indices, thresh, original_indices = False):
    """
    original_indices is a boolean of whether you want the indices in the matching to refer to the original indices BEFORE point removal
    """
    matching = {}
    removed_counter = 0

    for i in range(source.shape[0]):
        if i in source_indices_removed:
            removed_counter += 1
            continue
        ind = i - removed_counter
        if G[ind].max() > 1 / source_points_removed.shape[0] * thresh:
            if original_indices:
                matching[i] = target_indices[G[ind].argmax()]
            else:
                matching[ind] = G[ind].argmax()


    return matching

def construct_correctness_dict(G, source, source_points_removed, source_indices_removed, target_indices, thresh):
    correct_dict = {}
    removed_counter = 0

    for i in range(source.shape[0]):
        if i in source_indices_removed:
            removed_counter +=1
            continue
        ind = i - removed_counter
        if G[ind].max() > 1 / source_points_removed.shape[0] * thresh:
            if target_indices[G[ind].argmax()] == i:
                correct_dict[ind] = True
            else:
                correct_dict[ind] = False

    return correct_dict


def plot_matching_points_removed(source, target, thresh = 0.5, alpha = 0, matchtype = "FGW", switch_yz = True):

    
    G, source_points_removed, target_points_removed, source_indices_removed, target_indiced_removed, source_indices, target_indices = remove_points_then_match(source, target,alpha = alpha, matchtype = matchtype)

    matching = construct_index_match(G, source, source_points_removed, source_indices_removed, target_indices, thresh)
    correct_dict = construct_correctness_dict(G, source, source_points_removed, source_indices_removed, target_indices, thresh)

    x_ind = 0
    if switch_yz:
        y_ind = 2
        z_ind = 1
    else:
        y_ind = 1
        z_ind = 2

    fig = go.Figure()

    xs_red, ys_red, zs_red = [], [], []
    xs_gray, ys_gray, zs_gray = [], [], []

    for ind in matching:
        p1 = source_points_removed[ind]
        p2 = target_points_removed[matching[ind]]

        if correct_dict[ind]:
            xs, ys, zs = xs_gray, ys_gray, zs_gray
        else:
            xs, ys, zs = xs_red, ys_red, zs_red

        xs += [p1[x_ind], p2[x_ind], None]
        ys += [p1[y_ind], p2[y_ind], None]
        zs += [p1[z_ind], p2[z_ind], None]

    fig.add_trace(go.Scatter3d(
        x=xs_gray, y=ys_gray, z=zs_gray, 
        mode="lines",
        line=dict(color="gray", width=2)
    ))

    fig.add_trace(go.Scatter3d(
        x=xs_red, y=ys_red, z=zs_red, 
        mode="lines",
        line=dict(color="red", width=2)
    ))

    
    fig.add_trace(go.Scatter3d(
        x=source_points_removed[:, x_ind], y=source_points_removed[:, y_ind], z=source_points_removed[:, z_ind],
        mode="markers", marker=dict(size=5, color="blue"), name="Points 1"
    ))
    fig.add_trace(go.Scatter3d(
        x=target_points_removed[:, x_ind], y=target_points_removed[:, y_ind], z=target_points_removed[:, z_ind],
        mode="markers", marker=dict(size=5, color="red"), name="Points 2"
    ))

    return fig

def get_random_clouds(range_length = 100):
    lcp = LoadCloudPoint()
    return lcp.get_pointsclouds_random_range(range_length)

def test_acc_random_pose(accfunc, matchtype, range_length = 100, remove_points = False, alpha = 0.5, threshold = 0.5, p = float("inf")):
    import accuracy
    clouds = get_random_clouds(range_length)

    accs = []

    for i in range(range_length):
        if remove_points:
            if accfunc != accuracy.partial_accuracy:
                print("Using partial accuracy instead")
            if matchtype == "DPM":
                pi, I = compute_W_matrix_distance_matrix_input_finite_p(
                    sp.spatial.distance.cdist(clouds[0], clouds[0]),
                    sp.spatial.distance.cdist(clouds[i], clouds[i]),
                    p = p
                )
                correct = sum([k == pi[k] for k in I])
                correctly_missing = 0
                for j in range(clouds[0].shape[0]):
                    if (np.prod(clouds[0][j]) == 0 or np.prod(clouds[i][j]) == 0) and j not in I:
                        correctly_missing += 1
                accs.append((correct + correctly_missing) / clouds[0].shape[0])
                continue
            G, source_points_removed, target_points_removed, source_indices_removed, target_indices_removed, source_indices, target_indices = remove_points_then_match(clouds[0], clouds[i], matchtype = matchtype, alpha = alpha)
            acc = accuracy.partial_accuracy(G, clouds[0], source_points_removed, source_indices_removed, source_indices, target_indices_removed, target_indices, thresh = threshold)[0]
            accs.append(acc)
        else:
            C1 = sp.spatial.distance.cdist(clouds[0], clouds[0])
            C2 = sp.spatial.distance.cdist(clouds[i], clouds[i])           
            if matchtype == "FGW":
                M = ot.dist(clouds[0], clouds[i])
                G = ot.fused_gromov_wasserstein(M, C1, C2, alpha = alpha)
            elif matchtype == "pGW":
                a = np.ones(clouds[0].shape[0]) / clouds[0].shape[0]
                b = np.ones(clouds[i].shape[0]) / clouds[i].shape[0]
                G = ot.gromov.partial_gromov_wasserstein(C1, C2, a, b)
            else:
                if matchtype != "DPM":
                    print("Unknown matchtype provided, will proceed using distance profile matching. Current supported options are 'FGW', 'pGW', 'DPM'.")
                G = dpm_finite_p(C1, C2, p = p)   
            if accfunc == accuracy.accuracy:
                accs.append(accfunc(G))
            elif accfunc == accuracy.dist_accuracy:
                accs.append(accfunc(clouds[0], clouds[i], G))

    return accs

def test_acc_many_poses(accfunc, matchtype, num_poses = 100, range_length = 100, remove_points = False, alpha = 0.5, threshold = 0.5, p = float("inf")):
    all_accs = []
    for _ in range(num_poses):
        accs = test_acc_random_pose(accfunc, matchtype=matchtype, range_length=range_length, remove_points=remove_points, alpha = alpha, threshold=threshold, p = p)
        all_accs.append(accs)
    return np.array(all_accs).mean(axis = 0)

def compute_W_matrix_distance_matrix_input_finite_p(X_dists, Y_dists, p=float("inf")):
    n, _ = X_dists.shape
    m, _ = Y_dists.shape

    # Pre-sort each row (for 1D Wasserstein)
    X_sorted = np.sort(X_dists, axis=1)  # (n,d)
    Y_sorted = np.sort(Y_dists, axis=1)  # (m,d)

    # Compute pairwise Wasserstein distances using broadcasting
    abs_diff = np.abs(X_sorted[:, None, :] - Y_sorted[None, :, :])  # (n,m,d)
    W = np.mean(abs_diff, axis=2)  # (n,m)

    # Find argmin and threshold indices
    pi_arr = np.argmin(W, axis=1)
    I_arr = np.where(np.min(W, axis=1) < p)[0]

    # Optional: convert pi_arr to dict if you really want
    pi = {i: pi_arr[i] for i in range(n)}

    return pi, list(I_arr)


def dpm_finite_p(c1, c2, p = float("inf")):
    out = np.zeros((c1.shape[0], c2.shape[0]))
    pi, I = compute_W_matrix_distance_matrix_input_finite_p(c1, c2, p)
    for i in I:
        out[i, pi[i]] = 1 / c1.shape[0]
    return out

def acc_full_test(num_poses = 2):
    import accuracy
    matchtypes_labels = [
        ("FGW", 0),
        ("FGW", 0.5),
        ("FGW", 1),
        ("pGW", 0),
        ("DPM", 0)
    ]
    accfunc = accuracy.accuracy
    for matchtype, alpha in matchtypes_labels:
        if matchtype == "FGW":
            label = matchtype + f" alpha = {alpha}"
        else:
            label = matchtype
        plt.plot(
            test_acc_many_poses(accfunc, matchtype, num_poses=num_poses, remove_points=False, alpha = alpha),
            label = label
        )
        print(f"Plotted {label}")
    plt.xlabel("Timesteps From Start")
    plt.ylabel(r"Mean $\text{Acc}_{\text{full}}$")
    plt.title(r"Mean $\text{Acc}_{\text{full}}$ by Time Delta")
    plt.legend()
    plt.show()

def acc_dist_test(num_poses = 2):
    import accuracy
    matchtypes_labels = [
        ("FGW", 0),
        ("FGW", 0.5),
        ("FGW", 1),
        ("pGW", 0),
        ("DPM", 0)
    ]
    accfunc = accuracy.dist_accuracy
    for matchtype, alpha in matchtypes_labels:
        if matchtype == "FGW":
            label = matchtype + f" alpha = {alpha}"
        else:
            label = matchtype
        plt.plot(
            test_acc_many_poses(accfunc, matchtype, num_poses=num_poses, remove_points=False, alpha = alpha),
            label = label
        )
        print(f"Plotted {label}")
    plt.xlabel("Timesteps From Start")
    plt.ylabel(r"Mean $\text{Acc}_{\text{full}}$")
    plt.title(r"Mean $\text{Acc}_{\text{full}}$ by Time Delta")
    plt.legend()
    plt.show()

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
    def __init__(self, filepath=None, verbose = False):
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
        self.verbose = verbose
        if self.verbose:
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
        output = []
        for index in indices:
            output.append(self.point_cloud[index].reshape(-1, 3))
        return output
    
    def get_pointsclouds_random_range(self, range_length):
        start_idx = np.random.choice(self.point_cloud.shape[0] - range_length - 1)
        output = []
        for idx in np.arange(start_idx, start_idx + range_length):
            output.append(self.point_cloud[idx].reshape(-1, 3))
        return output
    
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

