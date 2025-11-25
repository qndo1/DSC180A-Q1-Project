from datasets import download_mocap
from datasets import txt_to_csv
import pandas as pd
from pathlib import Path
import ot
import numpy as np
import matplotlib.pyplot as plt
import utils
import sys

def animation_test():
    lcp = utils.LoadCloudPoint(filepath="datasets/csv_files/0005_Jogging001.csv")
    print("CSV loaded")
    N_frames = 100
    point_clouds = lcp.get_pointclouds_range(range(N_frames))
    points1_list = [point_clouds[0].reshape(-1, 3) for _ in range(len(point_clouds))]
    points2_list = [x.reshape(-1, 3) for x in point_clouds]
    G_list = []
    print("Starting point-matching")
    for i in range(len(points1_list)):
        this_sc = points1_list[i]
        this_tc = points2_list[i]
        M = ot.dist(this_sc, this_tc)
        G = ot.solve(
            M
            , np.ones(this_sc.shape[0]) / this_sc.shape[0]
            , np.ones(this_tc.shape[0]) / this_tc.shape[0]
        ).plan
        G_list.append(G)
        if i+1 % 10 == 0:
            print(f"Finished {i+1}/{N_frames}")

    fig = utils.animate_point_cloud_matches(
        points1_list,   # list of Nx3 point clouds
        points2_list,   # list of Nx3 point clouds
        G_list,         # list of matching matrices
        switch_xz=True,
        color_incorrect=True
    )
    print(sys.getsizeof(fig.to_plotly_json()))
    fig.show()

def point_removal_test(thresh):
    lcp = utils.LoadCloudPoint(filepath="datasets/csv_files/0005_Jogging001.csv")

    source, target = lcp.get_pointclouds_fixed_timestep(10, fixed_beginning_idx=0)

    fig = utils.plot_matching_points_removed(source, target, thresh=thresh)
    fig.show()

if __name__ == "__main__":
    #animation_test()
    #point_removal_test(.9)
    #point_removal_test(.5)

    pass


