# Formerly script.py
import sys
import os

# Add project root (one level up) to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.datasets.download_mocap import download_mocap
from src.datasets.txt_to_csv import txt_to_csv
import pandas as pd
from pathlib import Path
import ot
import numpy as np
import matplotlib.pyplot as plt
import src.utils as utils

if __name__ == "__main__":

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

    fig.show()


