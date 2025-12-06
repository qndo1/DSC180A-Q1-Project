# Formerly matt_script.py
import sys
import os
from pathlib import Path

import pandas as pd
import ot
import numpy as np
import matplotlib.pyplot as plt

# Add project root (one level up) to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # go up 2 levels to repo root
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
csv_dir = PROJECT_ROOT / "datasets" / "csv_files"


from dsc180a_q1_project import LoadCloudPoint, animate_point_cloud_matches

def main():

    csv_path = PROJECT_ROOT / "datasets" / "csv_files" / "0005_Jogging001.csv"
    lcp = LoadCloudPoint(filepath=csv_path)
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

    fig = animate_point_cloud_matches(
        points1_list,   # list of Nx3 point clouds
        points2_list,   # list of Nx3 point clouds
        G_list,         # list of matching matrices
        switch_yz=True,
        color_incorrect=True
    )
    print(sys.getsizeof(fig.to_plotly_json()))
    fig.show()

if __name__ == "__main__":
    main()
