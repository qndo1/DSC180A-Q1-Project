# Formerly script.py
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import ot
import matplotlib.pyplot as plt

# Add project root (one level up) to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # go up 2 levels to repo root
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from dsc180a_q1_project import download_mocap, txt_to_csv, compute_gw_and_plot

def main():
    # Download and convert data
    print("Downloading Mocap Data")
    download_mocap.main_downloader(download_limit=1)
    txt_to_csv.convert_all_txt(convert_limit=1)

    # Load first csv file
    csv_dir = PROJECT_ROOT / "datasets" / "csv_files"
    csv_list = sorted(csv_dir.glob("*.csv"))
    if not csv_list:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    
    first_csv = csv_list[0]
    df = pd.read_csv(first_csv)

    # Select two frames to compare
    frame_one = df.iloc[0].to_numpy().reshape(-1, 3)
    frame_two = df.iloc[499].to_numpy().reshape(-1, 3)
    # Compute GW and plot
    fig, G0 = compute_gw_and_plot(frame_one, frame_two)
    fig.show()

    # M = ot.dist(frame_one, frame_two)

    # N = frame_one.shape[0]

    # a = np.ones(N) / N
    # b = np.ones(N) / N

    # G = ot.solve(M, a, b).plan

    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection="3d")

    # # scatter the two frames
    # ax.scatter(frame_one[:, 2], frame_one[:, 1], frame_one[:, 0], c="C0", s=30, label="frame 1")
    # ax.scatter(frame_two[:, 2], frame_two[:, 1], frame_two[:, 0], c="C1", marker="^", s=40, label="frame 500")
    # ax.set_xlim(-1000, 1000)
    # ax.set_ylim(-1000, 1000)
    # ax.set_zlim(0, 2000)

    # # draw lines for transport pairings (weighted by mass)
    # threshold = 1e-8
    # for i in range(N):
    #     for j in range(N):
    #         w = G[i, j]
    #         if w > threshold:
    #             zs = [frame_one[i, 0], frame_two[j, 0]]
    #             ys = [frame_one[i, 1], frame_two[j, 1]]
    #             xs = [frame_one[i, 2], frame_two[j, 2]]
    #             alpha = min(1.0, w * N * 5)  # scale alpha so larger transported mass is more visible
    #             ax.plot(xs, ys, zs, c="gray", alpha=alpha, linewidth=1)

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()