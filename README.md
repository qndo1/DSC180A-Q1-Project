# DSC180A-Q1-Project
Repository for Quarter 1 Replication Project, DSC 180A  

## Overview 
This repository contains the implementation of our Q1 project, focused on point cloud matching, calculating distance mectrics, as well as optimal transport methods (W1, W2, Gromov-Wasserstein). This project explores how different profile matching and accuracy functions can be applied on MOCAP (motion-capture) data.


## Repository Structure
```
DSC180A-Q1-Project/
├── datasets/               # point clouds
│   ├── csv_files/          # MOCAP data in CSV form
│   └── txt_files/          # MOCAP data in TXT form
├── src/                    # All processing, matching, and evaluation code
│   ├── __init__.py         # make src a Python package
│   ├── accuracy/           # accuracy(), dist_accuracy()
│   ├── utils/              # LoadCloudPoint, DistanceProfile, plotting functions
│   └── plot_utils.py       # optional: 3D plotting & animation
├── notebooks/              # Used for visualizations
├── scripts/
│   ├── matt_script.py      # Multi-frame matching animation using OT plans
│   └── script.py           # Downloads mocap data, converts CSV, runs GW
├── requirements.txt        # Reproducible conda environment
└── README.md               # Project documentation
```

### Notebooks

All notebooks are in the `notebooks/` folder and are used for exploration, visualization, and testing of point cloud matching algorithms.

| Notebook | Purpose / Experiments |
|----------|---------------------|
| `distance_profile.ipynb` | Load and visualize point clouds, remove outliers, cluster joints, compute L2 distance profiles, 3D visualizations, and matching accuracy. |
| `distance_profile_qndo.ipynb` | Compare distance-profile OT (L1 & L2), vanilla OT, and GW OT matchings; visualize 3D matchings and compute accuracy. |
| `fused_gromov.ipynb` | Apply Fused Gromov-Wasserstein OT, sweep alpha parameters, evaluate multi-timestep accuracy, and compare with standard Wasserstein. |


## Datasets


### SFU MOCAP data
run
```
    script.py
```
This should download one of the mocap files and show the optimal transport matching between the first frame and the 500th frame. We've set the download and conversion limits as being 1, but feel free to remove those limits in script.py to download and convert the entire dataset.

![Image showing the matching between the first and 500th frame of mocap data](images/frame1_frame500_pairing_example.png)

## Sources/References
1. **Python Optimal Transport (POT)** – Library for optimal transport computations, including Wasserstein and Gromov-Wasserstein distances: [https://pythonot.github.io/](https://pythonot.github.io/)
2. **Motion Capture (MoCap) Dataset** – CMU MoCap database used for point cloud data: [https://mocap.cs.sfu.ca/](https://mocap.cs.sfu.ca/)
3. **Robust Point Matching with Distance Profiles** – Paper introducing distance-profile-based robust point cloud matching: [https://arxiv.org/pdf/2312.12641](https://arxiv.org/pdf/2312.12641)

