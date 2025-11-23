# DSC180A-Q1-Project
Repository for Quarter 1 Replication Project, DSC 180A  

## Overview 
This repository contains the implementation of our Q1 project, focused on point cloud matching, calculating distance mectrics, as well as optimal transport methods (W1, W2, Gromov-Wasserstein). This project explores how different profile matching and accuracy functions can be applied on MOCAP (motion-capture) data.


## Repository Structure

DSC180A-Q1-Project/
│
├── datasets/ # point clouds
│ ├── csv_files/ # MOCAP data in CSV form
│ ├── txt_files/ # MOCAP data in txt form
│
├── src/ # All processing, matching, and evaluation code
│ ├── __init__/ # make src a python package
│ ├── accuracy/ # accuracy(), dist_accuracy()
│ ├── utils/ # LoadCloudPoint, DistanceProfile, plotting functions
│
├── notebooks/ # Used for visualizations
├── scripts/
│ ├── matt_script.py      # Multi-frame matching animation using OT plans
│ ├── script.py           # Downloads mocap data, converts CSV, runs GW 
│
├── requirements.txt # Reproducible conda environment
└── README.md # Project documentation



## Datasets

### SFU MOCAP data
run
```
    script.py
```
This should download one of the mocap files and show the optimal transport matching between the first frame and the 500th frame. We've set the download and conversion limits as being 1, but feel free to remove those limits in script.py to download and convert the entire dataset.

![Image showing the matching between the first and 500th frame of mocap data](images/frame1_frame500_pairing_example.png)
