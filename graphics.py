import utils

if __name__ == "__main__":
    # Only averaging accuracy across two poses to save computation time, graphics in the paper were done with 100
    utils.acc_full_test(num_poses = 2)
    utils.acc_dist_test(num_poses = 2)
    