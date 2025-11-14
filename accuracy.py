import numpy

def accuracy(matching):
    """
    Compute accuracy of matching given a square matrix. Assumes ordering of point clouds is the same, i.e. assumes the matching should be a scaled version of the identity matrix. Also assumes the matching is one to one, i.e. a scaled permutation matrix.
    """
    if matching.shape[0] != matching.shape[1]:
        raise ValueError("Matching matrix is not square.")
    
    n_points = matching.shape[0]

    correct = 0

    for i in range(n_points):
        if matching[i, i] == 1 / n_points:
            correct += 1

    return correct / n_points
    
