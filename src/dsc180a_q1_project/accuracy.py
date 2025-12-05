import numpy as np

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
    
def dist_accuracy(pc1, pc2, matching):
    row_normed = (np.linalg.inv(np.diag(np.sum(matching, axis = 1))) @ matching)
    diffs = row_normed @ pc2 - pc2
    diffnorms = np.linalg.norm(diffs, axis = 1)
    return np.mean(diffnorms)

def partial_accuracy(G, source, source_points_removed, source_indices_removed, source_indices, target_indices_removed, target_indices, thresh):
    from utils import construct_index_match
    """
    Returns:
    (# correct matchings + # correctly missing matchings) / (# of original points)
    (# correct matchings) / (# possible correct matchings)
    (# correctly missing matchings) / (# points that were removed in either target or source)
    """

    matching = construct_index_match(G, source, source_points_removed, source_indices_removed, target_indices, thresh, original_indices=True)

    correct = 0
    correctly_missing = 0
    total_correct = len(set(source_indices).intersection(set(target_indices)))
    total_missing = source.shape[0] - total_correct
    total = source.shape[0]

    for i in range(total):
        if i in matching and matching[i] == i:
            correct += 1
        elif i in source_indices_removed or i in target_indices_removed:
            if i not in matching:
                correctly_missing += 1
        else:
            pass

    if total_missing == 0:
        return (correct + correctly_missing) / total, correct / total_correct

    return (correct + correctly_missing) / total, correct / total_correct, correctly_missing / total_missing

