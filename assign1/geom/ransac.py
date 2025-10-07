import numpy as np
from geom.dlt import dlt_homography, reprojection_errors

def ransac_homography(pts1, pts2, iters=2000, thresh=3.0, seed=0):
    """
    RANSAC to robustly estimate homography.

    Input
    -----
    pts1, pts2 : (N,2) arrays, N>=4
    iters      : number of RANSAC iterations
    thresh     : inlier threshold (pixels) on symmetric transfer error
    seed       : RNG seed

    Output
    ------
    H_best     : (3,3) homography refit on inliers (or None)
    inliers    : (N,) bool mask of inliers w.r.t. H_best

    ------------
    - Randomly sample minimal set (4), fit model, count inliers, keep best,
      and refit on consensus set. (lec02 RANSAC)
    """
    assert pts1.shape == pts2.shape and pts1.shape[0] >= 4
    rng = np.random.default_rng(seed)
    N = pts1.shape[0]
    best_H = None; best_inliers = None; best_score = -1

    #############################
    ######### Implement here ####
    # Hint:
    # - Loop 'iters' times:
    #   * Randomly choose 4 unique indices.
    #   * Fit H via DLT on the 4 points.
    #   * Compute symmetric transfer errors on all pairs.
    #   * Mark inliers by 'thresh'; if better than best, store H & mask.
    # - If no valid model, return (None, zeros(N)).
    # - Else refit H on best inliers (DLT) and return.
    #############################
    raise NotImplementedError
