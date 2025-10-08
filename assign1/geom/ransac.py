import numpy as np
from geom.dlt import dlt_homography, reprojection_errors

def ransac_homography(pts1, pts2, iters=2000, thresh=0.0001, seed=0):
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
    for _ in range(iters):
        #1) 최소 표본 4점 무작위 선택
        idx = rng.choice(N, size=4, replace=False)

        #2) DLT로 모델 추정
        try:
            Hc = dlt_homography(pts1[idx], pts2[idx], use_normalization=True)
        except Exception:
            continue

        #3) 전체data에 대해 reproj_Error 계산
        errs = reprojection_errors(Hc, pts1, pts2)
        inliers = errs < thresh
        score = int(inliers.sum())
        
        if score > best_score:
            best_score = score
            best_H = Hc
            best_inliers = inliers

    #if non valid
    if best_H is None or best_inliers is None or best_score < 4:
        return None, np.zeros(N, dtype=bool)

    #refit
    inl_idx = np.where(best_inliers)[0]
    try:
        H_refit = dlt_homography(pts1[inl_idx], pts2[inl_idx], use_normalization=True)
    except Exception:
        H_refit = best_H

    #final inlier update
    final_inliers = reprojection_errors(H_refit, pts1, pts2) < thresh

    return H_refit, final_inliers
