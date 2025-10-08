import numpy as np

# --------------------------------------------
# keypoint_filter.py  (Slides: edge response test via Hessian ratio)
# --------------------------------------------

def hessian_edge_reject(D: np.ndarray, y: int, x: int, edge_th: float = 10.0) -> bool:
    """
    Hessian-based edge rejection (Tr^2 / Det test).

    DEPENDS-ON: none
    USED-BY   : filter_keypoints

    Return
    ------
    keep : bool (True if NOT edge-like)
    """
    #############################
    ######### Implement here ####
    # Hints:
    # - Estimate a 2×2 Hessian at (y,x) from local finite differences.
    # - Compute trace^2 / det, check against ((r+1)^2)/r and det>0.
    
    # 1. Approximate the second derivatives
    Dxx = D[y, x+1] + D[y, x-1] - 2.0 * D[y, x]
    Dyy = D[y+1, x] + D[y-1, x] - 2.0 * D[y, x]
    Dxy = (D[y+1, x+1] - D[y+1, x-1] - D[y-1, x+1] + D[y-1, x-1]) / 4.0

    # 2. Compute the trace and determinant
    traceH = Dxx + Dyy
    detH = Dxx * Dyy - Dxy * Dxy

    # 3. Edge ratio test
    # keypoint is rejected if it lies along an edge,
   # if (Tr(H)^2 / Det(H)) >((r+1)^2 / r),
    # (r is the edge threshold)
    r = edge_th
    ratio = (r+1)**2 / r
    keep = (detH>1e-12) and ((traceH * traceH)/detH < ratio)

    return keep
    
    #############################


def filter_keypoints(dog_pyr, kpts, edge_th: float = 10.0):
    """
    Apply edge rejection to 3D-NMS candidates.

    DEPENDS-ON: hessian_edge_reject
    USED-BY   : SIFT pipeline
    """
    out = []
    for (o, s, y, x) in kpts:
        D = dog_pyr[o][s]
        if 1 <= x < D.shape[1] - 1 and 1 <= y < D.shape[0] - 1:
            if hessian_edge_reject(D, y, x, edge_th=edge_th):
                out.append((o, s, y, x))
    return out
