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
    #############################
    raise NotImplementedError


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
