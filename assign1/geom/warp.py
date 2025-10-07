import numpy as np

def warp_inverse_map(src, H, out_shape, fill=0.0):
    """
    Warp source image to target canvas using inverse mapping with bilinear sampling.

    Input
    -----
    src       : (H,W) or (H,W,3) float array in [0,1]
    H         : (3,3) homography mapping src -> dst
    out_shape : (H_out, W_out)
    fill      : background fill value

    Output
    ------
    out : warped image shaped out_shape (and same channels as src)

    ------------
    - Homogeneous coordinates + projective transform (lec02/lec03).
    - Inverse mapping is standard for resampling.
    """
    Hh, Hw = out_shape
    if src.ndim == 2: C = 1; src_c = src[...,None]
    else: C = src.shape[2]; src_c = src
    out = np.full((Hh, Hw, C), fill, dtype=np.float64)

    #############################
    ######### Implement here ####
    # Hint:
    # - Create a grid of destination coords; convert to homogeneous.
    # - Map by H^{-1} to source coords; divide by w.
    # - Bilinear sample (check 4 neighbors; skip out-of-bounds).
    # - Write to 'out' per channel. Return gray if C==1.
    #############################
    raise NotImplementedError


def compose_bounds(img, H):
    """
    Compute bounding box after warping 'img' by H (for panorama canvas sizing).

    Output
    ------
    x_min, y_min, x_max, y_max : ints (inclusive/exclusive as you prefer)
    """
    H0, W0 = img.shape[:2]
    corners = np.array([[0,0,1],[W0,0,1],[0,H0,1],[W0,H0,1]], dtype=np.float64).T
    warped = (H @ corners); warped = (warped[:2,:]/warped[2:,:]).T
    xs = np.concatenate([warped[:,0],[0,W0]]); ys = np.concatenate([warped[:,1],[0,H0]])
    x_min, x_max = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    return x_min, y_min, x_max, y_max
