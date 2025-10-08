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
    H0, W0 = src.shape[:2]
    #Construct a grid of destination pixel coordinates (x, y)
    xs, ys = np.meshgrid(np.arange(Hw, dtype=np.float64),
                         np.arange(Hh, dtype=np.float64))
    P = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=0)  # (3, Hh*Hw)

    #Apply inverse homography
    Hinv = np.linalg.inv(H)
    S = Hinv @ P
    w = S[2, :]
    valid_w = np.abs(w) > 1e-12
    x = np.empty_like(w); y = np.empty_like(w)
    x[valid_w] = S[0, valid_w] / w[valid_w]
    y[valid_w] = S[1, valid_w] / w[valid_w]

    #Check for valid coordinates within the source bounds
    valid_xy = (x >= 0) & (y >= 0) & (x < W0 - 1) & (y < H0 - 1)
    valid = valid_w & valid_xy
    if not np.any(valid):
        return out[..., 0] if C == 1 else out

    xv = x[valid]; yv = y[valid]
    x0 = np.floor(xv).astype(np.int64); y0 = np.floor(yv).astype(np.int64)
    x1 = x0 + 1; y1 = y0 + 1

    ax = xv - x0; ay = yv - y0
    w00 = (1 - ax) * (1 - ay)
    w01 = ax * (1 - ay)
    w10 = (1 - ax) * ay
    w11 = ax * ay

   # 4) Bilinear interpolation
    out_flat = out.reshape(-1, C)
    idx_dst = np.flatnonzero(valid)

    v00 = src_c[y0,  x0,  :]
    v01 = src_c[y0,  x1,  :]
    v10 = src_c[y1,  x0,  :]
    v11 = src_c[y1,  x1,  :]

    out_flat[idx_dst, :] = (
        (w00[:, None] * v00) +
        (w01[:, None] * v01) +
        (w10[:, None] * v10) +
        (w11[:, None] * v11)
    )

    out = out_flat.reshape(Hh, Hw, C)
    return out[..., 0] if C == 1 else out


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
