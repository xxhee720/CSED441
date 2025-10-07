import numpy as np
from .gaussian import image_gradients

# --------------------------------------------
# descriptor.py  (Slides: 4x4 cells × 8 bins = 128D, aligned to dominant angle)
# --------------------------------------------

def sift_descriptor(gauss_pyr, kpt, theta: float,
                    cell: int = 4, bins: int = 8, scale: int = 8):
    """
    128-D SIFT-like descriptor (4x4 cells * 8 bins), aligned by theta.

    DEPENDS-ON: gaussian.image_gradients
    USED-BY   : matching

    Return
    ------
    desc : (128,) float32 or None
    """
    o, s, y, x = kpt
    G = gauss_pyr[o][s]
    H, W = G.shape

    Ix, Iy = image_gradients(G)
    mag = np.sqrt(Ix**2 + Iy**2)
    ang = (np.arctan2(Iy, Ix) - theta) % (2*np.pi)

    win = cell * scale
    y0, y1 = max(0, y - win//2), min(H, y + win//2)
    x0, x1 = max(0, x - win//2), min(W, x + win//2)
    if y1 - y0 <= 1 or x1 - x0 <= 1:
        return None

    m = mag[y0:y1, x0:x1]
    a = ang[y0:y1, x0:x1]

    # Split into a (cell x cell) grid
    hstep = max(1, (y1 - y0) // cell)
    wstep = max(1, (x1 - x0) // cell)
    desc = np.zeros((cell, cell, bins), dtype=np.float64)

    #############################
    ######### Implement here ####
    # Hints:
    # - For each cell, aggregate a simple orientation histogram from its pixels.
    # - Hard-assign angles to bins; weights from gradient magnitudes.
    #############################
    raise NotImplementedError

    #############################
    ######### Implement here ####
    # Hints:
    # - Flatten to 128-D and L2-normalize.
    # - (Optional) small clipping before renormalization if you want textbook SIFT.
    #############################
    raise NotImplementedError
