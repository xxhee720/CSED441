import numpy as np
from .gaussian import image_gradients, gaussian_kernel_1d, separable_conv2d

# --------------------------------------------
# orientation.py  (Slides: gradient-based orientation histogram, multi-peak)
# --------------------------------------------

def orientation_assignment(gauss_pyr, kpt, num_bins: int = 36, peak_rel: float = 0.8):
    """
    Assign dominant orientations to a keypoint.

    DEPENDS-ON: gaussian.image_gradients
                (Optional weighting) gaussian.gaussian_kernel_1d, gaussian.separable_conv2d
    USED-BY   : descriptor.sift_descriptor

    Output
    ------
    orientations : list[float] (radians)
    """
    o, s, y, x = kpt
    G = gauss_pyr[o][s]
    Ix, Iy = image_gradients(G)
    mag = np.sqrt(Ix**2 + Iy**2)
    ang = np.arctan2(Iy, Ix)

    # Local window near (y,x)
    win = 8
    y0, y1 = max(0, y - win), min(G.shape[0], y + win + 1)
    x0, x1 = max(0, x - win), min(G.shape[1], x + win + 1)
    m = mag[y0:y1, x0:x1].copy()
    a = ang[y0:y1, x0:x1].copy()

    k = gaussian_kernel_1d(1.5)
    m = separable_conv2d(m, k)

    #############################
    ######### Implement here ####
    # Hints:
    # - Build a histogram over [0, 2π) with 'num_bins' bins using window magnitudes.
    # - Select all bin centers whose height ≥ peak_rel * (global max).
    #############################
    
    #1. normalize angles to [0, 2π)
    a=(a + 2*np.pi) % (2*np.pi)

    #2. build an orientation histogram with 'num_bins'
    nb = num_bins
    hist = np.zeros(nb, dtype=np.float64)

    # Compute fractional bin positions
    bin_f = (a * nb) / (2*np.pi)
    b0 = np.floor(bin_f).astype(int) % nb
    w1 = bin_f - np.floor(bin_f)
    b1 = (b0 + 1) % nb

    # Distribute magnitudes between neighboring bins
    np.add.at(hist, b0, m * (1.0 - w1))
    np.add.at(hist, b1, m * w1)

    # 3)smooth the histogram
    hist = np.convolve(hist, [1, 1, 1], mode='same')

    # 4)Find all peaks above the relative threshold
    peak = hist.max()
    if peak <= 0:
        return [0.0]

    th = peak_rel * peak
    orientations = []
    for i in range(nb):
        li = hist[(i - 1) % nb]
        ci = hist[i]
        ri = hist[(i + 1) % nb]
        
        
        if ci >= th and ci > li and ci > ri:
            #5)refinement using parabolic interpolation
            denom = (li - 2*ci + ri)
            if abs(denom) < 1e-12: offset = 0.0
            else: offset = 0.5 * (li - ri)/denom
            
            bin_pos = (i + offset) % nb
            angle = (bin_pos/nb) * 2*np.pi
            #Convert angle to range [-π, π)
            angle = (angle + np.pi)%(2*np.pi)-np.pi
            orientations.append(float(angle))

    # if no valid orientation found
    if not orientations:
        imax = int(np.argmax(hist))
        angle = (imax/nb)*2*np.pi
        angle = (angle+np.pi)%(2*np.pi)-np.pi
        orientations.append(float(angle))

    return orientations
