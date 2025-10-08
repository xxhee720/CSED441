import numpy as np
from .gaussian import gaussian_blur

# --------------------------------------------
# dog.py  (Slides: scale space, k=2^(1/S), DoG, 3×3×3 NMS, contrast)
# --------------------------------------------

def build_gaussian_pyramid(gray: np.ndarray, sigma0: float = 1.6,
                           octaves: int = 4, scales: int = 3):
    """
    Build SIFT-style Gaussian pyramid with (scales+3) levels per octave.

    DEPENDS-ON: gaussian_blur
    USED-BY   : build_dog_pyramid, SIFT keypoint detection

    Output
    ------
    pyr : list[list[np.ndarray]]  (octaves × (scales+3))
    """
    #############################
    ######### Implement here ####
    # Hint:
    # - Use k = 2^(1/scales).
    # - First level in each octave has absolute sigma0.
    # - Next levels use incremental blur (level-to-level).
    # - Next octave base = previous octave's mid-level downsampled by 2.
    #############################
    img = gray.astype(np.float64, copy=False) #Convert input image to double precision
    k = 2.0 ** (1.0 / float(scales)) # Scale factor
    pyr = []

    base = img #init base image

    for o in range(octaves):
        sigmas_abs = [sigma0 * (k ** s) for s in range(scales + 3)] # Compute absolute sigma values
        levels = []

        # level 0: apply gaussian blur w/ base simga
        l0 = gaussian_blur(base, sigmas_abs[0])
        levels.append(l0)

        # other levels: apply incremental blur
        prev_abs = sigmas_abs[0]
        for s in range(1, scales + 3):
            sigma_inc = np.sqrt(max(sigmas_abs[s]**2 - prev_abs**2, 1e-12))
            li = gaussian_blur(levels[-1], sigma_inc)
            levels.append(li)
            prev_abs = sigmas_abs[s]

        pyr.append(levels) # store all images

        base = levels[scales][::2, ::2] # Next octave base = previous octave's mid-level downsampled by 2.

    return pyr


def build_dog_pyramid(gauss_pyr):
    """
    DoG pyramid: D[o][s] = G[o][s+1] - G[o][s]

    DEPENDS-ON: none
    USED-BY   : nms_3d, keypoint_filter
    """
    dog = []
    for levels in gauss_pyr:
        d = [levels[i+1] - levels[i] for i in range(len(levels) - 1)]
        dog.append(d)
    return dog


def nms_3d(dog_pyr, contrast_th: float = 0.03):
    """
    3×3×3 NMS in (x,y,scale) to propose keypoints; apply contrast threshold.

    DEPENDS-ON: none
    USED-BY   : keypoint_filter.filter_keypoints

    Output
    ------
    kpts : list[(o, s, y, x)]
    """
    kpts = []
    for o, dlevels in enumerate(dog_pyr):
        for s in range(1, len(dlevels) - 1):
            Dm, D0, Dp = dlevels[s-1], dlevels[s], dlevels[s+1]
            H, W = D0.shape
            for y in range(1, H-1):
                for x in range(1, W-1):
                    v = D0[y, x]
                    if abs(v) < contrast_th:
                        continue
                    patch = np.array([
                        Dm[y-1:y+2, x-1:x+2],
                        D0[y-1:y+2, x-1:x+2],
                        Dp[y-1:y+2, x-1:x+2]])
                    if (v > 0 and v >= patch.max()) or (v < 0 and v <= patch.min()):
                        kpts.append((o, s, y, x))
    return kpts
