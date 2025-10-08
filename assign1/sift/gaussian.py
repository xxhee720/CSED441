import numpy as np

# --------------------------------------------
# gaussian.py  (Slides: smoothing, Gaussian kernel, padding, DoG gradients)
# --------------------------------------------

def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """
    Create a 1-D Gaussian kernel (L1-normalized).

    Input
    -----
    sigma  : float (>0)
    radius : int | None  (None => about 3*sigma)

    Output
    ------
    k : (2*radius+1,) float64, sum==1

    Notes
    -----
    REF (used by): gaussian_blur, image_gradients, orientation_assignment
    """
    #############################
    ######### Implement here ####
    # Hint: build symmetric 1-D Gaussian samples and normalize to sum=1.
    if radius is None:
        radius = int(3 * sigma + 0.5)
    x = np.arange(-radius, radius+1)
    # Gaussian
    k = np.exp(-(x**2) / (2 * sigma**2))
    # L1 norm to sum=1
    k /= np.sum(k)
    
    return k
    #############################    
    
def separable_conv2d(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Separable 2-D convolution with 1-D kernel k (horizontal then vertical).

    DEPENDS-ON: none
    USED-BY   : gaussian_blur, orientation_assignment, image_gradients

    - Reflect padding
    - Supports (H,W) and (H,W,C)
    """
    img = img.astype(np.float64)
    r = len(k) // 2
    H, W = img.shape[:2]

    pad2d = ((r, r), (r, r))
    if img.ndim == 2:
        x = np.pad(img, pad2d, mode='reflect')
        tmp = np.zeros_like(x)
        for i in range(-r, r + 1):
            tmp[:, r:r+W] += k[i + r] * x[:, r + i : r + i + W]
        y = np.zeros_like(tmp)
        for i in range(-r, r + 1):
            y[r:r+H, :] += k[i + r] * tmp[r + i : r + i + H, :]
        return y[r:r+H, r:r+W]
    else:
        x = np.pad(img, ((0,0),) + pad2d, mode='reflect')
        tmp = np.zeros_like(x)
        for i in range(-r, r + 1):
            tmp[:, :, r:r+W] += k[i + r] * x[:, :, r + i : r + i + W]
        y = np.zeros_like(tmp)
        for i in range(-r, r + 1):
            y[:, r:r+H, :] += k[i + r] * tmp[:, r + i : r + i + H, :]
        return y[:, r:r+H, r:r+W]


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Wrapper: build a Gaussian 1-D kernel and call separable_conv2d.

    DEPENDS-ON: gaussian_kernel_1d, separable_conv2d
    USED-BY   : pyramid building, blending weights
    """
    k = gaussian_kernel_1d(sigma)
    return separable_conv2d(img, k)


def image_gradients(gray: np.ndarray, sigma: float = 1.0):
    """
    Derivative-of-Gaussian (DoG) gradients (lecture-compliant; no Sobel).

    Input
    -----
    gray  : (H,W) float
    sigma : Gaussian sigma for pre-smoothing

    Output
    ------
    Ix, Iy : (H,W) float64

    DEPENDS-ON: gaussian_kernel_1d, separable_conv2d
    USED-BY   : orientation_assignment, descriptor
    """
    g = gray.astype(np.float64)

    #############################
    ######### Implement here ####
    # Hint:
    # - Combine Gaussian smoothing along one axis with a simple central difference
    #   along the other axis (separable idea).
    # - Build a small 1-D derivative kernel (no Sobel).
    # - Produce Ix and Iy consistent with DoG from the slides.

    if sigma is not None and sigma>0: # If sigma is provided, apply Gaussian smoothing first.
        k=gaussian_kernel_1d(sigma)
        sm=separable_conv2d(g,k)
    else:
        sm=g

    def prod_Ix(img): # Compute horizontal gradient using central differences.
        p=np.pad(img,((0,0),(1,1)),mode="reflect")
        return 0.5*(p[:,2:]-p[:,:-2])
    def prod_Iy(img): # Similar to Ix
        p=np.pad(img,((1, 1),(0, 0)), mode="reflect")
        return 0.5*(p[2:,:]-p[:-2,:])
    
    Ix = prod_Ix(sm) #Produce Ix and Iy consistent
    Iy = prod_Iy(sm)
    
    return Ix, Iy
    #############################
