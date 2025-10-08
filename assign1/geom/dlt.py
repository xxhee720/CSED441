import numpy as np

def normalize_points(pts):
    """
    Isotropic normalization (translation to centroid, scale to mean dist = sqrt(2)).

    Slides basis
    ------------
    - Normalization improves numerical stability in eigen/SVD-based DLT. (lec02)
    """
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c)**2).sum(axis=1)).mean() + 1e-12
    s = np.sqrt(2.0) / d
    T = np.array([[s,0,-s*c[0]],[0,s,-s*c[1]],[0,0,1]], dtype=np.float64)
    pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
    pts_n = (T @ pts_h.T).T[:, :2]
    return T, pts_n


def dlt_homography(pts1, pts2, use_normalization=False):
    """
    Normalized DLT (Direct Linear Transform) to estimate H (3x3), with 4+ correspondences.

    Input
    -----
    pts1, pts2 : (N,2) arrays, pixel coordinates (N>=4, not all collinear)

    Output
    ------
    H : (3,3) homography, scale-fixed so that H[2,2] ~ 1

    ------------
    - Linear DLT setup with two equations per correspondence, solve by SVD
      (eigen viewpoint equivalent). Use normalization and de-normalize at end.
      (lec03 Homography Estimation; lec02 Model Fitting notes on normalization)
    """
    assert pts1.shape[0] >= 4 and pts2.shape[0] >= 4

    T1, n1 = normalize_points(pts1)
    T2, n2 = normalize_points(pts2)
    

    #############################
    ######### Implement here ####
    # Hint:
    # - Build A from normalized pairs (two rows per match).
    # - Compute SVD(A); take the singular vector for the smallest σ as h.
    # - Reshape to Hn (3x3). De-normalize: H = T2^{-1} @ Hn @ T1.
    # - Scale so that H[2,2] ~ 1 (if possible).
    #############################
    if use_normalization: #Normalize points (if requested)
        T1, n1 = normalize_points(pts1)
        T2, n2 = normalize_points(pts2)
    else:
        T1 = np.eye(3, dtype=np.float64)
        T2 = np.eye(3, dtype=np.float64)
        n1 = np.asarray(pts1,dtype=np.float64)
        n2 = np.asarray(pts2,dtype=np.float64)

    N = n1.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float64) #Construct DLT design matrix A
    u, v = n1[:,0], n1[:, 1]
    up, vp = n2[:, 0], n2[:, 1]

    # Each correspondence contributes two equations
    # [ -u  -v  -1   0   0   0   u*up  v*up  up ]
    A[0::2, 0] = -u
    A[0::2, 1] = -v
    A[0::2, 2] = -1.0
    A[0::2, 6] =  u * up
    A[0::2, 7] =  v * up
    A[0::2, 8] =  up

    # [  0   0   0  -u  -v  -1   u*vp  v*vp  vp ]
    A[1::2, 3] = -u
    A[1::2, 4] = -v
    A[1::2, 5] = -1.0
    A[1::2, 6] =  u * vp
    A[1::2, 7] =  v * vp
    A[1::2, 8] =  vp

    #Solve Ah = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :] 
    Hn = h.reshape(3, 3)

    #De-normalize the homography
    H = np.linalg.inv(T2) @ Hn @ T1

    #fix scale
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    else:
        norm = np.linalg.norm(H)
        if norm > 0:
            H = H / norm

    return H



def reprojection_errors(H, pts1, pts2):
    """
    Symmetric transfer error between pts1 <-> pts2 under homography H.
    Return (N,) float errors; large error for invalid projections.

    ------------
    - Evaluate homography fit by projecting both ways and measuring distances.
      (Consistent with model fitting practice in slides)
    """
    N = pts1.shape[0]
    X1 = np.hstack([pts1, np.ones((N,1))])
    X2 = np.hstack([pts2, np.ones((N,1))])

    # forward
    p = (H @ X1.T).T
    w = p[:,2]; valid = np.abs(w) > 1e-12
    p_norm = np.empty((N,2), dtype=np.float64)
    e1 = np.empty(N, dtype=np.float64)
    p_norm[valid] = p[valid,:2]/w[valid,None]
    e1[valid] = np.linalg.norm(p_norm[valid]-pts2[valid], axis=1)
    e1[~valid] = 1e9

    # backward
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(N, 1e9, dtype=np.float64)
    p2 = (Hinv @ X2.T).T
    w2 = p2[:,2]; valid2 = np.abs(w2) > 1e-12
    p2n = np.empty((N,2), dtype=np.float64)
    e2 = np.empty(N, dtype=np.float64)
    p2n[valid2] = p2[valid2,:2]/w2[valid2,None]
    e2[valid2] = np.linalg.norm(p2n[valid2]-pts1[valid2], axis=1)
    e2[~valid2] = 1e9
    return 0.5*(e1+e2)
