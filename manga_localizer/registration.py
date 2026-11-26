"""Registration of original English ink coordinates onto a fixed color raster."""
import cv2
import numpy as np

cv2.setNumThreads(1)


def features(image, max_side=1350):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    scale = min(1., max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    points, descriptors = cv2.SIFT_create(nfeatures=4800).detectAndCompute(small, None)
    xy = np.float32([p.pt for p in points]).reshape(-1,2) / scale
    return xy, descriptors


def correspondence(first, second):
    a,da = first
    b,db = second
    if da is None or db is None or min(len(da),len(db)) < 3:
        return None
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m,n in pairs if m.distance < .75*n.distance]
    if len(good) < 12:
        return None
    src = np.float32([a[m.queryIdx] for m in good])
    dst = np.float32([b[m.trainIdx] for m in good])
    # Hold out a deterministic sample before estimating the transform.
    train = np.arange(len(good)) % 5 != 0
    fit, votes = cv2.estimateAffine2D(src[train],dst[train],method=cv2.RANSAC,
                                      ransacReprojThreshold=2.5,maxIters=3500,confidence=.999)
    if fit is None or np.linalg.det(fit[:,:2]) <= 0:
        return None
    predicted = src @ fit[:,:2].T + fit[:,2]
    errors = np.linalg.norm(predicted-dst, axis=1)
    # Outlier-filter only with a loose (12 native pixel) correspondence limit.
    held = (~train) & (errors <= 12)
    return dict(matrix=fit, inliers=int(votes.sum()), held_xy=dst[held], held_errors=errors[held],
                candidates=len(good), held_total=int((~train).sum()))
