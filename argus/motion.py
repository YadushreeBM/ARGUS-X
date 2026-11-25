"""Motion estimation and global camera-motion compensation helpers."""
import cv2
import numpy as np

def estimate_camera_transform(previous_gray, current_gray):
    """Estimate previous->current frame affine transform from sparse optical flow."""
    if previous_gray is None or current_gray is None:
        return None
    try:
        points = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=250, qualityLevel=0.01, minDistance=8,
            blockSize=7
        )
        if points is None or len(points) < 8:
            return None
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None,
            winSize=(21, 21), maxLevel=3
        )
        if next_points is None or status is None:
            return None
        good_old = points[status.flatten() == 1]
        good_new = next_points[status.flatten() == 1]
        if len(good_old) < 8:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            good_old, good_new, method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
        if matrix is None or (inliers is not None and int(inliers.sum()) < 6):
            return None
        return matrix.astype(np.float32)
    except cv2.error:
        return None

def transform_bbox(bbox, matrix):
    """Transform all four bbox corners with a 2x3 affine matrix."""
    if matrix is None:
        return np.asarray(bbox, dtype=np.float32)
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    transformed = cv2.transform(corners.reshape(1, -1, 2), matrix).reshape(-1, 2)
    return np.array([
        transformed[:, 0].min(), transformed[:, 1].min(),
        transformed[:, 0].max(), transformed[:, 1].max()
    ], dtype=np.float32)

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return float(intersection / (area_a + area_b - intersection + 1e-8))

def center(bbox):
    return np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2], dtype=np.float32)

def bbox_diagonal(bbox):
    return float(np.hypot(bbox[2]-bbox[0], bbox[3]-bbox[1]) + 1e-6)
