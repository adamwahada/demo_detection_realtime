"""
Matching utilities for object tracking.
"""

import numpy as np
import scipy
from scipy.spatial.distance import cdist


def linear_assignment(cost_matrix, thresh):
    """
    Perform linear assignment using the Hungarian algorithm.
    
    Args:
        cost_matrix: Cost matrix for assignment
        thresh: Threshold for valid assignments
    
    Returns:
        matches: List of matched pairs (track_idx, detection_idx)
        unmatched_a: List of unmatched track indices
        unmatched_b: List of unmatched detection indices
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap_jv(cost_matrix, extend_cost=True, cost_limit=thresh)
    
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    
    return matches, unmatched_a, unmatched_b


def lap_jv(cost, extend_cost=False, cost_limit=np.inf, return_cost=True):
    """
    Linear assignment problem solver using Jonker-Volgenant algorithm.
    
    Args:
        cost: Cost matrix
        extend_cost: Whether to extend the cost matrix
        cost_limit: Cost limit for valid assignments
        return_cost: Whether to return the total cost
    
    Returns:
        cost: Total assignment cost
        x: Assignment for rows
        y: Assignment for columns
    """
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return 0, np.array([], dtype=int), np.array([], dtype=int)
    
    # Use scipy's linear_sum_assignment (Hungarian algorithm)
    cost_limit_matrix = cost.copy()
    cost_limit_matrix[cost_limit_matrix > cost_limit] = cost_limit + 1
    
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_limit_matrix)
    
    # Filter out assignments that exceed cost limit
    valid = cost[row_ind, col_ind] <= cost_limit
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]
    
    # Create assignment arrays
    x = np.full(cost.shape[0], -1, dtype=int)
    y = np.full(cost.shape[1], -1, dtype=int)
    
    x[row_ind] = col_ind
    y[col_ind] = row_ind
    
    if return_cost:
        total_cost = cost[row_ind, col_ind].sum()
        return total_cost, x, y
    else:
        return x, y


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU between tracks.
    
    Args:
        atracks: List of tracks (first set)
        btracks: List of tracks (second set)
    
    Returns:
        cost_matrix: Cost matrix based on 1 - IoU
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    
    return cost_matrix


def ious(atlbrs, btlbrs):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        atlbrs: Nx4 array of boxes in (top, left, bottom, right) format
        btlbrs: Mx4 array of boxes in (top, left, bottom, right) format
    
    Returns:
        ious: NxM array of IoU values
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float32),
        np.ascontiguousarray(btlbrs, dtype=np.float32)
    )
    
    return ious


def bbox_ious(boxes1, boxes2):
    """
    Compute IoU between two sets of bounding boxes.
    
    Args:
        boxes1: Nx4 array of boxes in (x1, y1, x2, y2) format
        boxes2: Mx4 array of boxes in (x1, y1, x2, y2) format
    
    Returns:
        ious: NxM array of IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = np.clip(rb - lt, 0, None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    ious = inter / np.clip(union, 1e-6, None)
    
    return ious


def embedding_distance(tracks, detections, metric='cosine'):
    """
    Compute distance between track and detection embeddings.
    
    Args:
        tracks: List of tracks with embeddings
        detections: List of detections with embeddings
        metric: Distance metric ('cosine' or 'euclidean')
    
    Returns:
        cost_matrix: Distance matrix
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    
    return cost_matrix


def fuse_score(cost_matrix, detections):
    """
    Fuse cost matrix with detection scores.
    
    Args:
        cost_matrix: Cost matrix
        detections: List of detections with scores
    
    Returns:
        fused_cost: Cost matrix fused with detection scores
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fused_cost = 1 - fuse_sim
    
    return fused_cost
