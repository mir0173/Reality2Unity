"""Original normalization and top-view conversion from the 2024 appendix."""

import numpy as np


def normalize_depth(depth_map, max_val):
    normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    scaled = normalized * max_val
    return scaled


def convert_to_top_view(front_distance_map):
    H, W = front_distance_map.shape
    tmp_map = np.full((H, W), W)
    for i in range(H):
        for j in range(W):
            for k in range(H - int(front_distance_map[i][j]), H):
                tmp_map[k][j] -= 1
    return np.array(tmp_map)
