import os
import random
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def boxes_overlap(a, b, min_gap: int = 0) -> bool:
    # a, b: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # expand boxes by min_gap so we enforce distance
    ax1 -= min_gap; ay1 -= min_gap; ax2 += min_gap; ay2 += min_gap
    bx1 -= min_gap; by1 -= min_gap; bx2 += min_gap; by2 += min_gap

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    return (inter_x2 > inter_x1) and (inter_y2 > inter_y1)
