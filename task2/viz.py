import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def read_label_file(path: str):
    objs = []
    if not os.path.exists(path):
        return objs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cls, x1, y1, x2, y2 = line.strip().split()
            objs.append((int(cls), int(x1), int(y1), int(x2), int(y2)))
    return objs


def draw_boxes(ax, objs):
    for (cls, x1, y1, x2, y2) in objs:
        w = x2 - x1
        h = y2 - y1
        rect = plt.Rectangle((x1, y1), w, h, fill=False, linewidth=1)
        ax.add_patch(rect)
        ax.text(x1, y1, str(cls), fontsize=8)


def save_mosaic(split_dir: str, out_path: str, n: int = 16):
    img_dir = os.path.join(split_dir, "images")
    lab_dir = os.path.join(split_dir, "labels")

    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])[:n]
    cols = int(math.sqrt(n))
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(cols * 3, rows * 3))
    for i, fn in enumerate(files, 1):
        img_path = os.path.join(img_dir, fn)
        lab_path = os.path.join(lab_dir, fn.replace(".png", ".txt"))

        img = np.array(Image.open(img_path))
        objs = read_label_file(lab_path)

        ax = plt.subplot(rows, cols, i)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        draw_boxes(ax, objs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
