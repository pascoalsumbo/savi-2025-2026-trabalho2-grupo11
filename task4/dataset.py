import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def read_labels_txt(path: str) -> List[Tuple[int, int, int, int, int]]:
    """
    Expected label format per line (common in synthetic generators):
      class x1 y1 x2 y2
    Returns list of (cls, x1, y1, x2, y2)
    """
    labels = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            x1, y1, x2, y2 = map(int, parts[1:5])
            labels.append((cls, x1, y1, x2, y2))
    return labels


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def crop_and_resize(im: np.ndarray, box, out_size=28) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = im[y1:y2, x1:x2]
    pil = Image.fromarray(patch)
    pil = pil.resize((out_size, out_size), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


class SynthPatchesDataset(Dataset):
    """
    Generates patches for Task 4:
      - digit patches from GT boxes
      - background patches sampled randomly with low IoU to any GT box
    Labels:
      0..9 digits, 10 = background
    """

    def __init__(
        self,
        dataset_root: str = "task2/data",
        version: str = "synth_vA",
        split: str = "train",
        bg_per_image: int = 3,
        bg_iou_max: float = 0.05,
        seed: int = 0,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.version = version
        self.split = split
        self.bg_per_image = bg_per_image
        self.bg_iou_max = bg_iou_max

        random.seed(seed)

        self.images_dir = os.path.join(dataset_root, version, split, "images")
        self.labels_dir = os.path.join(dataset_root, version, split, "labels")

        self.ids = sorted([fn.replace(".png", "") for fn in os.listdir(self.images_dir) if fn.endswith(".png")])

        # index: list of (img_id, kind, box_or_none, class)
        # kind: "digit" or "bg"
        self.index = []
        for img_id in self.ids:
            lab_path = os.path.join(self.labels_dir, f"{img_id}.txt")
            labels = read_labels_txt(lab_path)
            # digit samples
            for (cls, x1, y1, x2, y2) in labels:
                self.index.append((img_id, "digit", (x1, y1, x2, y2), cls))
            # background samples (fixed count per image)
            for _ in range(bg_per_image):
                self.index.append((img_id, "bg", None, 10))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        img_id, kind, box, cls = self.index[idx]

        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        im = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        H, W = im.shape

        # read labels (needed for BG sampling)
        lab_path = os.path.join(self.labels_dir, f"{img_id}.txt")
        gt = read_labels_txt(lab_path)
        gt_boxes = [(x1, y1, x2, y2) for (_, x1, y1, x2, y2) in gt]

        if kind == "digit":
            patch = crop_and_resize(im, box, out_size=28)

        else:
            # sample random 28x28 patch with low IoU to any GT box
            # try multiple attempts
            for _ in range(200):
                x1 = random.randint(0, W - 28)
                y1 = random.randint(0, H - 28)
                cand = (x1, y1, x1 + 28, y1 + 28)
                ok = True
                for g in gt_boxes:
                    if iou(cand, g) > self.bg_iou_max:
                        ok = False
                        break
                if ok:
                    patch = im[y1:y1 + 28, x1:x1 + 28]
                    break
            else:
                # fallback: if we fail, just take a random patch anyway
                x1 = random.randint(0, W - 28)
                y1 = random.randint(0, H - 28)
                patch = im[y1:y1 + 28, x1:x1 + 28]

        # to tensor, normalize like MNIST
        t = torch.from_numpy(patch).float() / 255.0   # (28,28)
        t = t.unsqueeze(0)                            # (1,28,28)
        t = (t - MNIST_MEAN) / MNIST_STD

        y = torch.tensor(cls, dtype=torch.long)
        return t, y
