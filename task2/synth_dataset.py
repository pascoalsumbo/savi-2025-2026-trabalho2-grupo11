import os
import random
import numpy as np
from typing import List, Tuple, Optional

from PIL import Image
from torchvision import datasets

from utils import ensure_dir, boxes_overlap


def mnist_source(root: str):
    # usa torchvision MNIST (sem link externo)
    train = datasets.MNIST(root=root, train=True, download=True)
    test  = datasets.MNIST(root=root, train=False, download=True)
    return train, test


def sample_digit(mnist_ds):
    idx = random.randrange(0, len(mnist_ds))
    img, label = mnist_ds[idx]
    # img é PIL (28x28), label int
    return img, int(label)


def resize_digit(img: Image.Image, size: int) -> Image.Image:
    # size x size
    return img.resize((size, size), resample=Image.BILINEAR)


def paste_digit(canvas: np.ndarray, digit: np.ndarray, x1: int, y1: int):
    # canvas: HxW uint8, digit: hxw uint8
    h, w = digit.shape
    roi = canvas[y1:y1+h, x1:x1+w]
    # usar máximo para "colar" mantendo dígito brilhante
    np.maximum(roi, digit, out=roi)


def generate_one_scene(
    mnist_ds,
    canvas_size: int = 128,
    n_digits: int = 1,
    scale_range: Optional[Tuple[int, int]] = None,  # (min,max) ou None
    avoid_overlap: bool = True,
    min_gap: int = 2,
    max_tries: int = 200,
) -> Tuple[np.ndarray, List[Tuple[int,int,int,int,int]]]:
    """
    retorna:
      image: HxW uint8
      objects: lista de (class, x1,y1,x2,y2)
    """
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    objects = []

    for _ in range(n_digits):
        img_pil, cls = sample_digit(mnist_ds)

        if scale_range is None:
            size = 28
        else:
            size = random.randint(scale_range[0], scale_range[1])

        digit_pil = resize_digit(img_pil, size) if size != 28 else img_pil
        digit = np.array(digit_pil, dtype=np.uint8)  # hxw

        h, w = digit.shape

        placed = False
        for _try in range(max_tries):
            x1 = random.randint(0, canvas_size - w)
            y1 = random.randint(0, canvas_size - h)
            x2 = x1 + w
            y2 = y1 + h

            new_box = (x1, y1, x2, y2)

            if avoid_overlap:
                ok = True
                for (_, ox1, oy1, ox2, oy2) in objects:
                    if boxes_overlap(new_box, (ox1, oy1, ox2, oy2), min_gap=min_gap):
                        ok = False
                        break
                if not ok:
                    continue

            paste_digit(canvas, digit, x1, y1)
            objects.append((cls, x1, y1, x2, y2))
            placed = True
            break

        if not placed:
            # se não conseguir colocar (muito cheio), simplesmente para aqui
            break

    return canvas, objects


def save_sample(out_dir: str, idx: int, image: np.ndarray, objects: List[Tuple[int,int,int,int,int]]):
    img_dir = os.path.join(out_dir, "images")
    lab_dir = os.path.join(out_dir, "labels")
    ensure_dir(img_dir)
    ensure_dir(lab_dir)

    img_path = os.path.join(img_dir, f"{idx:06d}.png")
    lab_path = os.path.join(lab_dir, f"{idx:06d}.txt")

    Image.fromarray(image).save(img_path)

    with open(lab_path, "w", encoding="utf-8") as f:
        for (cls, x1, y1, x2, y2) in objects:
            f.write(f"{cls} {x1} {y1} {x2} {y2}\n")
