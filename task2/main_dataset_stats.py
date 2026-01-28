import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from viz import save_mosaic


def read_objs(label_path: str):
    objs = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            cls, x1, y1, x2, y2 = line.strip().split()
            objs.append((int(cls), int(x1), int(y1), int(x2), int(y2)))
    return objs


def collect_stats(split_dir: str):
    lab_dir = os.path.join(split_dir, "labels")
    label_files = sorted([f for f in os.listdir(lab_dir) if f.endswith(".txt")])

    class_counter = Counter()
    n_digits_list = []
    sizes = []  # (w,h)

    for lf in label_files:
        objs = read_objs(os.path.join(lab_dir, lf))
        n_digits_list.append(len(objs))
        for (cls, x1, y1, x2, y2) in objs:
            class_counter[cls] += 1
            sizes.append((x2 - x1, y2 - y1))

    sizes = np.array(sizes) if len(sizes) > 0 else np.zeros((0, 2))
    return class_counter, np.array(n_digits_list), sizes


def save_hist(arr, title, xlabel, out_path):
    plt.figure()
    plt.hist(arr, bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_bar(counter, title, out_path):
    keys = list(range(10))
    vals = [counter.get(k, 0) for k in keys]

    plt.figure()
    plt.bar(keys, vals)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(keys)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    versions = ["synth_vA", "synth_vB", "synth_vC", "synth_vD"]
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    for v in versions:
        for split in ["train", "test"]:
            split_dir = os.path.join(data_dir, v, split)
            tag = f"{v}_{split}"
            print("Processing:", tag)

            # Mosaic with boxes
            mosaic_path = os.path.join(out_dir, f"mosaic_{tag}.png")
            save_mosaic(split_dir, mosaic_path, n=16)

            # Stats
            class_counter, n_digits, sizes = collect_stats(split_dir)

            bar_path = os.path.join(out_dir, f"class_dist_{tag}.png")
            save_bar(class_counter, f"Class distribution ({tag})", bar_path)

            hist_path = os.path.join(out_dir, f"digits_per_image_{tag}.png")
            save_hist(n_digits, f"Digits per image ({tag})", "digits/image", hist_path)

            if sizes.shape[0] > 0:
                size_path = os.path.join(out_dir, f"digit_size_{tag}.png")
                save_hist(sizes[:, 0], f"Digit width distribution ({tag})", "width (px)", size_path)

    print("DONE. Check task2/outputs/")

if __name__ == "__main__":
    main()
