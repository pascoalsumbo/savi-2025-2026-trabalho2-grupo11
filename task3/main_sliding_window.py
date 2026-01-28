import os
import sys
import argparse
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="task2/data")
    p.add_argument("--version", type=str, default="synth_vA",
                   choices=["synth_vA", "synth_vB", "synth_vC", "synth_vD"])
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--image_id", type=int, default=0)

    p.add_argument("--model_path", type=str, default="task1/outputs/best_model.pt")

    p.add_argument("--window", type=int, default=28, help="Sliding window size (square).")
    p.add_argument("--stride", type=int, default=4, help="Stride in pixels (smaller = slower, better localization).")

    # Thresholding (enunciado)
    p.add_argument("--p_thresh", type=float, default=0.95, help="Softmax max prob threshold.")
    p.add_argument("--entropy_thresh", type=float, default=1.5, help="Entropy threshold (lower = more confident).")

    # NMS
    p.add_argument("--use_nms", action="store_true", help="Apply Non-Maximum Suppression.")
    p.add_argument("--nms_iou", type=float, default=0.30, help="NMS IoU threshold (lower = more aggressive).")

    # Visualization control
    p.add_argument("--show_scores", action="store_true", help="Draw class + confidence text on boxes.")
    p.add_argument("--max_draw", type=int, default=50, help="Max number of boxes to draw (avoid clutter).")
    p.add_argument("--save_raw", action="store_true", help="Also save raw detections image (before NMS).")

    p.add_argument("--batch_size", type=int, default=256, help="Batch size for patch inference.")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--out_dir", type=str, default="task3/outputs")
    return p.parse_args()


def mnist_normalize(t: torch.Tensor) -> torch.Tensor:
    """
    MNIST normalization used in Task 1:
    mean=0.1307, std=0.3081
    t: (N,1,28,28) float in [0,1]
    """
    return (t - 0.1307) / 0.3081


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    probs: (N,10)
    returns entropy: (N,)
    """
    p = torch.clamp(probs, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=1)


def load_model(model_path: str, device: str):
    """
    Imports ModelBetterCNN from task1/model.py and loads weights.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task1_path = os.path.join(project_root, "task1")
    if task1_path not in sys.path:
        sys.path.insert(0, task1_path)

    try:
        from model import ModelBetterCNN  # task1/model.py
    except Exception as e:
        raise RuntimeError(
            "Não consegui importar ModelBetterCNN de task1/model.py. "
            "Confirma que tens task1/model.py e a classe ModelBetterCNN."
        ) from e

    model = ModelBetterCNN()
    state = torch.load(os.path.join(project_root, model_path), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_scene_image(dataset_root: str, version: str, split: str, image_id: int) -> np.ndarray:
    """
    returns image as uint8 array (H,W)
    """
    img_path = os.path.join(dataset_root, version, split, "images", f"{image_id:06d}.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
    im = Image.open(img_path).convert("L")
    return np.array(im, dtype=np.uint8)


def sliding_windows(im: np.ndarray, window: int, stride: int) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Creates patches and their boxes.
    Returns:
      patches: (N, window, window) uint8
      boxes: list of (x1,y1,x2,y2) in original image coords
    """
    H, W = im.shape
    patches = []
    boxes = []
    for y in range(0, H - window + 1, stride):
        for x in range(0, W - window + 1, stride):
            patch = im[y:y + window, x:x + window]
            patches.append(patch)
            boxes.append((x, y, x + window, y + window))
    return np.stack(patches, axis=0), boxes


@torch.no_grad()
def detect_patches(model, patches_uint8: np.ndarray, boxes, device: str,
                   p_thresh: float, entropy_thresh: float, batch_size: int):
    """
    Runs model on patches and returns detections as list of:
      (cls, conf, entropy, x1,y1,x2,y2)
    """
    dets = []
    N = patches_uint8.shape[0]

    for i in range(0, N, batch_size):
        batch = patches_uint8[i:i + batch_size]          # (B,h,w)
        bt = torch.from_numpy(batch).float() / 255.0     # (B,h,w)
        bt = bt.unsqueeze(1)                             # (B,1,h,w)

        if bt.shape[-1] != 28 or bt.shape[-2] != 28:
            bt = F.interpolate(bt, size=(28, 28), mode="bilinear", align_corners=False)

        bt = mnist_normalize(bt).to(device)

        logits = model(bt)                  # (B,10)
        probs = F.softmax(logits, dim=1)    # (B,10)
        conf, pred = probs.max(dim=1)       # (B,)
        ent = entropy_from_probs(probs)     # (B,)

        keep = (conf >= p_thresh) & (ent <= entropy_thresh)
        keep_idx = torch.where(keep)[0].cpu().numpy()

        for j in keep_idx:
            x1, y1, x2, y2 = boxes[i + j]
            dets.append((int(pred[j].item()), float(conf[j].item()), float(ent[j].item()), x1, y1, x2, y2))

    return dets


def iou(boxA, boxB) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def nms(dets, iou_thresh: float = 0.3):
    """
    dets: list of (cls, conf, ent, x1,y1,x2,y2)
    returns filtered dets after non-maximum suppression
    """
    if not dets:
        return []

    # Sort by confidence (desc)
    dets_sorted = sorted(dets, key=lambda d: d[1], reverse=True)

    kept = []
    while dets_sorted:
        best = dets_sorted.pop(0)
        kept.append(best)
        best_box = best[3:7]

        remaining = []
        for d in dets_sorted:
            if iou(best_box, d[3:7]) <= iou_thresh:
                remaining.append(d)
        dets_sorted = remaining

    return kept


def draw_detections(im: np.ndarray, dets, out_path: str, title: str,
                    show_scores: bool = False, max_draw: int = 50):
    plt.figure(figsize=(6, 6))
    plt.imshow(im, cmap="gray")
    ax = plt.gca()

    # Optionally limit how many boxes we draw (avoid clutter)
    dets_to_draw = dets[:max_draw]

    for (cls, conf, ent, x1, y1, x2, y2) in dets_to_draw:
        w = x2 - x1
        h = y2 - y1

        rect = plt.Rectangle(
            (x1, y1), w, h,
            fill=False,
            linewidth=2,
            edgecolor="lime"
        )
        ax.add_patch(rect)

        if show_scores:
            ax.text(
                x1, max(0, y1 - 3),
                f"{cls} ({conf:.2f})",
                fontsize=9,
                color="lime",
                bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none")
            )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.model_path, args.device)
    im = load_scene_image(args.dataset_root, args.version, args.split, args.image_id)

    t0 = time.time()
    patches, boxes = sliding_windows(im, window=args.window, stride=args.stride)
    t1 = time.time()

    dets_raw = detect_patches(
        model=model,
        patches_uint8=patches,
        boxes=boxes,
        device=args.device,
        p_thresh=args.p_thresh,
        entropy_thresh=args.entropy_thresh,
        batch_size=args.batch_size
    )
    t2 = time.time()

    dets_final = dets_raw
    if args.use_nms:
        dets_final = nms(dets_raw, iou_thresh=args.nms_iou)

    base = f"{args.version}_{args.split}_{args.image_id:06d}_w{args.window}_s{args.stride}_p{args.p_thresh:.2f}_e{args.entropy_thresh:.2f}"
    if args.use_nms:
        base += f"_nms{args.nms_iou:.2f}"

    out_path = os.path.join(args.out_dir, f"det_{base}.png")

    title = (f"Sliding Window | {args.version}/{args.split} id={args.image_id} "
             f"| window={args.window} stride={args.stride} | dets={len(dets_final)}"
             + (" (NMS)" if args.use_nms else ""))

    draw_detections(im, dets_final, out_path, title, show_scores=args.show_scores, max_draw=args.max_draw)

    if args.save_raw:
        raw_path = os.path.join(args.out_dir, f"det_{base}_RAW.png")
        raw_title = (f"RAW dets={len(dets_raw)} | window={args.window} stride={args.stride}")
        draw_detections(im, dets_raw, raw_path, raw_title, show_scores=args.show_scores, max_draw=args.max_draw)
        print(f"Saved RAW: {raw_path}")

    print("=== Task 3: Sliding Window Detection ===")
    print(f"Device: {args.device}")
    print(f"Image: {args.version}/{args.split} id={args.image_id:06d} shape={im.shape}")
    print(f"Total windows: {len(boxes)} (window={args.window}, stride={args.stride})")
    print(f"Detections (raw): {len(dets_raw)}")
    if args.use_nms:
        print(f"Detections (after NMS): {len(dets_final)} | NMS IoU={args.nms_iou}")
    print(f"Timing: window_gen={t1 - t0:.3f}s | inference={t2 - t1:.3f}s | total={t2 - t0:.3f}s")
    print(f"Saved: {out_path}")

    # Debug: print first few final detections
    for k, d in enumerate(dets_final[:10]):
        cls, conf, ent, x1, y1, x2, y2 = d
        print(f"  det[{k}] cls={cls} conf={conf:.3f} ent={ent:.3f} box=({x1},{y1},{x2},{y2})")


if __name__ == "__main__":
    main()
