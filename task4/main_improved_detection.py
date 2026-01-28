import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt

from task4.dataset import SynthPatchesDataset, MNIST_MEAN, MNIST_STD
from task4.model import ModelCNN11
from task4.trainer import TrainConfig, Trainer


def parse_args():
    p = argparse.ArgumentParser()

    # mode
    p.add_argument("--mode", type=str, default="train", choices=["train", "detect"])

    # data
    p.add_argument("--dataset_root", type=str, default="task2/data")
    p.add_argument("--version", type=str, default="synth_vA")

    # train
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--bg_per_image", type=int, default=3)
    p.add_argument("--bg_iou_max", type=float, default=0.05)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)

    # detect (sliding window)
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--image_id", type=int, default=0)
    p.add_argument("--window", type=int, default=28)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--p_thresh", type=float, default=0.90)
    p.add_argument("--use_nms", action="store_true")
    p.add_argument("--nms_iou", type=float, default=0.30)
    p.add_argument("--max_draw", type=int, default=50)

    # model i/o
    p.add_argument("--model_path", type=str, default="task4/outputs/best_model_task4.pt")
    p.add_argument("--out_dir", type=str, default="task4/outputs")

    # device
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def mnist_normalize(t: torch.Tensor) -> torch.Tensor:
    return (t - MNIST_MEAN) / MNIST_STD


def sliding_windows(im: np.ndarray, window: int, stride: int):
    H, W = im.shape
    patches = []
    boxes = []
    for y in range(0, H - window + 1, stride):
        for x in range(0, W - window + 1, stride):
            patches.append(im[y:y + window, x:x + window])
            boxes.append((x, y, x + window, y + window))
    return np.stack(patches, axis=0), boxes


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
    if not dets:
        return []
    dets_sorted = sorted(dets, key=lambda d: d[1], reverse=True)
    kept = []
    while dets_sorted:
        best = dets_sorted.pop(0)
        kept.append(best)
        best_box = best[2:6]
        remaining = []
        for d in dets_sorted:
            if iou(best_box, d[2:6]) <= iou_thresh:
                remaining.append(d)
        dets_sorted = remaining
    return kept


@torch.no_grad()
def detect_scene(model, im_uint8: np.ndarray, device: str,
                 window: int, stride: int, p_thresh: float,
                 batch_size: int = 512):
    """
    Returns detections: (cls, conf, x1,y1,x2,y2) excluding background class 10
    """
    patches, boxes = sliding_windows(im_uint8, window=window, stride=stride)
    N = patches.shape[0]
    dets = []

    for i in range(0, N, batch_size):
        batch = patches[i:i + batch_size]
        bt = torch.from_numpy(batch).float() / 255.0
        bt = bt.unsqueeze(1)

        if bt.shape[-1] != 28 or bt.shape[-2] != 28:
            bt = F.interpolate(bt, size=(28, 28), mode="bilinear", align_corners=False)

        bt = mnist_normalize(bt).to(device)
        logits = model(bt)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        keep = conf >= p_thresh
        idxs = torch.where(keep)[0].cpu().numpy()

        for j in idxs:
            cls = int(pred[j].item())
            if cls == 10:
                continue  # background
            c = float(conf[j].item())
            x1, y1, x2, y2 = boxes[i + j]
            dets.append((cls, c, x1, y1, x2, y2))

    return dets, boxes


def draw_dets(im: np.ndarray, dets, out_path: str, title: str, max_draw: int = 50):
    plt.figure(figsize=(6, 6))
    plt.imshow(im, cmap="gray")
    ax = plt.gca()

    dets_to_draw = dets[:max_draw]
    for (cls, conf, x1, y1, x2, y2) in dets_to_draw:
        w = x2 - x1
        h = y2 - y1
        rect = plt.Rectangle((x1, y1), w, h, fill=False, linewidth=2, edgecolor="lime")
        ax.add_patch(rect)
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


def run_train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print("Device:", args.device)

    ds = SynthPatchesDataset(
        dataset_root=args.dataset_root,
        version=args.version,
        split="train",
        bg_per_image=args.bg_per_image,
        bg_iou_max=args.bg_iou_max
    )

    n_val = int(len(ds) * args.val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(args.device == "cuda"))

    model = ModelCNN11(dropout_p=args.dropout, num_classes=11)

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        out_dir=args.out_dir,
        num_classes=11
    )
    trainer = Trainer(model, cfg)
    hist = trainer.fit(train_loader, val_loader)

    print("\n=== DONE (TRAIN) ===")
    print(f"Best val acc: {hist['best_val_acc']:.4f}")
    print(f"Saved best model: {hist['best_model_path']}")


def run_detect(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print("Device:", args.device)

    # load scene image
    img_path = os.path.join(args.dataset_root, args.version, args.split, "images", f"{args.image_id:06d}.png")
    im = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)

    # load model
    model = ModelCNN11(num_classes=11)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    t0 = time.time()
    dets, boxes = detect_scene(
        model=model,
        im_uint8=im,
        device=args.device,
        window=args.window,
        stride=args.stride,
        p_thresh=args.p_thresh,
    )
    t1 = time.time()

    dets_raw = dets
    dets_final = dets_raw
    #if args.use_nms:
     #   dets_final = nms(dets_raw, iou_thresh=args.nms_iou)
    
    if args.use_nms:
        dets_final = []
        for cls in sorted(set([d[0] for d in dets_raw])):
            cls_dets = [d for d in dets_raw if d[0] == cls]
            dets_final.extend(nms(cls_dets, iou_thresh=args.nms_iou))

    base = f"{args.version}_{args.split}_{args.image_id:06d}_w{args.window}_s{args.stride}_p{args.p_thresh:.2f}"
    if args.use_nms:
        base += f"_nms{args.nms_iou:.2f}"

    out_path = os.path.join(args.out_dir, f"det_task4_{base}.png")
    title = f"Task4 Improved | dets={len(dets_final)} | stride={args.stride}" + (" (NMS)" if args.use_nms else "")
    draw_dets(im, dets_final, out_path, title, max_draw=args.max_draw)

    print("=== DONE (DETECT) ===")
    print(f"Image: {img_path} shape={im.shape}")
    print(f"Total windows: {len(boxes)} (window={args.window}, stride={args.stride})")
    print(f"Detections raw (non-bg): {len(dets_raw)}")
    if args.use_nms:
        print(f"Detections after NMS: {len(dets_final)} | NMS IoU={args.nms_iou}")
    print(f"Detect time: {t1 - t0:.3f}s")
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_detect(args)


if __name__ == "__main__":
    main()
