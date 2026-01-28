import argparse
import os
import torch

from dataset import DataConfig, get_dataloaders
from model import ModelBetterCNN
from trainer import TrainConfig, Trainer
from metrics import predict_all, compute_metrics
from utils import ensure_dir, save_confusion_matrix, save_text, save_training_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--outputs", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    out_dir = os.path.join(os.path.dirname(__file__), args.outputs)
    ensure_dir(out_dir)

    # Data
    dcfg = DataConfig(
        data_root=os.path.join(os.path.dirname(__file__), "data"),
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=2
    )
    train_loader, val_loader, test_loader = get_dataloaders(dcfg)

    # Model
    model = ModelBetterCNN(dropout_p=args.dropout)

    # Train
    tcfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device
    )
    trainer = Trainer(model, tcfg)
    hist = trainer.fit(train_loader, val_loader)  # agora devolve history completo

    # Save training curves (loss + acc)
    curves_base = os.path.join(out_dir, "training_curves.png")
    save_training_curves(hist, curves_base)
    curves_loss_path = curves_base.replace(".png", "_loss.png")
    curves_acc_path = curves_base.replace(".png", "_acc.png")

    # Test metrics
    y_true, y_pred = predict_all(trainer.model, test_loader, device=device)
    m = compute_metrics(y_true, y_pred)

    # Save outputs
    model_path = os.path.join(out_dir, "best_model.pt")
    torch.save(trainer.model.state_dict(), model_path)

    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    save_confusion_matrix(m["confusion_matrix"], cm_path)

    report_path = os.path.join(out_dir, "report.txt")
    header = []
    header.append("=== SAVI-MNIST Task 1 ===\n")
    header.append(f"Best val acc: {hist['best_val_acc']:.4f}\n")
    header.append(f"Macro Precision: {m['macro']['precision']:.4f}\n")
    header.append(f"Macro Recall:    {m['macro']['recall']:.4f}\n")
    header.append(f"Macro F1:        {m['macro']['f1']:.4f}\n\n")
    header.append("Classification Report (test):\n")
    header.append(m["report_text"])
    header.append("\n\nArtifacts:\n")
    header.append(f"- Model: {model_path}\n")
    header.append(f"- Confusion Matrix: {cm_path}\n")
    header.append(f"- Training Curves (Loss): {curves_loss_path}\n")
    header.append(f"- Training Curves (Accuracy): {curves_acc_path}\n")

    save_text("".join(header), report_path)

    print("\n=== DONE ===")
    print(f"Saved model to: {model_path}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved training curves (loss) to: {curves_loss_path}")
    print(f"Saved training curves (acc) to: {curves_acc_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
