import os
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    out_dir: str = "task4/outputs"
    num_classes: int = 11


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg
        self.device = cfg.device

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        os.makedirs(cfg.out_dir, exist_ok=True)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total = 0
        correct = 0
        total_loss = 0.0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.size(0)

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return {"loss": avg_loss, "acc": acc}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        best_val_acc = -1.0
        best_path = os.path.join(self.cfg.out_dir, "best_model_task4.pt")

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "best_val_acc": None,
            "best_model_path": best_path,
        }

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            total = 0
            correct = 0
            total_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False)
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
                total_loss += loss.item() * y.size(0)

                pbar.set_postfix(loss=loss.item())

            train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)

            val_stats = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_stats["loss"])
            history["val_acc"].append(val_stats["acc"])

            print(
                f"Epoch {epoch:02d}/{self.cfg.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_stats['loss']:.4f} acc {val_stats['acc']:.4f}"
            )

            if val_stats["acc"] > best_val_acc:
                best_val_acc = val_stats["acc"]
                torch.save(self.model.state_dict(), best_path)

        history["best_val_acc"] = best_val_acc
        return history
