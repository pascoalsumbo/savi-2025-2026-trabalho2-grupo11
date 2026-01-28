from typing import Dict, Any
import numpy as np

import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)


@torch.no_grad()
def predict_all(model: torch.nn.Module, loader, device: str):
    model.eval()
    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()

        y_true.append(y.numpy())
        y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred


def compute_metrics(y_true, y_pred) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)

    # por classe
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(10)), average=None, zero_division=0
    )

    # macro global
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    return {
        "confusion_matrix": cm,
        "per_class": {"precision": p, "recall": r, "f1": f1, "support": support},
        "macro": {"precision": p_macro, "recall": r_macro, "f1": f1_macro},
        "report_text": report
    }
