from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    val_split: float = 0.1
    seed: int = 42
    use_full_train: bool = True  # enunciado: usar os 60k


def get_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Normalização comum para MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_full = datasets.MNIST(root=cfg.data_root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=cfg.data_root, train=False, download=True, transform=transform)

    # Split treino/val
    val_len = int(len(train_full) * cfg.val_split)
    train_len = len(train_full) - val_len

    g = torch.Generator().manual_seed(cfg.seed)
    train, val = random_split(train_full, lengths=[train_len, val_len], generator=g)

    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
