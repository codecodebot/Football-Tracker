"""Train a lightweight regression model for correcting projections."""

from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class MappingMLP(nn.Module):
    """
    Simple MLP that maps image-space coordinates to field-space coordinates.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(
    data_path: str,
    model_path: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    val_split: float,
) -> Tuple[MappingMLP, float]:
    """
    Train the correction model using logged (input, target) pairs.

    Args:
        data_path: Path to .npz file containing 'inputs' and 'targets'.
        model_path: Path to save the best model weights.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for Adam optimizer.
        val_split: Fraction of data used for validation.

    Returns:
        model: Trained MappingMLP model.
        best_val: Best validation loss achieved.
    """
    data = np.load(data_path)
    inputs = torch.tensor(data["inputs"], dtype=torch.float32)
    targets = torch.tensor(data["targets"], dtype=torch.float32)

    dataset = TensorDataset(inputs, targets)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
    )

    model = MappingMLP()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_val = float("inf")

    for _ in range(epochs):
        # Training
        model.train()
        for x_batch, y_batch in train_loader:
            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)

    return model, best_val


def load_model(model_path: str) -> MappingMLP:
    """
    Load a trained MappingMLP model from disk.

    Args:
        model_path: Path to saved model weights.

    Returns:
        Loaded MappingMLP model in eval mode.
    """
    model = MappingMLP()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
