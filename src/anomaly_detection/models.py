from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DenoisingAutoencoder(nn.Module):
    """Symmetric feed-forward denoising autoencoder for tabular network flows."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= latent_dim:
            raise ValueError(f"input_dim ({input_dim}) must be greater than latent_dim ({latent_dim})")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        encoder_layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            encoder_layers.extend(
                [nn.Linear(previous, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)]
            )
            previous = width
        encoder_layers.extend([nn.Linear(previous, latent_dim), nn.GELU()])
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        previous = latent_dim
        for width in reversed(hidden_dims):
            decoder_layers.extend(
                [nn.Linear(previous, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)]
            )
            previous = width
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        noisy = x
        if self.training and noise_std > 0.0:
            noisy = x + torch.randn_like(x) * noise_std
        return self.decoder(self.encoder(noisy))

    def reconstruction_contributions(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            contrib = (x - recon).pow(2)
        return recon, contrib


@dataclass(frozen=True)
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


def train_autoencoder(
    model: DenoisingAutoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    noise_std: float,
    gradient_clip_norm: float,
    device: torch.device,
    num_workers: int = 0,
) -> TrainingHistory:
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("Training and validation arrays must not be empty.")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, min(5, patience // 3 or 1))
    )

    train_history: list[float] = []
    val_history: list[float] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch, noise_std=noise_std)
            loss = criterion(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            train_total += float(loss.item()) * len(batch)
            train_count += len(batch)

        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device, non_blocking=True)
                recon = model(batch)
                loss = criterion(recon, batch)
                val_total += float(loss.item()) * len(batch)
                val_count += len(batch)

        train_loss = train_total / max(train_count, 1)
        val_loss = val_total / max(val_count, 1)
        train_history.append(train_loss)
        val_history.append(val_loss)
        scheduler.step(val_loss)

        print(
            f"epoch {epoch:03d}/{max_epochs} | train_loss={train_loss:.6f} "
            f"| val_loss={val_loss:.6f}"
        )

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs (patience={patience}).")
                break

    if best_state is None:
        raise RuntimeError("Autoencoder training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    return TrainingHistory(train_history, val_history)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def save_autoencoder(model: DenoisingAutoencoder, path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_autoencoder(path: str | Path, device: torch.device) -> DenoisingAutoencoder:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    meta = checkpoint["metadata"]
    model = DenoisingAutoencoder(
        input_dim=int(meta["input_dim"]),
        hidden_dims=[int(x) for x in meta["hidden_dims"]],
        latent_dim=int(meta["latent_dim"]),
        dropout=float(meta["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model
