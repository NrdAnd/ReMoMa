from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str | Path,
        patience: int = 15,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience = patience
        self.grad_clip = grad_clip
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> dict:
        history: dict[str, list] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
        }
        best_val_loss = float("inf")
        patience_cnt = 0

        epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="ep")
        for epoch in epoch_bar:
            tr_loss, tr_acc = self._train_epoch(train_loader)
            va_loss, va_acc = self._eval_epoch(val_loader)
            self.scheduler.step(va_loss)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                patience_cnt = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best.pt")
                tag = " *"
            else:
                patience_cnt += 1
                tag = ""

            epoch_bar.set_postfix(
                tr_loss=f"{tr_loss:.4f}", tr_acc=f"{tr_acc:.3f}",
                va_loss=f"{va_loss:.4f}", va_acc=f"{va_acc:.3f}",
            )
            tqdm.write(
                f"Ep {epoch:03d} | "
                f"train {tr_loss:.4f}/{tr_acc:.3f} | "
                f"val {va_loss:.4f}/{va_acc:.3f}{tag}"
            )

            if patience_cnt >= self.patience:
                tqdm.write(f"Early stopping at epoch {epoch}.")
                break

        return history

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = correct = total = 0

        for batch in tqdm(loader, desc="  train", leave=False, unit="batch"):
            batch = batch.to(self.device)
            out = self.model(batch)
            loss = self.criterion(out, batch.y)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            correct += (out.argmax(1) == batch.y).sum().item()
            total += batch.num_graphs

        return total_loss / total, correct / total

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = correct = total = 0

        for batch in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
            batch = batch.to(self.device)
            out = self.model(batch)
            loss = self.criterion(out, batch.y)

            total_loss += loss.item() * batch.num_graphs
            correct += (out.argmax(1) == batch.y).sum().item()
            total += batch.num_graphs

        return total_loss / total, correct / total

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in tqdm(loader, desc="  predict", leave=False, unit="batch"):
            batch = batch.to(self.device)
            all_preds.extend(self.model(batch).argmax(1).cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

        return np.array(all_preds), np.array(all_labels)

    def load_best(self) -> None:
        self.model.load_state_dict(
            torch.load(self.checkpoint_dir / "best.pt", map_location=self.device)
        )
