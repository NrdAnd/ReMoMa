from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
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
        static_graph_batching: bool = False,
        static_edge_index: torch.Tensor | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience = patience
        self.grad_clip = grad_clip
        self.static_graph_batching = static_graph_batching
        self.static_edge_index = static_edge_index
        if self.static_graph_batching and self.static_edge_index is None:
            raise ValueError("static_edge_index is required when static_graph_batching=True")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
    ) -> dict:
        history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "val_f1": [],
        }
        best_val_f1 = 0.0
        patience_cnt = 0

        epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="ep")
        for epoch in epoch_bar:
            tr_loss, tr_acc = self._train_epoch(train_loader)
            va_loss, va_acc, va_f1, va_cm = self._eval_epoch(val_loader)
            self.scheduler.step(va_loss)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            history["val_f1"].append(va_f1)

            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                patience_cnt = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best.pt")
                tag = " *"
            else:
                patience_cnt += 1
                tag = ""

            epoch_bar.set_postfix(
                tr_loss=f"{tr_loss:.4f}", tr_acc=f"{tr_acc:.3f}",
                va_loss=f"{va_loss:.4f}", va_f1=f"{va_f1:.3f}",
            )
            tqdm.write(
                f"Ep {epoch:03d} | "
                f"train {tr_loss:.4f}/{tr_acc:.3f} | "
                f"val {va_loss:.4f}/acc={va_acc:.3f}/f1={va_f1:.3f}{tag}"
            )
            tqdm.write(
                f"  CM        pred_down  pred_flat   pred_up\n"
                f"  true_down {va_cm[0,0]:>9,} {va_cm[0,1]:>9,} {va_cm[0,2]:>9,}\n"
                f"  true_flat {va_cm[1,0]:>9,} {va_cm[1,1]:>9,} {va_cm[1,2]:>9,}\n"
                f"  true_up   {va_cm[2,0]:>9,} {va_cm[2,1]:>9,} {va_cm[2,2]:>9,}"
            )

            if patience_cnt >= self.patience:
                tqdm.write(f"Early stopping at epoch {epoch}.")
                break

        return history

    def _forward_batch(self, batch):
        if self.static_graph_batching:
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            out = self.model.forward_static(x, self.static_edge_index)
            num_graphs = int(y.shape[0])
            return out, y, num_graphs

        batch = batch.to(self.device)
        out = self.model(batch)
        y = batch.y
        num_graphs = int(batch.num_graphs)
        return out, y, num_graphs

    def _train_epoch(self, loader) -> tuple[float, float]:
        self.model.train()
        total_loss = correct = total = 0

        for batch in tqdm(loader, desc="  train", leave=False, unit="batch"):
            out, y, num_graphs = self._forward_batch(batch)
            loss = self.criterion(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * num_graphs
            correct += (out.argmax(1) == y).sum().item()
            total += num_graphs

        return total_loss / total, correct / total

    @torch.no_grad()
    def _eval_epoch(self, loader) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0
        all_preds: list = []
        all_labels: list = []

        for batch in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
            out, y, num_graphs = self._forward_batch(batch)
            loss = self.criterion(out, y)

            total_loss += loss.item() * num_graphs
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        preds  = np.array(all_preds)
        labels = np.array(all_labels)
        avg_loss = total_loss / len(labels)
        acc = (preds == labels).mean()
        f1  = f1_score(labels, preds, average="macro", zero_division=0)
        cm  = confusion_matrix(labels, preds, labels=[0, 1, 2])
        return avg_loss, acc, f1, cm

    @torch.no_grad()
    def predict(self, loader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in tqdm(loader, desc="  predict", leave=False, unit="batch"):
            out, y, _ = self._forward_batch(batch)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        return np.array(all_preds), np.array(all_labels)

    def load_best(self) -> None:
        self.model.load_state_dict(
            torch.load(self.checkpoint_dir / "best.pt", map_location=self.device)
        )
