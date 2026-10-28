"""Supervised BPTT training with surrogate gradients."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import functional as SF
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import Config
from .data import FallDataset, shifted_eval_set
from .evaluate import compute_metrics
from .model import SCNN

LOG_COLUMNS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
               "val_precision", "val_recall", "val_f1", "val_auc"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_sets(data: dict, cfg: Config) -> dict[str, dict]:
    """Val/test arrays: falls re-cut at eval.fall_offsets_s, plus the aligned test set."""
    sets = {}
    for s in ("val", "test"):
        if cfg.eval.fall_offsets_s:
            sets[s] = shifted_eval_set(data, s, cfg.eval.fall_offsets_s, cfg.data.sample_rate)
        else:
            sets[s] = {"x": data[f"x_{s}"], "fs1": data[f"fs1_{s}"], "y": data[f"y_{s}"]}
    sets["test_aligned"] = {"x": data["x_test"], "fs1": data["fs1_test"], "y": data["y_test"]}
    return sets


def make_loaders(data: dict, cfg: Config) -> dict[str, DataLoader]:
    """Train loader is class-balanced with a WeightedRandomSampler; val/test are ordered."""
    t = cfg.train
    y_train = torch.from_numpy(data["y_train"])
    weights = (1.0 / torch.bincount(y_train).double())[y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(cfg.seed))
    shift = None
    if t.fall_shift_s:
        lo, hi = (int(round(v * cfg.data.sample_rate)) for v in t.fall_shift_s)
        shift = (data, "train", (lo, hi))
    train_set = FallDataset(data["x_train"], data["fs1_train"], data["y_train"], t.noise_std, shift)
    loaders = {"train": DataLoader(train_set, batch_size=t.batch_size, sampler=sampler, drop_last=True,
                                   num_workers=t.num_workers)}
    for name, arr in eval_sets(data, cfg).items():
        loaders[name] = DataLoader(FallDataset(arr["x"], arr["fs1"], arr["y"]),
                                   batch_size=t.batch_size * 4, shuffle=False, num_workers=t.num_workers)
    return loaders


def make_optimizer(model: SCNN, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with no weight decay on biases and a separate LR for the level-crossing thresholds."""
    t = cfg.train
    threshold = model.encoders["lc"].threshold
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p is threshold:
            continue
        (no_decay if name.endswith("bias") else decay).append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": t.weight_decay, "lr": t.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": t.lr},
        {"params": [threshold], "weight_decay": 0.0, "lr": t.threshold_lr},
    ])


class SpikingLoss(nn.Module):
    """Weighted sum of CE on the final output membrane and CE on output spike rates."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.w_mem, self.w_rate = cfg.train.loss_mem_weight, cfg.train.loss_rate_weight
        self.mem_ce = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
        self.rate_ce = SF.ce_rate_loss()

    def forward(self, spk_out, mem_out, y):
        loss = self.w_mem * self.mem_ce(mem_out, y)
        if self.w_rate > 0:
            loss = loss + self.w_rate * self.rate_ce(spk_out, y)
        return loss


def logits_from_outputs(spk_out: torch.Tensor, mem_out: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Decision logits, weighted the same way as the training loss."""
    return cfg.train.loss_mem_weight * mem_out + cfg.train.loss_rate_weight * spk_out.mean(0)


@torch.no_grad()
def predict(model: SCNN, loader: DataLoader, device: torch.device, cfg: Config,
            criterion: nn.Module | None = None):
    """Returns (y_true, P(fall), mean loss or nan)."""
    model.eval()
    ys, probs, loss_sum = [], [], 0.0
    for x, fs1, y in loader:
        x, fs1, y = x.to(device), fs1.to(device), y.to(device)
        spk_out, mem_out = model(x, fs1)
        if criterion is not None:
            loss_sum += criterion(spk_out, mem_out, y).item() * len(y)
        probs.append(F.softmax(logits_from_outputs(spk_out, mem_out, cfg), dim=1)[:, 1].cpu())
        ys.append(y.cpu())
    y_true, prob = torch.cat(ys).numpy(), torch.cat(probs).numpy()
    return y_true, prob, (loss_sum / len(y_true) if criterion is not None else float("nan"))


def fit(model: SCNN, loaders: dict, cfg: Config, device: torch.device, run_dir: Path) -> dict:
    """Train with early stopping on validation F1; the best checkpoint goes to run_dir/best.pt."""
    t = cfg.train
    criterion = SpikingLoss(cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t.scheduler_factor, patience=t.scheduler_patience, min_lr=t.min_lr)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(LOG_COLUMNS)

    best = {"val_f1": -1.0, "epoch": 0}
    stale = 0
    for epoch in range(1, t.epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch}/{t.epochs}", leave=False)
        for x, fs1, y in pbar:
            x, fs1, y = x.to(device), fs1.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            spk_out, mem_out = model(x, fs1)
            loss = criterion(spk_out, mem_out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()

            loss_sum += loss.item() * len(y)
            correct += (logits_from_outputs(spk_out, mem_out, cfg).argmax(1) == y).sum().item()
            seen += len(y)
            pbar.set_postfix(loss=f"{loss_sum / seen:.4f}", acc=f"{correct / seen:.3f}")

        y_val, p_val, val_loss = predict(model, loaders["val"], device, cfg, criterion)
        m = compute_metrics(y_val, p_val, threshold=0.5)
        lr = optimizer.param_groups[0]["lr"]
        row = [epoch, lr, loss_sum / seen, correct / seen, val_loss,
               m["accuracy"], m["precision"], m["recall"], m["f1"], m["auc"]]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])

        thr = model.encoders["lc"].threshold.detach().cpu().numpy()
        print(f"Epoch {epoch:3d} | lr {lr:.1e} | train loss {loss_sum / seen:.4f} acc {correct / seen:.3f} | "
              f"val loss {val_loss:.4f} acc {m['accuracy']:.3f} F1 {m['f1']:.3f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f} AUC {m['auc']:.3f} | "
              f"LC thr {np.array2string(thr, precision=2)}")

        scheduler.step(m["f1"])
        if m["f1"] > best["val_f1"]:
            best = {"val_f1": m["f1"], "epoch": epoch}
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), **best}, run_dir / "best.pt")
            stale = 0
        else:
            stale += 1
            if stale >= t.patience:
                print(f"Early stopping: no val F1 improvement for {t.patience} epochs.")
                break
    print(f"Best val F1 {best['val_f1']:.4f} at epoch {best['epoch']}")
    return best
