import os
import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from samples_dataset import SamplesDataset  # (num_days, num_hours, 3, N, N) → (tps, 3, N, N)

# ================== 配置 ==================
X_PATH = "dataset/samples_20250813_0.npy"    # SamplesDataset 懒加载
Y_PATH = "dataset/labels_20250813_0.npy"              # 若无标签可置 None
TPS = 8
MODEL_NAME = "PathFormerGNN"               # 或 "Res3D"
LOG_ROOT = Path("./logs")
CKpt_ROOT = LOG_ROOT / "checkpoints" / MODEL_NAME
TB_DIR = LOG_ROOT / "tb_logs" / MODEL_NAME

BATCH_SIZE = 32
NUM_EPOCHS = 100
K_FOLDS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
MIXED_PRECISION = True
SAVE_INTERVAL = 25                          # 仅在间隔时保存“最新模型”；最后一个 epoch 再保存“最佳模型”
RESUME = True                               # 自动从最近的间隔快照/last.pth 恢复
RESUME_PATH = None                          # 指定断点优先
SEED = 42


# ================== 工具函数 ==================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def fold_dir(root: Path, fold: int) -> Path:
    d = root / f"fold_{fold+1}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_ckpt(path: Path, *, fold: int, epoch: int, best_val: float, best_epoch: int,
              model: nn.Module, optimizer, scheduler,
              train_losses, val_losses,
              train_idx, val_idx):
    torch.save({
        "meta": {"fold": fold, "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "indices": {"train_idx": np.asarray(train_idx), "val_idx": np.asarray(val_idx)},
    }, path)


def load_ckpt(path: Path, model: nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"]) 
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"]) 
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"]) 
    m = ckpt["meta"]
    return m["fold"], m["epoch"] + 1, m["best_val"], m.get("best_epoch", -1), ckpt["train_losses"], ckpt["val_losses"], ckpt["indices"]


def latest_snapshot_path(fdir: Path) -> Path | None:
    """返回该 fold 下最近的快照：优先 RESUME_PATH，其次 last.pth，再次按 epoch 号最大的一份。"""
    snap_candidates = []
    last_p = fdir / "last.pth"
    if last_p.exists():
        snap_candidates.append(last_p)
    for p in fdir.glob("snapshot_epoch*.pth"):
        snap_candidates.append(p)
    if not snap_candidates:
        return None
    # 解析 epoch 号排序（last.pth 放最后）
    def key(p: Path):
        if p.name.startswith("snapshot_epoch"):
            try:
                return int(p.stem.split("epoch")[-1])
            except Exception:
                return -1
        return 10**9
    snap_candidates.sort(key=key)
    return snap_candidates[-1]


# ================== 模型 ==================

def build_model(in_channels: int, out_features: int) -> nn.Module:
    from res3D import Res3D
    from pathFormerGNN import PathFormerGNN
    if MODEL_NAME == "Res3D":
        return Res3D(in_channels, out_features)
    elif MODEL_NAME == "PathFormerGNN":
        return PathFormerGNN(hidden=64, tcn_layers=2, depth=3, P=200)
    else:
        raise ValueError(f"未知模型: {MODEL_NAME}")


# ================== 单折训练 ==================

def train_one_fold(
    fold: int,
    full_ds: SamplesDataset,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    device: torch.device,
    writer: SummaryWriter,
) -> Dict[str, Any]:

    # 推断模型输入/输出维度（假定 y 为回归向量/标量）
    x0, y0 = full_ds[0]
    in_channels = x0.shape[1]  # (tps, 3, N, N) → C=3
    out_features = int(np.prod(y0.shape)) if isinstance(y0, torch.Tensor) else 1

    model = build_model(in_channels, out_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.amp.GradScaler('cuda', enabled=MIXED_PRECISION)

    train_loader = DataLoader(Subset(full_ds, tr_idx), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=min(4, os.cpu_count() or 1), pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(Subset(full_ds, va_idx), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=min(4, os.cpu_count() or 1), pin_memory=True,
                            persistent_workers=True, prefetch_factor=2)

    fdir = fold_dir(CKpt_ROOT, fold)

    # 断点恢复（仅从最近的“间隔/last”快照恢复）
    start_epoch, best_val, best_epoch = 0, float("inf"), -1
    train_losses: list[float] = []
    val_losses: list[float] = []

    resume_path = Path(RESUME_PATH) if RESUME_PATH else latest_snapshot_path(fdir)
    if RESUME and resume_path and resume_path.exists():
        _, start_epoch, best_val, best_epoch, train_losses, val_losses, _ = load_ckpt(
            resume_path, model, optimizer, scheduler
        )
        print(f"[Fold {fold+1}] Resumed from {resume_path} | start_epoch={start_epoch}, best_val={best_val:.6f}")

    global_step = start_epoch * max(1, len(train_loader))

    for epoch in range(start_epoch, NUM_EPOCHS):
        # ===== Train =====
        model.train(); running = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}", unit="batch"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float().view(xb.size(0), -1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=MIXED_PRECISION):
                pred = model(xb)
                if pred.ndim > 2:
                    pred = pred.view(pred.size(0), -1)
                loss = criterion(pred, yb)

            if MIXED_PRECISION:
                scaler.scale(loss).backward()
                if GRAD_CLIP and GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP and GRAD_CLIP > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            scheduler.step()
            running += float(loss.item())
            writer.add_scalar(f'fold{fold+1}/train_step_loss', float(loss.item()), global_step); global_step += 1

        # ===== Val =====
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float().view(xb.size(0), -1)
                with torch.amp.autocast("cuda", enabled=MIXED_PRECISION):
                    pv = model(xb)
                    if pv.ndim > 2:
                        pv = pv.view(pv.size(0), -1)
                    vloss += float(criterion(pv, yb).item())

        tr = running / max(1, len(train_loader))
        va = vloss / max(1, len(val_loader))
        train_losses.append(tr); val_losses.append(va)
        writer.add_scalar(f'fold{fold+1}/train_epoch_loss', tr, epoch+1)
        writer.add_scalar(f'fold{fold+1}/val_epoch_loss', va, epoch+1)
        print(f"[Fold {fold+1}] Epoch {epoch+1}/{NUM_EPOCHS} | Train {tr:.4f} | Val {va:.4f}")

        # 仅在间隔时保存“最新模型”（覆盖 last.pth，并保留一份带 epoch 的快照）
        if SAVE_INTERVAL and (epoch + 1) % SAVE_INTERVAL == 0:
            snap = fdir / f"snapshot_epoch{epoch+1}.pth"
            save_ckpt(snap, fold=fold, epoch=epoch, best_val=best_val, best_epoch=best_epoch,
                      model=model, optimizer=optimizer, scheduler=scheduler,
                      train_losses=train_losses, val_losses=val_losses,
                      train_idx=tr_idx, val_idx=va_idx)
            save_ckpt(fdir / "last.pth", fold=fold, epoch=epoch, best_val=best_val, best_epoch=best_epoch,
                      model=model, optimizer=optimizer, scheduler=scheduler,
                      train_losses=train_losses, val_losses=val_losses,
                      train_idx=tr_idx, val_idx=va_idx)

        # 更新最佳指标（但不立刻落盘，最后一个 epoch 再保存最佳模型）
        if va < best_val:
            best_val, best_epoch = va, epoch + 1

    # ===== 训练结束：这里才保存“最佳模型” =====
    best_path = fdir / "best.pth"
    save_ckpt(best_path, fold=fold, epoch=best_epoch - 1 if best_epoch > 0 else NUM_EPOCHS - 1,
              best_val=best_val, best_epoch=best_epoch,
              model=model, optimizer=optimizer, scheduler=scheduler,
              train_losses=train_losses, val_losses=val_losses,
              train_idx=tr_idx, val_idx=va_idx)
    print(f"[Fold {fold+1}] Final best saved: val={best_val:.6f} @ epoch {best_epoch}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val": best_val,
        "best_epoch": best_epoch,
    }

# ================== 入口（主控只调度，不塞满 main） ==================

def run():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CKpt_ROOT.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TB_DIR))

    full_ds = SamplesDataset(
        x_path=X_PATH,
        y_path=Y_PATH,
        tps=TPS,
        label_at="end",
        mmap=True,
        dtype_x=np.float32,
        dtype_y=np.float32,
    )

    num_samples = len(full_ds)
    all_idx = np.arange(num_samples)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    history = {}
    for fold, (tr_idx, va_idx) in enumerate(kf.split(all_idx)):
        print(f"=== Fold {fold+1}/{K_FOLDS} ===")
        h = train_one_fold(fold, full_ds, tr_idx, va_idx, device, writer)
        history[f"fold_{fold+1}"] = h

    writer.close()
    with open(CKpt_ROOT / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print("训练完成，日志保存在:", CKpt_ROOT)

if __name__ == "__main__":
    run()