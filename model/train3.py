# K-fold 交叉验证训练脚本（按论文逻辑）
# - 直接对“完整数据集”做 K 折
# - 每个实验（折）都从头初始化模型
# - 每个实验只在该折的“验证集”上计算指标（无独立 Test）
# - 将 K 个实验的验证指标做均值/方差作为最终报告
# - 归一化在 SamplesDataset 内部完成（建议用全数据或按你已有实现）

import os
import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from res3D import Res3D
from pathFormerGNN import PathFormerGNN
from samples_dataset import SamplesDataset  # 输出形状 (tps, 3, H, W)，Y 已在内部做归一化（如需）

# ================== 全局配置 ==================
X_PATH = "dataset/samples_20250820.npy"
Y_PATH = "dataset/labels_20250820.npy"
TPS = 4
MODEL_NAME = "PathFormerGNN"
# MODEL_NAME = "Res3D" 

LOG_ROOT = Path("./logs")
TB_DIR = LOG_ROOT / "tb_logs_kfold" / MODEL_NAME

K_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 10
MIXED_PRECISION = True
SEED = 42

# Windows 更稳妥：0；Linux 可调高
NUM_WORKERS = 0

# ================== 实用函数 ==================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def build_model(in_channels: int, out_features: int) -> nn.Module:

    if MODEL_NAME == "Res3D":
        return Res3D(in_channels, out_features)
    elif MODEL_NAME == "PathFormerGNN":
        return PathFormerGNN(hidden=64, tcn_layers=2, depth=3, P=200)
    else:
        raise ValueError(f"未知模型: {MODEL_NAME}")
    
def save_model(model: nn.Module, path: Path):
    """保存模型参数到指定路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")

# ---------- 指标 ----------
@torch.no_grad()
def eval_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon = 0.01) -> Tuple[float, float]:
    """返回 (MAE, RMSE)，y 可为 (B,) 或 (B,D)。"""
    y = y_true.view(-1).float(); p = y_pred.view(-1).float()
    mae = torch.sum(torch.abs(y - p) * y) / torch.sum(y)
    rmse = torch.sqrt(torch.sum(((y - p) ** 2) * y) / torch.sum(y)).item()
    mape = torch.sum(torch.abs(y - p)) / torch.sum(y)
    mspe = torch.sum((y - p) ** 2) / torch.sum(y ** 2)
    mae = mae.item(); mape = mape.item(); mspe = mspe.item()
    return mae, rmse, mape, mspe

# ================== 核心：单个 epoch 的训练 / 验证 ==================

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    criterion: nn.Module,
                    device: torch.device,
                    scaler: torch.amp.GradScaler | None,
                    global_step: int,
                    writer: SummaryWriter | None,
                    tag: str) -> Tuple[float, int]:
    """训练单个 epoch。返回 (epoch_avg_loss, global_step_after)."""
    model.train(); running = 0.0
    for xb, yb in tqdm(loader, desc=f"{tag}", unit="batch"):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float().view(xb.size(0), -1)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=MIXED_PRECISION):
            pred = model(xb)
            if pred.ndim > 2: pred = pred.view(pred.size(0), -1)
            loss = criterion(pred, yb)
        if scaler is not None:
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
        # scheduler.step()
        running += float(loss.item())
        if writer is not None:
            writer.add_scalar(f'{tag}/train_step_loss', float(loss.item()), global_step)
        
        global_step += 1
    return running / max(1, len(loader)), global_step

@torch.no_grad()
def validate_one_epoch(model: nn.Module,
                       loader: DataLoader,
                       criterion: nn.Module,
                       device: torch.device,
                       tag: str,
                       writer: SummaryWriter,
                       y_mean, y_std) -> Dict[str, float]:

    model.eval(); loss_sum = 0.0; n_batches = 0
    mae_sum, rmse_sum, mape_sum, mspe_sum = 0.0, 0.0, 0.0, 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float().view(xb.size(0), -1)
        with torch.amp.autocast("cuda", enabled=MIXED_PRECISION):
            pred = model(xb)
            if pred.ndim > 2: pred = pred.view(pred.size(0), -1)
            loss = criterion(pred, yb)
        loss_sum += float(loss.item()); n_batches += 1
        # 反归一化后计算指标
        if y_mean is not None and y_std is not None:
            pred = torch.abs(pred * y_std)
            yb = yb * y_std
        mae, rmse, mape, mspe = eval_metrics(yb, pred, epsilon=0.01)
        mae_sum += mae; rmse_sum += rmse; mape_sum += mape; mspe_sum += mspe
    out = {
        "val_loss": loss_sum / max(1, n_batches),
        "val_mae": mae_sum / max(1, n_batches),
        "val_rmse": rmse_sum / max(1, n_batches),
        "val_mape": mape_sum / max(1, n_batches),
        "val_mspe": mspe_sum / max(1, n_batches)
    }
    # print(f"[{tag}] {out}")
    if writer is not None:
        writer.add_scalar(f'{tag}/val_loss', out["val_loss"]) 
        writer.add_scalar(f'{tag}/val_mae',  out["val_mae"]) 
        writer.add_scalar(f'{tag}/val_rmse', out["val_rmse"]) 
        writer.add_scalar(f'{tag}/val_mape', out["val_mape"])
        writer.add_scalar(f'{tag}/val_mspe', out["val_mspe"])
    return out

def plot_val_linear_fit(model, val_loader, device, y_mean, y_std, fold, epoch, MODEL_NAME, LOG_ROOT):
    model.eval()
    all_preds, all_trues = [], []
    for xb, yb in val_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float().view(xb.size(0), -1)
        with torch.amp.autocast("cuda", enabled=MIXED_PRECISION):
            pred = model(xb)
            if pred.ndim > 2: pred = pred.view(pred.size(0), -1)
        # 反归一化
        if y_mean is not None and y_std is not None:
            pred = pred * y_std
            yb = yb * y_std
        all_preds.append(pred.detach().cpu().numpy())
        all_trues.append(yb.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 1)
    all_trues = np.concatenate(all_trues, axis=0).reshape(-1, 1)

    reg = LinearRegression().fit(all_preds, all_trues)
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    r2 = reg.score(all_preds, all_trues)

    plt.figure(figsize=(6,6))
    plt.scatter(all_preds, all_trues, s=8, alpha=0.3, label="Samples")
    x_line = np.linspace(all_preds.min(), all_preds.max(), 100)
    plt.plot(x_line, reg.predict(x_line.reshape(-1,1)), 'r-', label=f'Fit: y={slope:.3f}x+{intercept:.3f}, $R^2$={r2:.3f}')
    plt.plot(x_line, x_line, 'g--', label='Ideal: y=x')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Fold {fold+1} Linear Fit (val set)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(LOG_ROOT, exist_ok=True)
    fig_path = LOG_ROOT / f"{MODEL_NAME}_fold{fold+1}_epoch{epoch}_val_fit.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"线性回归图已保存到 {fig_path}")
    
# ================== 单个实验（单折） ==================

def run_one_experiment(fold: int,
                       full_ds: SamplesDataset,
                       tr_idx: np.ndarray,
                       va_idx: np.ndarray,
                       device: torch.device,
                       writer: SummaryWriter | None) -> Dict[str, Any]:
    # —— 构建 DataLoader
    train_loader = DataLoader(Subset(full_ds, tr_idx), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(Subset(full_ds, va_idx),   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # —— 模型/优化器（每个实验从头初始化）
    model = build_model(in_channels=3, out_features=200).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.amp.GradScaler('cuda', enabled=MIXED_PRECISION)
    y_mean, y_std = full_ds.y_mean, full_ds.y_std
    tag = f"fold{fold+1}"
    global_step = 0
    best_val = float('inf'); best_epoch = -1; history = []

    for epoch in range(NUM_EPOCHS):
        tr_loss, global_step = train_one_epoch(model, train_loader, optimizer, scheduler,
                                               criterion, device, scaler, global_step, writer, tag)
        val_out = validate_one_epoch(model, val_loader, criterion, device, tag, writer, y_mean, y_std)
        print(f"[Fold {fold+1}] Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {tr_loss:.6f}, {val_out}")
        history.append({"epoch": epoch+1, "train_loss": tr_loss, **val_out})
        if writer is not None:
            writer.add_scalar(f'{tag}/train_epoch_loss', tr_loss, epoch+1)
        if val_out["val_loss"] < best_val:
            best_val, best_epoch = val_out["val_loss"], epoch + 1
        if epoch % 10 == 0 :
            save_model(model, LOG_ROOT / f"best_model_{MODEL_NAME}_fold{fold+1}.pth")
            plot_val_linear_fit(model, val_loader, device, y_mean, y_std, fold, epoch, MODEL_NAME, LOG_ROOT/f"{MODEL_NAME}_folfd{fold+1}_val")
    
    # 取验证集上所有预测和真实值，做线性回归拟合并画图
    plot_val_linear_fit(model, val_loader, device, y_mean, y_std, fold, epoch, MODEL_NAME, LOG_ROOT)

    print(f"[Fold {fold+1}] Best ValLoss={best_val:.6f} @ epoch {best_epoch}")
    return {"best_val": best_val, "best_epoch": best_epoch, "history": history}

# ================== K 折调度 ==================

def run_kfold():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TB_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(TB_DIR))

    # —— 构建完整数据集（归一化在 Dataset 内部完成）
    full_ds = SamplesDataset(x_path=X_PATH, y_path=Y_PATH, tps=TPS,
                             label_at="end", mmap=True,
                             dtype_x=np.float16, dtype_y=np.float32)

    all_idx = np.arange(len(full_ds))
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    results = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(all_idx)):
        exp = run_one_experiment(fold, full_ds, tr_idx, va_idx, device, writer)
        results.append(exp)
        # 保存每折的详细结果
        fold_dir = LOG_ROOT / "kfold_fold_results"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / f"fold{fold+1}_{MODEL_NAME}.json", "w", encoding="utf-8") as f:
            json.dump(exp, f, ensure_ascii=False, indent=2)
        hist_df = pd.DataFrame(exp["history"])
        hist_df.to_csv(fold_dir / f"fold{fold+1}_{MODEL_NAME}_history.csv", index=False, encoding="utf-8")
        print(f"Fold {fold+1} results saved to {fold_dir / f'fold{fold+1}_{MODEL_NAME}.json'}")
    

    writer.close()

    # 汇总 K 折结果
    best_vals = np.array([r["best_val"] for r in results], dtype=float)
    summary = {
        "fold_best_vals": best_vals.tolist(),
        "mean_best_val": float(best_vals.mean()),
        "std_best_val": float(best_vals.std(ddof=0)),
        "config": {
            "k_folds": K_FOLDS, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "model": MODEL_NAME,
        }
    }
    print("=== K-Fold Summary ===")
    print(json.dumps({k: (v if not isinstance(v, float) else round(v, 6)) for k, v in summary.items()}, indent=2, ensure_ascii=False))

    # 保存到文件
    out_dir = LOG_ROOT / "kfold_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"summary_{MODEL_NAME}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_kfold()
