import os
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter   

from samples_dataset import SamplesDataset

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

# ================= 参数 =================
sample_path = 'dataset/samples_20250806.npz'
label_path = 'dataset/labels_20250806.npz'
MODEL_NAME = 'PathFormerGNN'  # 'Res3D' 或 'PathFormerGNN'
BATCH_SIZE = 32
NUM_EPOCHS = 100
K_FOLDS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
MIXED_PRECISION = False  # 是否启用混合精度训练
SAVE_INTERVAL = 25   # 每隔 N 轮保存“到目前为止最优”
RESUME = True
RESUME_PATH = None
SEED = 42

# ================= 工具函数 =================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def save_ckpt(path, fold, epoch, best_val, best_epoch, model, optimizer, scheduler, train_losses, val_losses, train_idx, val_idx):
    torch.save({
        'meta': dict(fold=fold, epoch=epoch, best_val=best_val, best_epoch=best_epoch),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'indices': dict(train_idx=train_idx.cpu().numpy(), val_idx=val_idx.cpu().numpy())
    }, path)

def load_ckpt(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    m = ckpt['meta']
    return m['fold'], m['epoch'] + 1, m['best_val'], m['best_epoch'], ckpt['train_losses'], ckpt['val_losses'], ckpt['indices']

def fold_dir(root, fold):
    d = root / f"fold_{fold+1}"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ================= 数据加载 =================
set_seed(SEED)
samples = np.load(sample_path)['samples']
samples = torch.tensor(samples).permute(0, 2, 3, 4, 1)
labels = torch.tensor(np.load(label_path)['labels'] / 100)

NUM_ALL = labels.shape[0]
train_idx_all = torch.randperm(NUM_ALL)[:int(0.9 * NUM_ALL)]
train_set, train_label = samples[train_idx_all], labels[train_idx_all]

# ================= 模型选择 =================
from res3D import Res3D
from pathFormerGNN import PathFormerGNN

in_channels, out_features = samples.shape[1], labels.shape[1]
model = Res3D(in_channels, out_features) if MODEL_NAME == 'Res3D' else PathFormerGNN(hidden=64, tcn_layers=2, depth=3, P=200)
model = model.to(device)

# ================= 训练流程 =================
ckpt_root = Path('./logs/checkpoints') / MODEL_NAME
writer = SummaryWriter(log_dir=str(Path('./logs/tb_logs') / MODEL_NAME))

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
history_all = {}

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_set)):
    tr_idx = torch.as_tensor(tr_idx)
    va_idx = torch.as_tensor(va_idx)
    print(f"\n=== Fold {fold+1}/{K_FOLDS} ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device_str,enabled=MIXED_PRECISION)

    start_epoch, best_val, best_epoch = 0, float('inf'), -1
    train_losses, val_losses = [], []

    resume_path = Path(RESUME_PATH) if RESUME_PATH else fold_dir(ckpt_root, fold) / 'last.pth'
    if resume_path.exists() and RESUME:
        _, start_epoch, best_val, best_epoch, train_losses, val_losses, _ = load_ckpt(resume_path, model, optimizer, scheduler)
        print(f"Resumed from {resume_path}, start_epoch={start_epoch}, best_val={best_val:.6f}")

    train_ds = TensorDataset(train_set[tr_idx], train_label[tr_idx])
    val_ds = TensorDataset(train_set[va_idx], train_label[va_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train(); running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}", unit="batch"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_str, enabled=MIXED_PRECISION):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if GRAD_CLIP:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            running_loss += loss.item()

            writer.add_scalar(f'fold{fold+1}/train_step_loss', loss.item(), global_step)
            global_step += 1

        # 验证
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_str,enabled=MIXED_PRECISION):
                    val_loss += criterion(model(x), y).item()

        train_epoch_loss = running_loss / len(train_loader)
        val_epoch_loss = val_loss / len(val_loader)
        train_losses.append(train_epoch_loss); val_losses.append(val_epoch_loss)

        writer.add_scalar(f'fold{fold+1}/train_epoch_loss', train_epoch_loss, epoch+1)
        writer.add_scalar(f'fold{fold+1}/val_epoch_loss', val_epoch_loss, epoch+1)

        # 保存 best
        if val_epoch_loss < best_val:
            best_val, best_epoch = val_epoch_loss, epoch+1
            # save_ckpt(fold_dir(ckpt_root, fold) / 'best.pth', fold, epoch, best_val, best_epoch, model, optimizer, scheduler, train_losses, val_losses, tr_idx, va_idx)
            print(f"→ New best: val={best_val:.6f} @ epoch {best_epoch}")

        # 区间快照
        if SAVE_INTERVAL and (epoch + 1) % SAVE_INTERVAL == 0:
            save_ckpt(fold_dir(ckpt_root, fold) / f'best_until_epoch{epoch+1}.pth', fold, epoch, best_val, best_epoch, model, optimizer, scheduler, train_losses, val_losses, tr_idx, va_idx)

        # 每个fold最后一个epoch再保存最好的
        save_ckpt(fold_dir(ckpt_root, fold) / 'best.pth', fold, epoch, best_val, best_epoch, model, optimizer, scheduler, train_losses, val_losses, tr_idx, va_idx)

    history_all[f'fold_{fold+1}'] = {'train_losses': train_losses, 'val_losses': val_losses, 'best_val': best_val, 'best_epoch': best_epoch}

writer.close()
json.dump(history_all, open(ckpt_root / 'training_history.json', 'w'), indent=2)
print("训练完成，日志保存在:", ckpt_root)
