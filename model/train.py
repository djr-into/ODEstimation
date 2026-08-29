import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False 

from res3D import Res3D  
from pathFormerGNN import PathFormerGNN
MODEL_NAME = 'PathFormerGNN'

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 200
K_FOLDS = 5
LEARNING_RATE = 3e-4
SWA_LR = 0.05
T_MAX = 100

# Load data
samples = np.load('dataset/samples_20250728.npz')['samples']
samples = torch.tensor(samples).permute(0, 2, 3, 4, 1)
flow_labels = np.load('dataset/labels_20250728.npz')['labels']
labels = flow_labels/100
labels = torch.tensor(labels)
print(f"样本形状: {samples.shape}, 标签形状: {labels.shape}")

# Split data into train and test sets
NUM_ALL = labels.shape[0]
NUM_TRAIN = int(0.9 * NUM_ALL)
NUM_TEST = NUM_ALL - NUM_TRAIN
print(f"总样本数: {NUM_ALL}, 训练样本数: {NUM_TRAIN}, 测试样本数: {NUM_TEST}")
# 随机打乱索引
indices = torch.randperm(NUM_ALL)
train_indices = indices[:NUM_TRAIN]
test_indices = indices[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
train_set = samples[train_indices]
test_set = samples[test_indices]
train_label = labels[train_indices]
test_label = labels[test_indices]

# Initialize model, loss function, optimizer, and schedulers
in_channels = samples.shape[1]  
out_features = labels.shape[1] 
if MODEL_NAME == 'Res3D':
    model = Res3D(in_channels, out_features)
elif MODEL_NAME == 'PathFormerGNN':
    model = PathFormerGNN(hidden=64, tcn_layers=2, depth=3, P=200)
else:
    raise ValueError(f"未知模型名称: {MODEL_NAME}")
print(f"模型结构: {model}")
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-8)
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

# K-Fold Cross Validation
kf = KFold(n_splits=K_FOLDS, shuffle=True)
train_losses = []
val_losses = []

# 使用 K-Fold 交叉验证进行训练和验证
for fold, (train_idx, val_idx) in enumerate(kf.split(train_set)):
    print(f"折 {fold + 1}/{K_FOLDS}")

    # 准备数据加载器
    X_train, X_val = train_set[train_idx], train_set[val_idx]
    y_train, y_val = train_label[train_idx], train_label[val_idx]
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        # 使用 tqdm 添加进度条
        with tqdm(train_loader, desc=f"折 {fold + 1} - 轮次 {epoch + 1}/{NUM_EPOCHS}", unit="batch") as t:
            for inputs, labels in t:
                optimizer.zero_grad()  # 清除梯度
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                running_loss += loss.item()

                # 如果达到 SWA 阶段，更新 SWA 模型参数并调整学习率
                if epoch >= NUM_EPOCHS * 0.8 :
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                # 更新进度条显示的损失
                t.set_postfix(loss=loss.item())

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        with torch.no_grad():  # 禁用梯度计算
            for val_inputs, val_labels in val_loader:
                val_outputs = model(val_inputs)  # 前向传播
                loss = criterion(val_outputs, val_labels)  # 计算验证损失
                val_loss += loss.item()

        # 记录训练和验证损失
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        print(f"轮次 [{epoch + 1}/{NUM_EPOCHS}], 训练损失: {running_loss / len(train_loader):.4f}, 验证损失: {val_loss / len(val_loader):.4f}")

    # 更新 SWA 模型的 BatchNorm 统计信息
    torch.optim.swa_utils.update_bn(train_loader, swa_model)

def plot_learning_curve(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
plot_learning_curve(train_losses, val_losses)

# Save the trained model
torch.save(model.state_dict(), f"{MODEL_NAME}_model.pth")
torch.save(swa_model.state_dict(), f"swa_{MODEL_NAME}_model.pth")

