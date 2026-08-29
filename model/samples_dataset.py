import numpy as np
import torch
from torch.utils.data import Dataset

class SamplesDataset(Dataset):
    def __init__(self, x_path, y_path=None, tps=8, label_at="end",
                 mmap=True, dtype_x=np.float16, dtype_y=np.float32,
                 normalize=True):
        self.x_path = x_path
        self.y_path = y_path
        self._mmap = mmap
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y
        self.tps = int(tps)
        self.label_at = label_at
        self.normalize = normalize

        # 懒加载对象
        self._X = None
        self._Y = None

        # 临时打开一次读取 shape
        X = np.load(self.x_path, mmap_mode='r' if mmap else None)
        self.num_days, self.num_hours, self.C, self.H, self.W = X.shape
        print(f"数据集X形状: {X.shape} ")
        del X
        assert self.num_hours >= self.tps
        self.win_per_day = self.num_hours - self.tps + 1
        self.total = self.num_days * self.win_per_day
        print(f"总样本数: {self.total} (天数: {self.num_days}, 每天窗口数: {self.win_per_day})")

        # 如果需要归一化，先计算 mean/std
        if normalize:
            self.x_mean, self.x_std = self._compute_x_stats()
            if y_path is not None:
                self.y_mean, self.y_std = self._compute_y_stats()
            else:
                self.y_mean, self.y_std = None, None

    def _lazy_open(self):
        if self._X is None:
            self._X = np.load(self.x_path, mmap_mode='r' if self._mmap else None)
        if self.y_path is not None and self._Y is None:
            self._Y = np.load(self.y_path, mmap_mode='r' if self._mmap else None)

    def _compute_x_stats(self):
        # print("计算 X 通道归一化参数（去 0 值）...")
        X = np.load(self.x_path, mmap_mode='r' if self._mmap else None)
        mean = np.zeros(self.C, dtype=np.float32)
        std = np.zeros(self.C, dtype=np.float32)
        for c in range(self.C):
            data_c = X[0, :, c, :, :].ravel()
            data_c = data_c[data_c != 0]  # 去掉 0 值
            mean[c] = data_c.mean()
            std[c] = data_c.astype(np.float32).std()
            std[c] = np.inf   # 控制
        std[std == 0] = 1.0
        print(f"X 通道均值: {mean}, 标准差: {std}")
        del X
        return mean, std

    def _compute_y_stats(self):
        # print("计算 Y 归一化参数（去 0 值）...")
        Y = np.load(self.y_path, mmap_mode='r' if self._mmap else None)
        print(f"Y 形状: {Y.shape}")
        data = Y.ravel()
        data = data[data != 0]
        mean = data.mean()
        std = data.astype(np.float32).std()
        if std == 0:
            std = 1.0
        del Y
        print(f"Y 均值: {mean}, 标准差: {std}")
        return mean, std

    def __len__(self):
        return self.total

    def _label_hour_idx(self, start):
        if self.label_at == "start":
            return start
        elif self.label_at == "center":
            return start + (self.tps // 2)
        return start + self.tps - 1  # 默认 end

    def __getitem__(self, idx):
        self._lazy_open()

        day = idx // self.win_per_day
        start = idx % self.win_per_day
        end = start + self.tps

        x_np = np.asarray(self._X[day, start:end])  # (tps, C, H, W)
        if self.dtype_x is not None and x_np.dtype != self.dtype_x:
            x_np = x_np.astype(self.dtype_x, copy=False)

        # Transpose to (C, H, W, tps) for output shape [3, 256, 256, 8]
        x_np = np.transpose(x_np, (1, 2, 3, 0))  # (C, H, W, tps)

        if self.normalize:
            # Reshape mean/std for broadcasting: (C, 1, 1, 1)
            x_np = (x_np - self.x_mean.reshape(self.C, 1, 1, 1)) / self.x_std.reshape(self.C, 1, 1, 1)

        x = torch.from_numpy(x_np)  # shape: (C, H, W, tps)

        if self._Y is None:
            return x

        # h = self._label_hour_idx(start)
        y_np = np.asarray(self._Y[idx]).reshape(-1)  # (num_rs,) 200
        if self.dtype_y is not None and y_np.dtype != self.dtype_y:
            y_np = y_np.astype(self.dtype_y, copy=False)

        if self.normalize and self.y_mean is not None:
            # y_np = (y_np - self.y_mean) / self.y_std
            y_np = y_np / self.y_std

        y = torch.from_numpy(y_np)
        return x, y

if __name__ == "__main__":
    X_PATH = "dataset/samples_20250818.npy"
    Y_PATH = "dataset/labels_20250818.npy"
    TPS = 8
    dataset = SamplesDataset(X_PATH, Y_PATH, tps=TPS, label_at="end", mmap=True)