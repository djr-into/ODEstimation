import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Res3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), downsample=False):
        super(Res3DBlock, self).__init__()
        self.downsample = downsample
        # 主分支
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), 
                              stride=stride, padding=(1, 1, 0))  # 使用传入的stride
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 1), 
                              stride=(1, 1, 1), padding=(1, 1, 0))
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 残差连接 - 仅在需要时添加
        self.residual_connection = None
        if in_channels != out_channels or any(s > 1 for s in stride) or downsample:
            # 使用1x1卷积匹配通道数和空间尺寸
            conv_stride = stride if any(s > 1 for s in stride) else (1, 1, 1)
            self.residual_connection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=conv_stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        # 下采样层 - 仅在需要时添加
        self.downsample_layer = None
        if downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=(2, 2, 2), 
                         stride=(2, 2, 2), padding=0),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        

    def forward(self, x):
        identity = x
        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 残差连接
        if self.residual_connection is not None:
            identity = self.residual_connection(identity)
        out += identity
        out = self.relu(out)
        
        # 下采样
        if self.downsample and self.downsample_layer is not None:
            out = self.downsample_layer(out)
        
        return out

class Res3D(nn.Module):
    def __init__(self, in_channels, out_features):
        super(Res3D, self).__init__()
        # 初始卷积层 - 减少空间尺寸
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        )
        
        # 残差块
        self.block1 = Res3DBlock(32, 64, downsample=True)
        self.block2 = Res3DBlock(64, 64, downsample=False)
        self.block3 = Res3DBlock(64, 64, downsample=False)
        
        # 全局平均池化替代全连接层
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # 全连接层
        self.fc = nn.Linear(64, out_features)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 遍历所有子模块
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # He初始化（适合ReLU）
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                # BN层权重初始化为1，偏置为0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层缩小初始化范围
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    def forward(self, x):

        # 初始卷积和池化
        x = self.initial_conv(x)  # [B, 3, 256, 256, 8] -> [B, 32, 64, 64, 8]
        
        # 残差块
        x = self.block1(x)  # [B, 32, 64, 64, 8] -> [B, 32, 32, 32, 4]
        x = self.block2(x)  # [B, 32, 32, 32, 4] -> [B, 64, 16, 16, 2]
        x = self.block3(x)  # [B, 64, 16, 16, 2] -> [B, 64, 8, 8, 1]
        
        # 全局平均池化
        x = self.global_pool(x)  # [B, 64, 8, 8, 1] -> [B, 64, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]
        
        # Dropout和全连接层
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, T = 2, 16, 8
    X = torch.zeros(B, 3, N, N, T)

    # 随机构造合理数据
    edge_mask = (torch.rand(B, N, N) > 0.7)
    for b in range(B):
        edge_mask[b].fill_diagonal_(False)
        for t in range(T):
            X[b, 0, :, :, t] = (torch.rand(N, N) * edge_mask[b]).float()   # 匹配流量
            X[b, 1, :, :, t] = 0.1 + 0.2 * torch.rand(N, N)                # 行程时间
        node_series = torch.rand(N, T)
        for i in range(N):
            X[b, 2, i, i, :] = node_series[i]                               # 节点对角线广播

    model = Res3D(in_channels=3, out_features=200)
    print(model)
    # 输出参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    y, alpha_list = model(X)
    print("y:", y.shape)  # [B,200]
    print("alpha[0]:", alpha_list[0].shape)  # [200, E_0]