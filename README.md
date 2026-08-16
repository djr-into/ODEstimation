# ODEstimation

基于 GPS 轨迹数据与图神经网络的城市路网 OD 流量估计系统。

以太仓市路网为研究对象，构建出行路径集合，并通过深度学习模型预测各路径的交通流量分布。

---

## 目录结构

```
ODEstimation/
├── trajmap.ipynb              # 步骤一：GPS 数据处理与地图匹配
├── PathSetConstruction.ipynb  # 步骤二：路径集合构建与轨迹匹配
├── model/
│   ├── pathFormerGNN.py           # PathFormerGNN 模型（原始版本）
│   ├── pathFormerGNN_refactored.py# PathFormerGNN 模型（规范化注释版）
│   ├── res3D.ipynb                # Res3D 模型训练与评估
│   └── 最大流量生成树.ipynb        # 最大流量生成树分析
├── data/
│   ├── taicangNet/            # 太仓市路网数据（shapefile + 路径集 RS.pkl）
│   ├── rawdata/               # 原始数据（行程时间、节点/路段关系表）
│   ├── traj/                  # GPS 轨迹数据（pickle 格式）
│   ├── sumonet/               # SUMO 仿真网络文件
│   └── output/                # 地图匹配结果（match_res.csv）
├── logs/                      # 模型训练日志与性能曲线
│   ├── PFQ-GNN/               # PathFormerGNN 5 折交叉验证日志
│   ├── RES3D/                 # Res3D 5 折交叉验证日志
│   └── TCN*/                  # TCN 变体日志
├── tools/                     # 第三方工具（SUMO 相关库）
└── cache/                     # 中间计算缓存
```

---

## 整体流程

```
原始 GPS 轨迹
      │
      ▼
[trajmap.ipynb]
  行程切分 → 地图匹配（gotrackit）→ 生成 OD 点
      │
      ▼
[PathSetConstruction.ipynb]
  路径惩罚算法 → 生成 k=20 候选路径集 → 轨迹-路径匹配
      │
      ▼
[model/]
  深度学习训练（PathFormerGNN / Res3D / TCN）→ 路径流量预测
```

---

## 数据说明

| 文件 / 目录 | 说明 |
|---|---|
| `data/taicangNet/nodes.shp` | 路网节点（4,437 个） |
| `data/taicangNet/links.shp` | 路网路段（10,376 条） |
| `data/taicangNet/RS.pkl` | 预计算路径集（每 OD 对 20 条，816 MB） |
| `data/rawdata/timing.csv` | 路段行程时间数据（149 MB） |
| `data/output/match_res.csv` | 地图匹配结果（400 万+ 条记录） |
| `data/traj/` | 原始 GPS 轨迹（81,582 辆车） |

---

## 模型说明

### PathFormerGNN

输入：`[B, 3, N, N, T]` 的稠密张量

| 通道 | 含义 |
|---|---|
| 0 | 路段匹配流量 |
| 1 | 路段行程时间 |
| 2 | 节点流量广播平面（对角线存储） |

主要模块：

- **EdgeTimeEncoder_NoShift**：融合节点时序与边属性，通过门控 + TCN 生成边的时序表示
- **EdgeNodeEdgeBlock**：E→N→E 消息传递残差块，实现边特征的邻域感知更新
- **SoftPathReadoutLite**：可学习路径查询向量 + 软注意力，聚合出路径级别流量

输出：`[B, 200]`，200 条路径的预测流量

### Res3D

输入：`(3, 200, 200, 8)` 的 3D 张量，输出 233 条路径的流量预测。

---

## 评估指标

| 指标 | 说明 |
|---|---|
| WMAE | 加权平均绝对误差 |
| WMAPE | 加权平均绝对百分比误差 |
| RMSE | 均方根误差 |
| MAPE | 平均绝对百分比误差 |

所有模型均采用 5 折交叉验证，日志保存于 `logs/` 目录。

---

## 依赖环境

```
Python       >= 3.11
torch
torch_scatter
geopandas
osmnx
networkx
pandas
numpy
scikit-learn
gotrackit      # GPS 地图匹配
matplotlib
seaborn
```

SUMO 相关工具已包含在 `tools/` 目录中，无需单独安装。

---

## 快速开始

1. **地图匹配**：运行 `trajmap.ipynb`，处理原始 GPS 数据，生成 `data/output/match_res.csv`
2. **路径集构建**：运行 `PathSetConstruction.ipynb`，生成路径集并完成轨迹-路径匹配
3. **模型训练**：运行 `model/res3D.ipynb` 或直接调用 `model/pathFormerGNN_refactored.py` 中的 `PathFormerGNN` 类
