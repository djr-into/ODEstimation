"""
PathFormerGNN：基于图神经网络的路径流量预测模型

输入格式：X [B, 3, N, N, T]
  - 通道 0：匹配流量（边特征）
  - 通道 1：行程时间（边特征，用于门控）
  - 通道 2：节点流量广播平面（对角线存储节点时序）

输出格式：
  - y [B, P]：P 条路径的预测流量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from typing import List, Tuple


def recover_node_sequence_diag(node_plane_NNT: torch.Tensor) -> torch.Tensor:
    """从广播后的节点平面恢复节点时序（固定 diag）。

    Args:
        node_plane_NNT: [N, N, T]，节点 1 维特征广播到 N×N 平面，
                        节点 i 在时间步 t 的值存于 (i, i, t)。

    Returns:
        x_seq: [T, N, 1]，每个节点在各时间步的取值。
    """
    x_NT = node_plane_NNT.diagonal(dim1=0, dim2=1).contiguous()  # [T, N]
    return x_NT.unsqueeze(-1)  # [T, N, 1]


class EdgeTimeEncoder_NoShift(nn.Module):
    """边时序编码器：节点→边门控 + TCN。

    将节点时序特征和边属性融合，通过门控机制和时间卷积网络
    生成每条边的时序隐藏表示。

    Args:
        node_in:    节点输入特征维度，默认为 1。
        hidden:     隐藏层维度，默认为 64。
        tcn_layers: TCN 层数，使用指数膨胀因子 2^k，默认为 2。
    """

    def __init__(self, node_in: int = 1, hidden: int = 64, tcn_layers: int = 2):
        super().__init__()
        self.node_proj = nn.Linear(node_in, hidden)
        # 边特征：[匹配流量, 行程时间]
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        convs = []
        for b in range(tcn_layers):
            dil = 2 ** b
            convs += [
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=dil, dilation=dil),
                nn.ReLU(),
            ]
        self.tcn = nn.Sequential(*convs)

    def forward(
        self,
        x_seq_TN1: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_seq: List[torch.Tensor],
    ) -> torch.Tensor:
        """计算每条边的时序隐藏表示。

        Args:
            x_seq_TN1:     [T, N, 1]，节点时序特征。
            edge_index:    [2, E]，边的源/目节点索引（long）。
            edge_attr_seq: 长度为 T 的列表，每个元素形状 [E, 2]，
                           表示该时间步各边的 [匹配流量, 行程时间]。

        Returns:
            z_ETD: [E, T, hidden]，每条边的时序隐藏向量。
        """
        T, N, _ = x_seq_TN1.shape
        H_TNH = F.relu(self.node_proj(x_seq_TN1))           # [T, N, H]
        src = edge_index[0]                                  # [E]

        # 沿节点维索引源节点特征，避免越界
        h_src = H_TNH.index_select(dim=1, index=src).transpose(0, 1).contiguous()  # [E, T, H]

        ea_ET2 = torch.stack(edge_attr_seq, dim=1).contiguous()                     # [E, T, 2]
        gate = F.relu(self.edge_mlp(ea_ET2.view(-1, 2))).view(ea_ET2.size(0), T, -1)  # [E, T, H]

        z = gate * h_src                                     # [E, T, H]
        z = z.permute(0, 2, 1).contiguous()                  # [E, H, T]
        z = self.tcn(z)
        z = z.permute(0, 2, 1).contiguous()                  # [E, T, H]
        return z


class EdgeNodeEdgeBlock(nn.Module):
    """原图上的 E→N→E 残差块（轻量边到边卷积）。

    通过消息传递将边特征汇聚到节点，再将节点信息写回边，
    实现边特征的邻域感知更新。

    Args:
        hidden: 隐藏层维度。
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.lin_e_in = nn.Linear(hidden, hidden, bias=False)
        self.lin_e_out = nn.Linear(hidden, hidden, bias=False)
        self.update = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        z_EH: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """单步 E→N→E 消息传递。

        Args:
            z_EH:      [E, H]，当前边特征。
            src:       [E]，边的源节点索引。
            dst:       [E]，边的目标节点索引。
            num_nodes: 图中节点总数 N。

        Returns:
            z_out: [E, H]，更新后的边特征（含残差连接和层归一化）。
        """
        # E→N：分别聚合入边和出边
        m_in = self.lin_e_in(z_EH)                                                       # [E, H]
        m_out = self.lin_e_out(z_EH)                                                     # [E, H]
        h_in = scatter_add(m_in, dst, dim=0, dim_size=num_nodes)                         # [N, H]
        h_out = scatter_add(m_out, src, dim=0, dim_size=num_nodes)                       # [N, H]

        # 度归一化，防止高度数节点主导
        deg_in = torch.bincount(dst, minlength=num_nodes).clamp(min=1).to(z_EH.dtype).unsqueeze(-1)
        deg_out = torch.bincount(src, minlength=num_nodes).clamp(min=1).to(z_EH.dtype).unsqueeze(-1)
        h_in = h_in / deg_in
        h_out = h_out / deg_out

        # N→E：将节点信息写回边
        z_msg = torch.cat([z_EH, h_out.index_select(0, src), h_in.index_select(0, dst)], dim=-1)  # [E, 3H]
        z_new = self.update(z_msg)
        return self.norm(z_EH + z_new)


class SoftPathReadoutLite(nn.Module):
    """软路径读出层（不建线图，E→N→E 平滑鼓励路径连续性）。

    使用可学习的路径查询向量，通过软注意力机制从边特征中
    聚合出各路径的预测流量。

    Args:
        hidden: 隐藏层维度。
        P:      路径数量，默认为 200。
    """

    def __init__(self, hidden: int, P: int = 200):
        super().__init__()
        self.q = nn.Parameter(torch.randn(P, hidden))   # 可学习路径查询向量 [P, H]
        self.edge2flow = nn.Linear(hidden, 1)           # 边特征→标量流量
        self.smooth = EdgeNodeEdgeBlock(hidden)         # 一次 E→N→E 平滑，鼓励路径连续性

    def forward(
        self,
        z_EH: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算路径级别预测流量和注意力权重。

        Args:
            z_EH:      [E, H]，边特征。
            src:       [E]，边的源节点索引。
            dst:       [E]，边的目标节点索引。
            num_nodes: 图中节点总数 N。

        Returns:
            y:     [P]，各路径预测流量。
            alpha: [P, E]，各路径对各边的注意力权重（已 softmax）。
        """
        z_s = self.smooth(z_EH, src, dst, num_nodes)    # [E, H]，平滑后的边特征
        scores = self.q @ (z_EH + z_s).T                # [P, E]，路径-边相似度
        alpha = torch.softmax(scores, dim=-1)            # [P, E]，归一化注意力
        F_e = self.edge2flow(z_EH).squeeze(-1)           # [E]，每条边的流量估计
        y = (alpha * F_e.unsqueeze(0)).sum(-1)           # [P]，加权聚合为路径流量
        return y, alpha


class PathFormerGNN(nn.Module):
    """PathFormerGNN：基于图神经网络的路径流量预测模型。

    整体架构：
      1. 将稠密矩阵输入转换为稀疏图表示（过滤零值边）
      2. EdgeTimeEncoder：融合节点时序与边属性，生成边的时序表示
      3. EdgeNodeEdgeBlock × depth：多层 E→N→E 消息传递，增强边特征
      4. SoftPathReadoutLite：软注意力路径读出，输出路径级别流量

    Args:
        hidden:          隐藏层维度，默认为 64。
        tcn_layers:      时序编码器中 TCN 层数，默认为 2。
        depth:           E→N→E 残差块层数，默认为 3。
        P:               预测路径数，默认为 200。
        edge_threshold:  边筛选阈值（按特征幅值过滤），默认为 0.0。
        keep_self_loops: 是否保留自环，默认为 False。
        debug_checks:    是否开启索引越界断言检查，默认为 False。
    """

    def __init__(
        self,
        hidden: int = 64,
        tcn_layers: int = 2,
        depth: int = 3,
        P: int = 200,
        edge_threshold: float = 0.0,
        keep_self_loops: bool = False,
        debug_checks: bool = False,
    ):
        super().__init__()
        self.edge_threshold = edge_threshold
        self.keep_self_loops = keep_self_loops
        self.debug_checks = debug_checks
        self.P = P

        self.encoder = EdgeTimeEncoder_NoShift(1, hidden, tcn_layers)
        self.blocks = nn.ModuleList([EdgeNodeEdgeBlock(hidden) for _ in range(depth)])
        self.readout = SoftPathReadoutLite(hidden, P)

    @staticmethod
    def _sanitize_edges(
        src: torch.Tensor, dst: torch.Tensor, N: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """过滤越界索引，保证所有节点编号满足 0 <= id < N。

        如果过滤后无有效边，则返回一条 (0→0) 的自环作为占位符。
        """
        mask_ok = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
        if mask_ok.sum() == 0:
            src = torch.zeros(1, dtype=torch.long, device=src.device)
            dst = torch.zeros(1, dtype=torch.long, device=src.device)
        else:
            src = src[mask_ok]
            dst = dst[mask_ok]
        return src, dst

    def _dense_to_graph_single(
        self, Xb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """将单样本稠密张量转换为稀疏图表示。

        Args:
            Xb: [3, N, N, T]，单样本输入。

        Returns:
            x_seq:         [T, N, 1]，节点时序特征。
            edge_index:    [2, E]，稀疏边索引（long）。
            edge_attr_seq: 长度为 T 的列表，每个元素 [E, 2]，
                           包含各时间步的 [匹配流量, 行程时间]。
        """
        _, N, N2, T = Xb.shape
        assert N == N2, f"N 维度不匹配: {N} vs {N2}"
        device = Xb.device

        edge_feat0 = Xb[0]  # [N, N, T]，匹配流量
        edge_feat1 = Xb[1]  # [N, N, T]，行程时间
        node_plane = Xb[2]  # [N, N, T]，节点流量广播平面

        # 从对角线恢复节点时序
        x_seq = recover_node_sequence_diag(node_plane).to(device)  # [T, N, 1]

        # 按特征幅值筛选有效边
        mag = edge_feat0.abs().sum(dim=-1) + edge_feat1.abs().sum(dim=-1)  # [N, N]
        mask = mag > self.edge_threshold
        if not self.keep_self_loops:
            idx = torch.arange(N, device=device)
            mask[idx, idx] = False

        src, dst = mask.nonzero(as_tuple=True)
        src = src.to(device=device, dtype=torch.long)
        dst = dst.to(device=device, dtype=torch.long)
        src, dst = self._sanitize_edges(src, dst, N)
        edge_index = torch.stack([src, dst], dim=0)  # [2, E]

        # 构建每个时间步的边属性列表
        edge_attr_seq: List[torch.Tensor] = []
        for t in range(T):
            a_t = edge_feat0[src, dst, t].unsqueeze(1)   # [E, 1]
            b_t = edge_feat1[src, dst, t].unsqueeze(1)   # [E, 1]
            edge_attr_seq.append(torch.cat([a_t, b_t], dim=1))  # [E, 2]

        return x_seq, edge_index, edge_attr_seq

    def _forward_single(self, Xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """单样本前向传播。

        Args:
            Xb: [3, N, N, T]，单样本输入。

        Returns:
            y_b:     [P]，路径预测流量。
            alpha_b: [P, E_b]，路径-边注意力权重。
        """
        x_seq, edge_index, edge_attr_seq = self._dense_to_graph_single(Xb)
        src, dst = edge_index[0], edge_index[1]
        N = Xb.size(1)

        if self.debug_checks:
            assert int(src.max()) < N and int(dst.max()) < N, (
                f"节点索引越界: max(src)={int(src.max())}, max(dst)={int(dst.max())}, N={N}"
            )

        # 1. 边时序编码，并沿时间维取均值聚合
        z_ETD = self.encoder(x_seq, edge_index, edge_attr_seq)   # [E, T, H]
        z_EH = z_ETD.mean(dim=1).contiguous()                    # [E, H]

        # 2. 多层 E→N→E 消息传递
        for blk in self.blocks:
            z_EH = blk(z_EH, src, dst, N)

        # 3. 软路径读出
        y_b, alpha_b = self.readout(z_EH, src, dst, N)          # [P], [P, E]
        return y_b, alpha_b

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """批量前向传播。

        由于每个样本的稀疏图结构不同（E_b 各异），目前逐样本串行处理。

        Args:
            X: [B, 3, N, N, T]，批量输入。
               - X[:, 0]：匹配流量（边特征）
               - X[:, 1]：行程时间（边特征）
               - X[:, 2]：节点流量广播平面（对角线存储节点时序）

        Returns:
            y: [B, P]，各样本各路径的预测流量。
        """
        assert X.dim() == 5 and X.size(1) == 3, f"期望输入形状 [B,3,N,N,T]，实际为 {tuple(X.shape)}"
        B = X.size(0)
        y_list = []
        for b in range(B):
            y_b, _ = self._forward_single(X[b])
            y_list.append(y_b.unsqueeze(0))   # [1, P]
        return torch.cat(y_list, dim=0)        # [B, P]


# ====== 最小前向自测 ======
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, T = 2, 16, 8
    X = torch.zeros(B, 3, N, N, T)

    # 随机构造合理数据
    edge_mask = torch.rand(B, N, N) > 0.7
    for b in range(B):
        edge_mask[b].fill_diagonal_(False)
        for t in range(T):
            X[b, 0, :, :, t] = (torch.rand(N, N) * edge_mask[b]).float()  # 匹配流量
            X[b, 1, :, :, t] = 0.1 + 0.2 * torch.rand(N, N)               # 行程时间
        node_series = torch.rand(N, T)
        for i in range(N):
            X[b, 2, i, i, :] = node_series[i]                              # 节点对角线广播

    model = PathFormerGNN(
        hidden=64, tcn_layers=2, depth=3, P=200,
        edge_threshold=0.0, keep_self_loops=False, debug_checks=True,
    )
    print(model)
    y = model(X)
    print("y:", y.shape)  # 期望 [2, 200]
