# pathFormerGNN_batch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from typing import List, Tuple


# ---------- 从广播后的节点平面恢复节点时序（固定 diag） ----------
def recover_node_sequence_diag(node_plane_NNT: torch.Tensor) -> torch.Tensor:
    """
    node_plane_NNT: [N, N, T]，节点 1 维特征广播到 N×N 的平面
    使用对角线 (i,i,:) 还原每个节点 i 在各时间步的取值
    return: x_seq [T, N, 1]
    """
    x_NT = node_plane_NNT.diagonal(dim1=0, dim2=1).contiguous()  # [T,N]
    return x_NT.unsqueeze(-1)  # [T,N,1]


# ---------- 边时序编码（无 time_shift）：节点→边门控 + TCN ----------
class EdgeTimeEncoder_NoShift(nn.Module):
    def __init__(self, node_in: int = 1, hidden: int = 64, tcn_layers: int = 2):
        super().__init__()
        self.node_proj = nn.Linear(node_in, hidden)
        self.edge_mlp  = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )  # 边特征: [匹配流量, 行程时间]
        convs = []
        for b in range(tcn_layers):
            dil = 2 ** b
            convs += [nn.Conv1d(hidden, hidden, kernel_size=3, padding=dil, dilation=dil), nn.ReLU()]
        self.tcn = nn.Sequential(*convs)

    def forward(self, x_seq_TN1: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr_seq: List[torch.Tensor]) -> torch.Tensor:
        """
        x_seq_TN1: [T,N,1]
        edge_index: [2,E] (long)
        edge_attr_seq: 长度 T 的 list，每个 [E,2]
        return: z_ETD: [E,T,hidden]
        """
        T, N, _ = x_seq_TN1.shape
        H_TNH = F.relu(self.node_proj(x_seq_TN1))              # [T,N,H]
        src = edge_index[0]                                    # [E] long

        # 只在节点维 (dim=1) 索引，防越界误判
        h_src = H_TNH.index_select(dim=1, index=src).transpose(0, 1).contiguous()  # [E,T,H]

        ea_ET2 = torch.stack(edge_attr_seq, dim=1).contiguous()   # [E,T,2]
        gate   = F.relu(self.edge_mlp(ea_ET2.view(-1, 2))).view(ea_ET2.size(0), T, -1)  # [E,T,H]

        z = gate * h_src                                      # [E,T,H]
        z = z.permute(0, 2, 1).contiguous()                   # [E,H,T]
        z = self.tcn(z)
        z = z.permute(0, 2, 1).contiguous()                   # [E,T,H]
        return z


# ---------- 原图上的 E->N->E 残差块（轻量“边到边”卷积） ----------
class EdgeNodeEdgeBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.lin_e_in  = nn.Linear(hidden, hidden, bias=False)
        self.lin_e_out = nn.Linear(hidden, hidden, bias=False)
        self.update = nn.Sequential(
            nn.Linear(3*hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, z_EH: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        # E->N 聚合（入/出）
        m_in  = self.lin_e_in(z_EH)                        # [E,H]
        m_out = self.lin_e_out(z_EH)                       # [E,H]
        h_in  = scatter_add(m_in,  dst, dim=0, dim_size=num_nodes)   # [N,H]
        h_out = scatter_add(m_out, src, dim=0, dim_size=num_nodes)   # [N,H]
        # 度归一化
        deg_in  = torch.bincount(dst, minlength=num_nodes).clamp(min=1).to(z_EH.dtype).unsqueeze(-1)
        deg_out = torch.bincount(src, minlength=num_nodes).clamp(min=1).to(z_EH.dtype).unsqueeze(-1)
        h_in  = h_in  / deg_in
        h_out = h_out / deg_out
        # N->E 回写
        z_msg = torch.cat([z_EH, h_out.index_select(0, src), h_in.index_select(0, dst)], dim=-1)  # [E,3H]
        z_new = self.update(z_msg)
        return self.norm(z_EH + z_new)


# ---------- 软路径读出（不建线图，E->N->E 平滑鼓励连续性） ----------
class SoftPathReadoutLite(nn.Module):
    def __init__(self, hidden: int, P: int = 200):
        super().__init__()
        self.q = nn.Parameter(torch.randn(P, hidden))
        self.edge2flow = nn.Linear(hidden, 1)
        self.smooth = EdgeNodeEdgeBlock(hidden)  # 一次 E->N->E 平滑

    def forward(self, z_EH: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        z_s = self.smooth(z_EH, src, dst, num_nodes)     # [E,H]
        scores = self.q @ (z_EH + z_s).T                 # [P,E]
        alpha  = torch.softmax(scores, dim=-1)           # [P,E]
        F_e    = self.edge2flow(z_EH).squeeze(-1)        # [E]
        y      = (alpha * F_e.unsqueeze(0)).sum(-1)      # [P]
        return y, alpha


# ---------- 主模型（支持 Batch） ----------
class PathFormerGNN(nn.Module):
    """
    输入 X: [B, 3, N, N, T]
      X[:,0]: 匹配流量 (edge feature)
      X[:,1]: 行程时间 (edge feature) —— Δt >> τ，仅作特征门控
      X[:,2]: 节点流量广播平面 (node feature broadcast 到 N×N)，用对角线还原节点序列
    输出:
      y:     [B, 200]
      alpha: 长度为 B 的列表；alpha[b] 形状 [200, E_b]
    """
    def __init__(self, hidden: int = 64, tcn_layers: int = 2, depth: int = 3,
                 P: int = 200, edge_threshold: float = 0.0, keep_self_loops: bool = False,
                 debug_checks: bool = False):
        super().__init__()
        self.edge_threshold = edge_threshold
        self.keep_self_loops = keep_self_loops
        self.debug_checks = debug_checks

        self.encoder = EdgeTimeEncoder_NoShift(1, hidden, tcn_layers)
        self.blocks  = nn.ModuleList([EdgeNodeEdgeBlock(hidden) for _ in range(depth)])
        self.readout = SoftPathReadoutLite(hidden, P)
        self.P = P

    @staticmethod
    def _sanitize_edges(src: torch.Tensor, dst: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 过滤越界索引，保证 0 <= id < N """
        mask_ok = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
        if mask_ok.sum() == 0:
            src = torch.zeros(1, dtype=torch.long, device=src.device)
            dst = torch.zeros(1, dtype=torch.long, device=src.device)
        else:
            src = src[mask_ok]
            dst = dst[mask_ok]
        return src, dst

    def _dense_to_graph_single(self, Xb: torch.Tensor):
        """
        单样本：Xb [3,N,N,T] -> x_seq:[T,N,1], edge_index:[2,E], edge_attr_seq: list[T] of [E,2]
        """
        _, N, N2, T = Xb.shape
        assert N == N2, f"N mismatch: {N} vs {N2}"
        device = Xb.device

        edge_feat0 = Xb[0]  # [N,N,T] 匹配流量
        edge_feat1 = Xb[1]  # [N,N,T] 行程时间
        node_plane = Xb[2]  # [N,N,T] 节点流量广播

        # 还原节点时序
        x_seq = recover_node_sequence_diag(node_plane).to(device)  # [T,N,1]

        # 选边
        mag = edge_feat0.abs().sum(dim=-1) + edge_feat1.abs().sum(dim=-1)  # [N,N]
        mask = mag > self.edge_threshold
        if not self.keep_self_loops:
            idx = torch.arange(N, device=device)
            mask[idx, idx] = False

        src, dst = mask.nonzero(as_tuple=True)
        src = src.to(device=device, dtype=torch.long)
        dst = dst.to(device=device, dtype=torch.long)
        src, dst = self._sanitize_edges(src, dst, N)
        edge_index = torch.stack([src, dst], dim=0)  # [2,E]

        # 每个时间步的边特征 [E,2]
        edge_attr_seq: List[torch.Tensor] = []
        for t in range(T):
            a_t = edge_feat0[src, dst, t].unsqueeze(1)  # [E,1]
            b_t = edge_feat1[src, dst, t].unsqueeze(1)  # [E,1]
            edge_attr_seq.append(torch.cat([a_t, b_t], dim=1))  # [E,2]

        return x_seq, edge_index, edge_attr_seq  # [T,N,1], [2,E], list[T] of [E,2]

    def _forward_single(self, Xb: torch.Tensor):
        """
        单样本前向，返回 (y_b:[P], alpha_b:[P,E_b])
        输入数据归一化：对匹配流量、行程时间、节点流量分别做归一化
        """
        x_seq, edge_index, edge_attr_seq = self._dense_to_graph_single(Xb)
        src, dst = edge_index[0], edge_index[1]
        N = Xb.size(1)

        if self.debug_checks:
            assert int(src.max()) < N and int(dst.max()) < N, \
                f"Index out of range: max(src)={int(src.max())}, max(dst)={int(dst.max())}, N={N}"

        # 边时序编码 & 时间聚合
        z_ETD = self.encoder(x_seq, edge_index, edge_attr_seq)  # [E,T,H]
        z_EH  = z_ETD.mean(dim=1).contiguous()                  # [E,H]

        # 原图上若干层 E->N->E
        for blk in self.blocks:
            z_EH = blk(z_EH, src, dst, N)

        # 软路径读出
        y_b, alpha_b = self.readout(z_EH, src, dst, N)         # [P], [P,E]
        return y_b, alpha_b

    def forward(self, X: torch.Tensor):
        """
        X: [B, 3, N, N, T]
        return:
          y: [B, P]
          alpha_list: List[ torch.Tensor [P, E_b] ]  （每个样本的注意力）
        """
        assert X.dim() == 5 and X.size(1) == 3, f"Expect [B,3,N,N,T], got {tuple(X.shape)}"
        B = X.size(0)
        y_list = []
        # alpha_list = []
        for b in range(B):
            y_b, alpha_b = self._forward_single(X[b])
            y_list.append(y_b.unsqueeze(0))   # [1,P]
            # alpha_list.append(alpha_b)        # [P,E_b]
        y = torch.cat(y_list, dim=0)          # [B,P]
        return y


# ====== 最小前向自测 ======
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

    model = PathFormerGNN(hidden=64, tcn_layers=2, depth=3, P=200,
                          edge_threshold=0.0, keep_self_loops=False, debug_checks=True)
    print(model)
    y, alpha_list = model(X)
    print("y:", y.shape)  # [B,200]
    print("alpha[0]:", alpha_list[0].shape)  # [200, E_0]
