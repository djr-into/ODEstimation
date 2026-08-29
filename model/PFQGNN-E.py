# PFQGNN-E.py
# Joint 2-hour path-flow estimation + 2-hour forecasting
# Minimal-change extension of pathFormerGNN.py:
#   1) temporal mean pooling -> last temporal state
#   2) one Path Query head -> estimation / prediction dual heads
#   3) hour-of-day context -> FiLM-conditioned Path Query
#
# Input:
#   X:         [B, 3, N, N, T]
#   hour_est:  [B]  target-window midpoint hour for estimation, in [0, 24)
#   hour_pred: [B]  target-window midpoint hour for prediction, in [0, 24)
#
# Output:
#   y_est:  [B, P]  estimated path flow for current 2-hour window
#   y_pred: [B, P]  forecast path flow for future 2-hour window

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from typing import List, Tuple


# ---------- 从广播后的节点平面恢复节点时序（固定 diag） ----------
def recover_node_sequence_diag(node_plane_NNT: torch.Tensor) -> torch.Tensor:
    """
    node_plane_NNT: [N, N, T]
    使用对角线 (i,i,:) 还原每个节点 i 在各时间步的取值。

    return:
        x_seq: [T, N, 1]
    """
    x_NT = node_plane_NNT.diagonal(dim1=0, dim2=1).contiguous()  # [T,N]
    return x_NT.unsqueeze(-1)  # [T,N,1]


# ---------- 边时序编码（保持原模型不变） ----------
class EdgeTimeEncoder_NoShift(nn.Module):
    def __init__(self, node_in: int = 1, hidden: int = 64, tcn_layers: int = 2):
        super().__init__()

        self.node_proj = nn.Linear(node_in, hidden)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )  # 边特征: [匹配流量, 行程时间]

        convs = []
        for b in range(tcn_layers):
            dil = 2 ** b
            convs += [
                nn.Conv1d(
                    hidden,
                    hidden,
                    kernel_size=3,
                    padding=dil,
                    dilation=dil,
                ),
                nn.ReLU(),
            ]
        self.tcn = nn.Sequential(*convs)

    def forward(
        self,
        x_seq_TN1: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_seq: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        x_seq_TN1:    [T,N,1]
        edge_index:   [2,E]
        edge_attr_seq: length-T list, each [E,2]

        return:
            z_ETD: [E,T,hidden]
        """
        T, _, _ = x_seq_TN1.shape

        H_TNH = F.relu(self.node_proj(x_seq_TN1))  # [T,N,H]
        src = edge_index[0]                         # [E]

        h_src = (
            H_TNH.index_select(dim=1, index=src)
            .transpose(0, 1)
            .contiguous()
        )  # [E,T,H]

        ea_ET2 = torch.stack(edge_attr_seq, dim=1).contiguous()  # [E,T,2]
        gate = F.relu(self.edge_mlp(ea_ET2.view(-1, 2))).view(
            ea_ET2.size(0), T, -1
        )  # [E,T,H]

        z = gate * h_src                 # [E,T,H]
        z = z.permute(0, 2, 1).contiguous()  # [E,H,T]
        z = self.tcn(z)
        z = z.permute(0, 2, 1).contiguous()  # [E,T,H]

        return z


# ---------- 原图上的 E->N->E 残差块（保持原模型不变） ----------
class EdgeNodeEdgeBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()

        self.lin_e_in = nn.Linear(hidden, hidden, bias=False)
        self.lin_e_out = nn.Linear(hidden, hidden, bias=False)

        self.update = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
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
        # E -> N 聚合（入 / 出）
        m_in = self.lin_e_in(z_EH)    # [E,H]
        m_out = self.lin_e_out(z_EH)  # [E,H]

        h_in = scatter_add(
            m_in, dst, dim=0, dim_size=num_nodes
        )  # [N,H]

        h_out = scatter_add(
            m_out, src, dim=0, dim_size=num_nodes
        )  # [N,H]

        # 度归一化
        deg_in = (
            torch.bincount(dst, minlength=num_nodes)
            .clamp(min=1)
            .to(z_EH.dtype)
            .unsqueeze(-1)
        )

        deg_out = (
            torch.bincount(src, minlength=num_nodes)
            .clamp(min=1)
            .to(z_EH.dtype)
            .unsqueeze(-1)
        )

        h_in = h_in / deg_in
        h_out = h_out / deg_out

        # N -> E 回写
        z_msg = torch.cat(
            [
                z_EH,
                h_out.index_select(0, src),
                h_in.index_select(0, dst),
            ],
            dim=-1,
        )  # [E,3H]

        z_new = self.update(z_msg)

        return self.norm(z_EH + z_new)


# ---------- Hour-of-day 条件化的软路径读出 ----------
class SoftPathReadoutLiteHour(nn.Module):
    """
    在原 SoftPathReadoutLite 基础上做最小修改：

        base path query q_p
                +
        hour-of-day context FiLM
                ↓
        context-conditioned path query

    hour_ctx 使用周期编码:
        [sin(2*pi*h/24), cos(2*pi*h/24)]
    """

    def __init__(self, hidden: int, P: int = 200):
        super().__init__()

        # 每条路径的基础 query
        self.q = nn.Parameter(torch.randn(P, hidden))

        # 2维 hour context -> gamma / beta，各 hidden 维
        self.hour_film = nn.Linear(2, 2 * hidden)

        # 保持原 readout 结构
        self.edge2flow = nn.Linear(hidden, 1)
        self.smooth = EdgeNodeEdgeBlock(hidden)

    def forward(
        self,
        z_EH: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        hour_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_EH:     [E,H]
        hour_ctx: [2]

        return:
            y:     [P]
            alpha: [P,E]
        """
        z_s = self.smooth(z_EH, src, dst, num_nodes)  # [E,H]

        # Hour-conditioned Path Query
        gamma, beta = self.hour_film(hour_ctx).chunk(2, dim=-1)  # [H], [H]

        # tanh 限制缩放幅度，初始化阶段更稳定
        q = self.q * (1.0 + torch.tanh(gamma).unsqueeze(0))
        q = q + beta.unsqueeze(0)  # [P,H]

        scores = q @ (z_EH + z_s).T       # [P,E]
        alpha = torch.softmax(scores, dim=-1)

        F_e = self.edge2flow(z_EH).squeeze(-1)  # [E]
        y = (alpha * F_e.unsqueeze(0)).sum(-1)  # [P]

        return y, alpha


# ---------- PFQGNN-E 主模型 ----------
class PFQGNNE(nn.Module):
    """
    PFQGNN-E: Joint Path-Flow Estimation + Forecasting

    与原 PathFormerGNN 相比只做三处核心修改：
      1. TCN 后使用最后时刻状态，而不是时间 mean；
      2. 一个 Path Query head 拆成 estimation / prediction 两个 head；
      3. 两个 head 分别接受其目标 2h window 中点对应的 hour-of-day context。

    输入:
        X:         [B,3,N,N,T]
        hour_est:  [B]，估计目标2h窗口的中点小时，例如 [03,05) -> 4
        hour_pred: [B]，预测目标2h窗口的中点小时，例如 [05,07) -> 6

    输出:
        y_est:  [B,P]
        y_pred: [B,P]
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

        # Backbone 完全沿用原模型
        self.encoder = EdgeTimeEncoder_NoShift(
            node_in=1,
            hidden=hidden,
            tcn_layers=tcn_layers,
        )

        self.blocks = nn.ModuleList(
            [EdgeNodeEdgeBlock(hidden) for _ in range(depth)]
        )

        # 双 Path Query heads
        self.readout_est = SoftPathReadoutLiteHour(hidden, P)
        self.readout_pred = SoftPathReadoutLiteHour(hidden, P)

        self.P = P

    @staticmethod
    def encode_hour(hour: torch.Tensor) -> torch.Tensor:
        """
        将小时编码为周期特征。

        hour:
            scalar tensor, 可取 0~24 之间的实数；
            因此后续如果想用 4.5h 这样的连续中点也可以。

        return:
            [2] = [sin, cos]
        """
        hour = hour.to(dtype=torch.float32)
        angle = 2.0 * torch.pi * hour / 24.0

        return torch.stack(
            [
                torch.sin(angle),
                torch.cos(angle),
            ],
            dim=-1,
        )

    @staticmethod
    def _sanitize_edges(
        src: torch.Tensor,
        dst: torch.Tensor,
        N: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        过滤越界索引，保证 0 <= id < N。
        """
        mask_ok = (
            (src >= 0)
            & (src < N)
            & (dst >= 0)
            & (dst < N)
        )

        if mask_ok.sum() == 0:
            src = torch.zeros(
                1,
                dtype=torch.long,
                device=src.device,
            )
            dst = torch.zeros(
                1,
                dtype=torch.long,
                device=src.device,
            )
        else:
            src = src[mask_ok]
            dst = dst[mask_ok]

        return src, dst

    def _dense_to_graph_single(self, Xb: torch.Tensor):
        """
        单样本:
            Xb [3,N,N,T]
                ->
            x_seq        [T,N,1]
            edge_index   [2,E]
            edge_attr_seq length-T list of [E,2]
        """
        _, N, N2, T = Xb.shape
        assert N == N2, f"N mismatch: {N} vs {N2}"

        device = Xb.device

        edge_feat0 = Xb[0]  # [N,N,T] 匹配流量
        edge_feat1 = Xb[1]  # [N,N,T] 行程时间
        node_plane = Xb[2]  # [N,N,T] 节点流量广播

        # 还原节点时序
        x_seq = recover_node_sequence_diag(node_plane).to(device)  # [T,N,1]

        # 保持原模型动态选边逻辑
        mag = (
            edge_feat0.abs().sum(dim=-1)
            + edge_feat1.abs().sum(dim=-1)
        )  # [N,N]

        mask = mag > self.edge_threshold

        if not self.keep_self_loops:
            idx = torch.arange(N, device=device)
            mask[idx, idx] = False

        src, dst = mask.nonzero(as_tuple=True)

        src = src.to(
            device=device,
            dtype=torch.long,
        )
        dst = dst.to(
            device=device,
            dtype=torch.long,
        )

        src, dst = self._sanitize_edges(
            src,
            dst,
            N,
        )

        edge_index = torch.stack(
            [src, dst],
            dim=0,
        )  # [2,E]

        # 每个时间步的边特征 [E,2]
        edge_attr_seq: List[torch.Tensor] = []

        for t in range(T):
            a_t = edge_feat0[
                src, dst, t
            ].unsqueeze(1)  # [E,1]

            b_t = edge_feat1[
                src, dst, t
            ].unsqueeze(1)  # [E,1]

            edge_attr_seq.append(
                torch.cat(
                    [a_t, b_t],
                    dim=1,
                )
            )  # [E,2]

        return (
            x_seq,
            edge_index,
            edge_attr_seq,
        )

    def _forward_single(
        self,
        Xb: torch.Tensor,
        hour_est: torch.Tensor,
        hour_pred: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        单样本前向。

        return:
            y_est:      [P]
            y_pred:     [P]
            alpha_est:  [P,E]
            alpha_pred: [P,E]
        """
        x_seq, edge_index, edge_attr_seq = self._dense_to_graph_single(Xb)

        src = edge_index[0]
        dst = edge_index[1]
        N = Xb.size(1)

        if self.debug_checks:
            assert int(src.max()) < N and int(dst.max()) < N, (
                f"Index out of range: "
                f"max(src)={int(src.max())}, "
                f"max(dst)={int(dst.max())}, "
                f"N={N}"
            )

        # ---------- Temporal backbone ----------
        z_ETD = self.encoder(
            x_seq,
            edge_index,
            edge_attr_seq,
        )  # [E,T,H]

        # 修改 1:
        # 原模型: z_ETD.mean(dim=1)
        # PFQGNN-E: 保留最后时间状态，强化预测时序方向
        z_EH = z_ETD[:, -1, :].contiguous()  # [E,H]

        # ---------- Spatial backbone ----------
        # 完全保持原 E->N->E 结构
        for blk in self.blocks:
            z_EH = blk(
                z_EH,
                src,
                dst,
                N,
            )

        # ---------- Global hour-of-day context ----------
        ctx_est = self.encode_hour(hour_est).to(
            device=z_EH.device,
            dtype=z_EH.dtype,
        )  # [2]

        ctx_pred = self.encode_hour(hour_pred).to(
            device=z_EH.device,
            dtype=z_EH.dtype,
        )  # [2]

        # 修改 2 + 3:
        # estimation / prediction 分别使用独立 Path Query，
        # 并由各自目标窗口 hour context 条件化。
        y_est, alpha_est = self.readout_est(
            z_EH,
            src,
            dst,
            N,
            ctx_est,
        )

        y_pred, alpha_pred = self.readout_pred(
            z_EH,
            src,
            dst,
            N,
            ctx_pred,
        )

        return (
            y_est,
            y_pred,
            alpha_est,
            alpha_pred,
        )

    def forward(
        self,
        X: torch.Tensor,
        hour_est: torch.Tensor,
        hour_pred: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        X:
            [B,3,N,N,T]

        hour_est:
            [B]

        hour_pred:
            [B]

        return:
            默认:
                y_est, y_pred
                shapes: [B,P], [B,P]

            return_attention=True:
                y_est, y_pred, alpha_est_list, alpha_pred_list
        """
        assert (
            X.dim() == 5 and X.size(1) == 3
        ), f"Expect [B,3,N,N,T], got {tuple(X.shape)}"

        B = X.size(0)

        assert hour_est.numel() == B, (
            f"hour_est should contain B={B} values, "
            f"got shape={tuple(hour_est.shape)}"
        )

        assert hour_pred.numel() == B, (
            f"hour_pred should contain B={B} values, "
            f"got shape={tuple(hour_pred.shape)}"
        )

        hour_est = hour_est.reshape(-1)
        hour_pred = hour_pred.reshape(-1)

        y_est_list = []
        y_pred_list = []

        alpha_est_list = []
        alpha_pred_list = []

        for b in range(B):
            (
                y_est_b,
                y_pred_b,
                alpha_est_b,
                alpha_pred_b,
            ) = self._forward_single(
                X[b],
                hour_est[b],
                hour_pred[b],
            )

            y_est_list.append(
                y_est_b.unsqueeze(0)
            )

            y_pred_list.append(
                y_pred_b.unsqueeze(0)
            )

            if return_attention:
                alpha_est_list.append(alpha_est_b)
                alpha_pred_list.append(alpha_pred_b)

        y_est = torch.cat(
            y_est_list,
            dim=0,
        )  # [B,P]

        y_pred = torch.cat(
            y_pred_list,
            dim=0,
        )  # [B,P]

        if return_attention:
            return (
                y_est,
                y_pred,
                alpha_est_list,
                alpha_pred_list,
            )

        return y_est, y_pred


# ---------- 最小前向自测 ----------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, N, T = 2, 16, 4
    X = torch.zeros(B, 3, N, N, T)

    # 随机构造合理数据
    edge_mask = torch.rand(B, N, N) > 0.7

    for b in range(B):
        edge_mask[b].fill_diagonal_(False)

        for t in range(T):
            X[b, 0, :, :, t] = (
                torch.rand(N, N) * edge_mask[b]
            ).float()

            X[b, 1, :, :, t] = (
                (0.1 + 0.2 * torch.rand(N, N))
                * edge_mask[b]
            ).float()

        node_series = torch.rand(N, T)

        for i in range(N):
            X[b, 2, i, i, :] = node_series[i]

    # 例：
    # estimation target = [03,05) -> midpoint 04:00
    # prediction target = [05,07) -> midpoint 06:00
    hour_est = torch.tensor([4.0, 8.0])
    hour_pred = torch.tensor([6.0, 10.0])

    model = PFQGNNE(
        hidden=64,
        tcn_layers=2,
        depth=3,
        P=200,
        edge_threshold=0.0,
        keep_self_loops=False,
        debug_checks=True,
    )

    print(model)

    y_est, y_pred = model(
        X,
        hour_est,
        hour_pred,
    )

    print("y_est:", y_est.shape)    # [B,200]
    print("y_pred:", y_pred.shape)  # [B,200]

    (
        y_est,
        y_pred,
        alpha_est_list,
        alpha_pred_list,
    ) = model(
        X,
        hour_est,
        hour_pred,
        return_attention=True,
    )

    print(
        "alpha_est[0]:",
        alpha_est_list[0].shape,
    )  # [200,E_0]

    print(
        "alpha_pred[0]:",
        alpha_pred_list[0].shape,
    )  # [200,E_0]
