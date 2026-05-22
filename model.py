"""
模块结构：
1. IICE-Conv: 瞬时交互因果增强卷积 (微观因果发现)
2. GroupInteractionAware (GIA): 群体交互感知 (宏观因果发现)
3. GroupFieldGenerator: 群体场生成模块
4. ExplicitSocialForce: 显式社交力建模 (双向物理链路)
5. ImplicitSocialForce: 隐式社交力建模 (物理势场)
6. CausalConstraintEnhancement: 因果约束增强与特征融合
7. CaDSLTraj: 主模型
"""

import torch
import torch.nn as nn
from laplace_decoder import MLPDecoder

class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class GaussianNLLLoss(nn.Module):
    """https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    """
    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = 0.5*(torch.log(scale**2) + torch.abs(target - loc)**2 / scale**2)
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):

    N, C, IH, IW = input.shape
    _, H, W, _ = grid.shape

    # 1. 将 [-1, 1] 坐标映射到 [0, IW-1] 和 [0, IH-1]
    ix = grid[..., 0]
    iy = grid[..., 1]

    if align_corners:
        ix = ((ix + 1) / 2) * (IW - 1)
        iy = ((iy + 1) / 2) * (IH - 1)
    else:
        ix = ((ix + 1) * IW - 1) / 2
        iy = ((iy + 1) * IH - 1) / 2

    # 2. 获取四个邻近像素的坐标
    ix0 = torch.floor(ix).long()
    iy0 = torch.floor(iy).long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # 3. 计算权重 (这些权重计算是可导的)
    dx = ix - ix0.float()
    dy = iy - iy0.float()

    # 4. 边界处理与掩码 (针对 padding_mode='zeros')
    mask = (ix >= 0) & (ix <= IW - 1) & (iy >= 0) & (iy <= IH - 1)
    mask = mask.float().unsqueeze(1)  # [N, 1, H, W]

    ix0 = torch.clamp(ix0, 0, IW - 1)
    ix1 = torch.clamp(ix1, 0, IW - 1)
    iy0 = torch.clamp(iy0, 0, IH - 1)
    iy1 = torch.clamp(iy1, 0, IH - 1)

    # 5. 安全地获取像素值 (利用 reshape 和 gather)
    # 将 input 展平为 [N, C, IH*IW]
    input_flat = input.view(N, C, -1)

    def get_val(x, y):
        idx = y * IW + x  # [N, H, W]
        idx = idx.unsqueeze(1).expand(-1, C, -1, -1).reshape(N, C, -1)
        return torch.gather(input_flat, 2, idx).reshape(N, C, H, W)

    v00 = get_val(ix0, iy0)
    v10 = get_val(ix1, iy0)
    v01 = get_val(ix0, iy1)
    v11 = get_val(ix1, iy1)

    # 6. 双线性插值融合
    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy)
    wc = (1 - dx) * dy
    wd = dx * dy

    # 增加维度以对齐通道 C
    wa, wb, wc, wd = wa.unsqueeze(1), wb.unsqueeze(1), wc.unsqueeze(1), wd.unsqueeze(1)
    out = wa * v00 + wb * v10 + wc * v01 + wd * v11
    # 应用 'zeros' 填充模式
    return out * mask

# ─────────────────────────────────────────────────────────────
# 1. 瞬时交互因果增强卷积 (IICE-Conv)
# ─────────────────────────────────────────────────────────────
class IICEConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, sigma=0.5):
        super(IICEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # 1. 衰减底数 λ: (0, 1)
        self.lambda_raw = nn.Parameter(torch.tensor(0.0))

        # 2. 状态系数 α: smooth 在 [1, 2], burst > smooth 且总和 <= 2.0
        self.alpha_smooth_raw = nn.Parameter(torch.tensor(-1.0))
        self.alpha_diff_raw = nn.Parameter(torch.tensor(0.0))

        # 3. 状态区分阈值 θ: 让网络自适应学习最佳边界 (保证为正数)
        self.state_threshold_raw = nn.Parameter(torch.tensor(0.5))

        # 后处理层
        self.activation = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(out_channels)
        self.proj = nn.Linear(out_channels, out_channels)

    def causal_indicator(self, delta):
        return (delta <= 0).float()

    def temporal_weight(self, delta):
        return torch.exp(-delta.float() ** 2 / (2 * self.sigma ** 2))

    def forward(self, x):
        N, T, C = x.shape
        device = x.device

        # --- 获取受到严格数学约束的超参数 ---
        lambda_base = torch.sigmoid(self.lambda_raw)
        alpha_smooth = 1.0 + torch.sigmoid(self.alpha_smooth_raw)
        alpha_burst = alpha_smooth + torch.sigmoid(self.alpha_diff_raw) * (2.0 - alpha_smooth)
        # 保证阈值 θ_s 为正
        state_threshold = F.softplus(self.state_threshold_raw)

        deltas = torch.arange(-(self.kernel_size - 1), 1, device=device).float()
        distances = torch.abs(deltas)

        causal_mask = self.causal_indicator(deltas)
        time_weights = self.temporal_weight(deltas)
        combined_mask = causal_mask * time_weights

        normalized_weight = torch.sigmoid(self.weight)
        masked_weight = normalized_weight * combined_mask.view(1, 1, -1)
        # [N, T, C] -> [N, C, T]，并行提取所有时间窗 [N, C, T, K]
        x_nct = x.permute(0, 2, 1)
        x_padded = F.pad(x_nct, (self.kernel_size - 1, 0))
        x_windows = x_padded.unfold(dimension=2, size=self.kernel_size, step=1)

        # 计算所有时间步 alpha_t 与 gamma_t
        alpha_t = torch.ones(N, T, 1, device=device, dtype=x.dtype) * alpha_smooth
        gamma_t = torch.ones(N, T, 1, device=device, dtype=x.dtype)
        if T > 1:
            v_cur, v_prev = x[:, 1:, 2:3], x[:, :-1, 2:3]
            dir_cur, dir_prev = x[:, 1:, 3:4], x[:, :-1, 3:4]
            interaction_state = torch.abs(v_cur - v_prev) + torch.abs(dir_cur - dir_prev)
            is_burst = (interaction_state > state_threshold).float()
            alpha_t[:, 1:, :] = is_burst * alpha_burst + (1 - is_burst) * alpha_smooth

            cos_sim = F.cosine_similarity(x[:, 1:, :], x[:, :-1, :], dim=-1, eps=1e-8).unsqueeze(-1)
            gamma_t[:, 1:, :] = (cos_sim + 1.0) / 2.0

        # 并行构建衰减项 λ^(α_t * |τ|)，并作用于所有时间窗
        exponent = alpha_t * distances.view(1, 1, -1)
        decay_mask = torch.pow(lambda_base, exponent)
        x_windows_decayed = x_windows * decay_mask.unsqueeze(1)

        # 卷积：Y_raw [N, T, out_C]
        Y_raw = torch.einsum('nctk,ock->nto', x_windows_decayed, masked_weight) + self.bias.view(1, 1, -1)
        Y = gamma_t * Y_raw

        # 后处理映射
        h = self.activation(Y)
        h = self.norm(h)
        h = self.proj(h)
        return h  # [N, T, out_C]


# ─────────────────────────────────────────────────────────────
# 2. 群体交互感知器 (GIA) - 宏观因果发现
# ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupInteractionAware(nn.Module):
    def __init__(self, micro_feat_dim, group_stat_dim, hidden_dim, num_heads=4,
                 max_group_size=10, min_group_size=3, pos_fe_dim=5):
        super(GroupInteractionAware, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_group_size = max_group_size
        self.min_group_size = min_group_size

        obs_feat_dim = micro_feat_dim + group_stat_dim + pos_fe_dim
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_feat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim)
        )

        self.mlp_ada = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 4)  # [γ0, γ1, λ, δ]
        )

        self.T_grid = 8
        self.temporal_offset_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, self.T_grid)
        )

        self.D_grid_max = 32
        self.spatial_offset_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, self.D_grid_max)
        )

        self.gtf_attn = nn.MultiheadAttention(hidden_dim // 2, num_heads, batch_first=True)
        self.gat_head_dim = hidden_dim // num_heads

        self.gat_w_weight = nn.Parameter(torch.empty(num_heads, self.gat_head_dim, hidden_dim))
        self.gat_w_bias = nn.Parameter(torch.zeros(num_heads, self.gat_head_dim))

        self.gat_attn_l = nn.Parameter(torch.empty(num_heads, self.gat_head_dim))
        self.gat_attn_r = nn.Parameter(torch.empty(num_heads, self.gat_head_dim))
        self.gat_attn_bias = nn.Parameter(torch.zeros(num_heads))
        nn.init.xavier_uniform_(self.gat_w_weight)
        nn.init.normal_(self.gat_attn_l, mean=0.0, std=0.02)
        nn.init.normal_(self.gat_attn_r, mean=0.0, std=0.02)

        self.layer_norm_lif = nn.LayerNorm(1)
        self.lif_scale = nn.Linear(1, 1)
        self.lif_shift = nn.Parameter(torch.zeros(1))

        self.gat_out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.group_pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim)
        )

    def compute_group_stats_seq(self, pos_seq, vel_seq, adj_seq):
        N, T, _ = pos_seq.shape
        # 1. 密度[T, N] -> [N, T, 1]
        density = adj_seq.sum(dim=-1).transpose(0, 1).unsqueeze(-1) / (N + 1e-8)

        # 2. 平均速度
        group_vel_sum = torch.bmm(adj_seq, vel_seq.transpose(0, 1))
        neighbor_counts = adj_seq.sum(dim=-1, keepdim=True) + 1e-8
        V_i = group_vel_sum / neighbor_counts  # [T, N, 2]
        V_i = V_i.transpose(0, 1)  # [N, T, 2]
        avg_speed = torch.norm(V_i, dim=-1, keepdim=True)  # [N, T, 1]

        # 3. 运动趋势
        V_i_T = V_i.transpose(0, 1)  # [T, N, 2]
        v_j_T = vel_seq.transpose(0, 1)  # [T, N, 2]
        # 计算余弦相似度并映射到 [0,1] -> [T, N, N]
        sim_matrix = F.cosine_similarity(V_i_T.unsqueeze(2), v_j_T.unsqueeze(1), dim=-1, eps=1e-8)
        sim_matrix = (sim_matrix + 1.0) / 2.0

        masked_sim = sim_matrix * adj_seq  # [T, N, N]
        dir_consist = masked_sim.sum(dim=-1).unsqueeze(-1) / neighbor_counts.squeeze(-1).unsqueeze(-1)
        dir_consist = dir_consist.transpose(0, 1)  # [N, T, 1]

        return torch.cat([density, avg_speed, dir_consist], dim=-1)  # [N, T, 3]

    def _uniform_grid_1d(self, size, device):
        return torch.linspace(-1.0, 1.0, size, device=device).unsqueeze(0)

    def _grid_sample_1d(self, feature_seq, adjusted_grid):
        feat_4d = feature_seq.unsqueeze(2)
        grid_x = adjusted_grid.unsqueeze(1).unsqueeze(-1)
        grid_y = torch.zeros_like(grid_x)
        grid = torch.cat([grid_x, grid_y], dim=-1)
        sampled = grid_sample(feat_4d, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return sampled.squeeze(2)

    def compute_gtf(self, obs_embed):
        N, T, H = obs_embed.shape
        device = obs_embed.device
        h_half = H // 2

        # 提取全局池化后的自适应参数
        ada_params = self.mlp_ada(obs_embed.mean(dim=1))  # [N, 4]
        gamma0 = ada_params[:, 0:1]
        gamma1 = ada_params[:, 1:2]
        lambda_ = ada_params[:, 2:3]
        delta_ = ada_params[:, 3:4]

        # GTF_T (时间因果):
        feat_T_raw = obs_embed.mean(dim=-1).unsqueeze(1)  # [N, 1, T]
        feat_T = F.adaptive_avg_pool1d(feat_T_raw, h_half).squeeze(1)  # [N, h_half]

        temporal_offset = self.temporal_offset_net(feat_T) * gamma1
        base_grid_T = self._uniform_grid_1d(self.T_grid, device)
        adjusted_grid_T = (base_grid_T + temporal_offset).clamp(-1.0, 1.0)
        gtf_T_raw = self._grid_sample_1d(feat_T.unsqueeze(1), adjusted_grid_T).squeeze(1)
        gtf_T = gamma0 * F.adaptive_avg_pool1d(gtf_T_raw.unsqueeze(1), h_half).squeeze(1)

        # GTF_D (空间因果)
        feat_D_raw = obs_embed.mean(dim=1).unsqueeze(1)  # [N, 1, H]
        feat_D = F.adaptive_avg_pool1d(feat_D_raw, h_half).squeeze(1)  # [N, h_half]

        spatial_offset = self.spatial_offset_net(feat_D) * gamma0
        base_grid_D = self._uniform_grid_1d(self.D_grid_max, device)
        adjusted_grid_D = (base_grid_D + spatial_offset).clamp(-1.0, 1.0)
        gtf_D_raw = self._grid_sample_1d(feat_D.unsqueeze(1), adjusted_grid_D).squeeze(1)
        gtf_D = gamma0 * F.adaptive_avg_pool1d(gtf_D_raw.unsqueeze(1), h_half).squeeze(1)

        # 交叉注意力融合
        gtf_attn_out, _ = self.gtf_attn(gtf_T.unsqueeze(1), gtf_D.unsqueeze(1), gtf_D.unsqueeze(1))
        gtf_attn_scalar = gtf_attn_out.norm(dim=-1) / (h_half ** 0.5)
        gtf_scalar = gtf_attn_scalar * gamma0
        gtf_D_scalar = gtf_D.norm(dim=-1, keepdim=True) / (h_half ** 0.5)

        gtf = gtf_scalar + gtf_D_scalar
        return gtf, lambda_, delta_

    def compute_lif_seq(self, micro_feat_seq, density_seq, dir_seq, lif_scale, lif_shift):
        delta_z = torch.norm(micro_feat_seq[:, 1:] - micro_feat_seq[:, :-1], dim=-1, keepdim=True)

        # 密度变化率 Δρ: [N, T-1, 1]
        density_t = density_seq[:, 1:]
        density_prev = density_seq[:, :-1]
        delta_density = (density_t - density_prev) / (density_prev.abs() + 1e-8)

        # 运动趋势
        dir_t = dir_seq[:, 1:]

        lif_base = delta_z + delta_density + (1.0 - dir_t)
        lif_norm = torch.sigmoid(lif_base)

        lif_transformed = lif_scale.unsqueeze(1) * lif_norm + lif_shift.unsqueeze(1)
        return lif_transformed  # [N, T-1, 1]

    def multi_head_gat(self, node_feats, adj):
        N = node_feats.shape[0]

        h = torch.einsum('nd,kod->kno', node_feats, self.gat_w_weight) + self.gat_w_bias.unsqueeze(1)

        e_left = torch.einsum('knd,kd->kn', h, self.gat_attn_l)   # [K, N]
        e_right = torch.einsum('knd,kd->kn', h, self.gat_attn_r)  # [K, N]
        e = e_left.unsqueeze(-1) + e_right.unsqueeze(-2) + self.gat_attn_bias.view(-1, 1, 1)  # [K, N, N]
        e = F.leaky_relu(e, negative_slope=0.2)

        mask = (adj > 0).to(dtype=e.dtype).unsqueeze(0)  # [1, N, N]
        e = torch.where(mask > 0, e, torch.full_like(e, -1e4))
        alpha = F.softmax(e, dim=-1) * adj.unsqueeze(0)  # [K, N, N]

        # 消息聚合并拼接
        head_out = torch.einsum('kij,kjd->kid', alpha, h)  # [K, N, D_h]
        out = head_out.permute(1, 0, 2).reshape(N, self.num_heads * self.gat_head_dim)
        return self.gat_out_proj(out)

    def forward(self, micro_feat_seq, pos_seq, vel_seq, pos_fe_seq, adj_raw_seq, eff_nei_thre, adj_prev=None,):
        N, T, _ = micro_feat_seq.shape
        device = micro_feat_seq.device

        group_stats_seq = self.compute_group_stats_seq(pos_seq, vel_seq, adj_raw_seq)

        obs_input = torch.cat([micro_feat_seq, group_stats_seq, pos_fe_seq], dim=-1)
        obs_embed = self.obs_embed(obs_input)

        # 3. 从序列中提取全局因果因子 (GTF) [N, 1]
        gtf, lambda_, delta_ = self.compute_gtf(obs_embed)

        # 4. 提取所有时间步的 LIF 序列 [N, T-1, 1]
        density_seq = group_stats_seq[:, :, 0:1]
        dir_seq = group_stats_seq[:, :, 2:3]
        lif_seq = self.compute_lif_seq(micro_feat_seq, density_seq, dir_seq, lambda_, delta_)

        #动态拓扑演化
        h_input_seq = torch.cat([micro_feat_seq, pos_fe_seq, group_stats_seq], dim=-1)
        norm_feat_seq = F.normalize(h_input_seq[:, 1:], dim=-1)  # [N, T-1, D]

        # 计算所有时间步的节点相似度矩阵 Sim_t
        sim_seq = torch.bmm(norm_feat_seq.transpose(0, 1), norm_feat_seq.transpose(0, 1).transpose(1, 2))
        sim_seq = (sim_seq + 1.0) / 2.0

        adj_t = adj_prev if adj_prev is not None else adj_raw_seq[0]

        # 沿着时间步从 1 到 T-1 进行马尔科夫递推
        for t in range(T - 1):
            lif_t = lif_seq[:, t, :]  # [N, 1]
            sim_t = sim_seq[t]  # [N, N]

            # gtf [N,1] 和 lif_t [N,1] 在矩阵乘法中作为行缩放器自然广播
            adj_new = gtf * adj_t + (1.0 - gtf) * torch.sigmoid(lif_t * sim_t)
            adj_new = adj_new * (adj_new >= eff_nei_thre).float()
            adj_new.fill_diagonal_(0)
            adj_t = adj_new

        final_adj = adj_t

        last_step_feats = obs_embed[:, -1, :]  # [N, H]
        macro_feat = self.multi_head_gat(last_step_feats, final_adj)

        group_repr = macro_feat.mean(dim=0, keepdim=True)
        group_repr = self.group_pool_proj(group_repr)
        macro_feat = macro_feat + group_repr.expand(N, -1)

        return macro_feat, final_adj, None, gtf, lif_seq[:, -1, :], group_stats_seq[:, -1, :]


# ─────────────────────────────────────────────────────────────
# 3. 群体场生成模块
# ─────────────────────────────────────────────────────────────
class GroupFieldGenerator(nn.Module):
    def __init__(self, micro_dim, macro_dim, field_dim, grid_size=16):
        super(GroupFieldGenerator, self).__init__()
        self.grid_size = grid_size
        self.field_dim = field_dim

        self.fusion_proj = nn.Linear(micro_dim + macro_dim, field_dim)
        self.fusion_bias = nn.Parameter(torch.zeros(field_dim))
        self.activation = nn.Tanh()

    def bilinear_interpolate(self, individual_feats, positions, grid_size):
        N, D = individual_feats.shape
        device = individual_feats.device
        G = grid_size

        gx = torch.clamp(positions[:, 0] * (G - 1), 0, G - 1)
        gy = torch.clamp(positions[:, 1] * (G - 1), 0, G - 1)

        gx0 = gx.long()
        gy0 = gy.long()
        gx1 = torch.clamp(gx0 + 1, 0, G - 1)
        gy1 = torch.clamp(gy0 + 1, 0, G - 1)

        dx = gx - gx0.float()
        dy = gy - gy0.float()

        # 计算双线性插值权重 [N, 1]
        w00 = ((1 - dx) * (1 - dy)).unsqueeze(1)
        w10 = (dx * (1 - dy)).unsqueeze(1)
        w01 = ((1 - dx) * dy).unsqueeze(1)
        w11 = (dx * dy).unsqueeze(1)

        # 映射到展平的网格索引 [N, D]
        idx00 = (gx0 * G + gy0).unsqueeze(1).expand(-1, D)
        idx10 = (gx1 * G + gy0).unsqueeze(1).expand(-1, D)
        idx01 = (gx0 * G + gy1).unsqueeze(1).expand(-1, D)
        idx11 = (gx1 * G + gy1).unsqueeze(1).expand(-1, D)

        group_field_flat = torch.zeros(G * G, D, device=device)

        group_field_flat.scatter_add_(0, idx00, w00 * individual_feats)
        group_field_flat.scatter_add_(0, idx10, w10 * individual_feats)
        group_field_flat.scatter_add_(0, idx01, w01 * individual_feats)
        group_field_flat.scatter_add_(0, idx11, w11 * individual_feats)

        group_field = group_field_flat.view(G, G, D)
        return group_field

    def bilinear_interpolate_batched(self, individual_feats, positions, grid_size, agent_mask=None):
        B, N, D = individual_feats.shape
        device = individual_feats.device
        G = grid_size

        if agent_mask is None:
            agent_mask = torch.ones(B, N, device=device, dtype=individual_feats.dtype)

        gx = torch.clamp(positions[:, :, 0] * (G - 1), 0, G - 1)
        gy = torch.clamp(positions[:, :, 1] * (G - 1), 0, G - 1)

        gx0 = gx.long()
        gy0 = gy.long()
        gx1 = torch.clamp(gx0 + 1, 0, G - 1)
        gy1 = torch.clamp(gy0 + 1, 0, G - 1)

        dx = gx - gx0.float()
        dy = gy - gy0.float()

        m = agent_mask.unsqueeze(-1)
        w00 = ((1 - dx) * (1 - dy)).unsqueeze(-1) * m
        w10 = (dx * (1 - dy)).unsqueeze(-1) * m
        w01 = ((1 - dx) * dy).unsqueeze(-1) * m
        w11 = (dx * dy).unsqueeze(-1) * m

        batch_offset = (torch.arange(B, device=device).view(B, 1) * (G * G)).long()
        idx00 = (batch_offset + gx0 * G + gy0).unsqueeze(-1).expand(-1, -1, D)
        idx10 = (batch_offset + gx1 * G + gy0).unsqueeze(-1).expand(-1, -1, D)
        idx01 = (batch_offset + gx0 * G + gy1).unsqueeze(-1).expand(-1, -1, D)
        idx11 = (batch_offset + gx1 * G + gy1).unsqueeze(-1).expand(-1, -1, D)

        field_flat = torch.zeros(B * G * G, D, device=device, dtype=individual_feats.dtype)
        feats = individual_feats
        field_flat.scatter_add_(0, idx00.reshape(-1, D), (w00 * feats).reshape(-1, D))
        field_flat.scatter_add_(0, idx10.reshape(-1, D), (w10 * feats).reshape(-1, D))
        field_flat.scatter_add_(0, idx01.reshape(-1, D), (w01 * feats).reshape(-1, D))
        field_flat.scatter_add_(0, idx11.reshape(-1, D), (w11 * feats).reshape(-1, D))

        return field_flat.view(B, G, G, D)

    def forward(self, micro_feat, macro_feat, positions):
        concat_feat = torch.cat([micro_feat, macro_feat], dim=-1)
        fused_feat = self.fusion_proj(concat_feat) + self.fusion_bias

        group_field = self.bilinear_interpolate(fused_feat, positions, self.grid_size)
        G = self.grid_size
        input_grid = group_field.permute(2, 0, 1).unsqueeze(0)  # [1, D, G, G]
        sample_pos = positions.unsqueeze(0).unsqueeze(2) * 2.0 - 1.0  # [1, N, 1, 2]
        # 行人位置感知的特征 [N, D]
        sampled_feat = grid_sample(input_grid, sample_pos, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze().t()
        return fused_feat, group_field,sampled_feat

    def forward_batched(self, micro_feat, macro_feat, positions, agent_mask=None):
        concat_feat = torch.cat([micro_feat, macro_feat], dim=-1)
        fused_feat = self.fusion_proj(concat_feat) + self.fusion_bias
        group_field = self.bilinear_interpolate_batched(fused_feat, positions, self.grid_size, agent_mask=agent_mask)

        G = self.grid_size
        input_grid = group_field.permute(0, 3, 1, 2)  # [B, D, G, G]
        sample_pos = positions.unsqueeze(2) * 2.0 - 1.0  # [B, N, 1, 2]
        sampled_feat = grid_sample(input_grid, sample_pos, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_feat = sampled_feat.squeeze(-1).permute(0, 2, 1)  # [B, N, D]

        if agent_mask is not None:
            m = agent_mask.unsqueeze(-1).to(sampled_feat.dtype)
            sampled_feat = sampled_feat * m
            fused_feat = fused_feat * m

        return fused_feat, group_field, sampled_feat


# ─────────────────────────────────────────────────────────────
# 4. 显式社交力建模 (双向物理链路)
# ─────────────────────────────────────────────────────────────
class ExplicitSocialForce(nn.Module):
    def __init__(self, field_dim, hidden_dim):
        super(ExplicitSocialForce, self).__init__()
        self.field_dim = field_dim

        self.constraint_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(field_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, field_dim)
        )

    def compute_field_gradient(self, group_field, positions, grid_size):
        G = grid_size
        if group_field.dim() == 3:
            field_var = group_field.var(dim=-1)  # [G, G]
            sign = torch.sign(field_var.mean() - field_var.median())

            shift_up = torch.roll(field_var, shifts=-1, dims=0)
            shift_down = torch.roll(field_var, shifts=1, dims=0)
            shift_left = torch.roll(field_var, shifts=-1, dims=1)
            shift_right = torch.roll(field_var, shifts=1, dims=1)

            # 构建边界掩码
            mask_up = torch.ones_like(field_var)
            mask_up[-1, :] = 0
            mask_down = torch.ones_like(field_var)
            mask_down[0, :] = 0
            mask_left = torch.ones_like(field_var)
            mask_left[:, -1] = 0
            mask_right = torch.ones_like(field_var)
            mask_right[:, 0] = 0

            # 并行计算所有网格的邻居均值
            sum_neighbors = shift_up * mask_up + shift_down * mask_down + shift_left * mask_left + shift_right * mask_right
            count_neighbors = mask_up + mask_down + mask_left + mask_right
            neighbor_mean = sum_neighbors / count_neighbors

            # 得到全图梯度矩阵 [G, G]
            grad_matrix = field_var - neighbor_mean

            # 并行提取所有行人的梯度
            gx = torch.clamp((positions[:, 0] * (G - 1)).long(), 0, G - 1)
            gy = torch.clamp((positions[:, 1] * (G - 1)).long(), 0, G - 1)

            gradients = grad_matrix[gx, gy].abs().unsqueeze(1) * sign
            return gradients, field_var.var()

        T = group_field.shape[0]
        N = positions.shape[1]
        field_var = group_field.var(dim=-1)  # [T, G, G]
        field_var_flat = field_var.reshape(T, -1)
        sign = torch.sign(field_var_flat.mean(dim=1) - field_var_flat.median(dim=1).values).view(T, 1, 1)

        shift_up = torch.roll(field_var, shifts=-1, dims=1)
        shift_down = torch.roll(field_var, shifts=1, dims=1)
        shift_left = torch.roll(field_var, shifts=-1, dims=2)
        shift_right = torch.roll(field_var, shifts=1, dims=2)

        base_mask = torch.ones((G, G), device=field_var.device, dtype=field_var.dtype)
        mask_up = base_mask.clone()
        mask_up[-1, :] = 0
        mask_down = base_mask.clone()
        mask_down[0, :] = 0
        mask_left = base_mask.clone()
        mask_left[:, -1] = 0
        mask_right = base_mask.clone()
        mask_right[:, 0] = 0
        mask_up = mask_up.unsqueeze(0)
        mask_down = mask_down.unsqueeze(0)
        mask_left = mask_left.unsqueeze(0)
        mask_right = mask_right.unsqueeze(0)

        sum_neighbors = shift_up * mask_up + shift_down * mask_down + shift_left * mask_left + shift_right * mask_right
        count_neighbors = mask_up + mask_down + mask_left + mask_right
        neighbor_mean = sum_neighbors / count_neighbors
        grad_matrix = field_var - neighbor_mean  # [T, G, G]

        gx = torch.clamp((positions[:, :, 0] * (G - 1)).long(), 0, G - 1)  # [T, N]
        gy = torch.clamp((positions[:, :, 1] * (G - 1)).long(), 0, G - 1)  # [T, N]
        t_idx = torch.arange(T, device=positions.device).unsqueeze(1).expand(T, N)
        gradients = grad_matrix[t_idx, gx, gy].abs().unsqueeze(-1) * sign
        return gradients, field_var.var(dim=(1, 2))

    def compute_constraint_coeff(self, gradients, normalized_grad, max_gradient, t, t0=0, T_decay=20):
        t_tensor = torch.as_tensor(t, device=gradients.device, dtype=gradients.dtype)
        decay = torch.exp(-(t_tensor - t0) / (T_decay + 1e-8))
        if gradients.dim() == 3:
            if decay.dim() == 0:
                decay = decay.repeat(gradients.shape[0])
            decay = decay.view(-1, 1, 1)
        else:
            decay = decay.view(1, 1)
        mlp_out = self.constraint_mlp(gradients)
        constraint_coeff = torch.sigmoid(mlp_out * normalized_grad + decay)
        return constraint_coeff

    def forward(self, micro_feat, group_field, positions, grid_size, t=0):
        device = micro_feat.device

        gradients, max_grad = self.compute_field_gradient(group_field, positions, grid_size)
        if gradients.dim() == 3:
            max_gradient = max_grad.view(-1, 1, 1).to(device) + 1e-8
        else:
            max_gradient = torch.as_tensor(max_grad, device=device, dtype=micro_feat.dtype) + 1e-8

        normalized_grad = gradients / max_gradient
        constraint_coeff = self.compute_constraint_coeff(gradients, normalized_grad, max_gradient, t)

        constrained_micro = constraint_coeff * (1.0 - normalized_grad) * micro_feat

        if constrained_micro.dim() == 3:
            T = constrained_micro.shape[0]
            agg_individual = constrained_micro.mean(dim=1, keepdim=True)  # [T, 1, D]
            field_var = group_field.var(dim=-1)  # [T, G, G]
            field_var_scalar = field_var.mean(dim=(1, 2), keepdim=True)  # [T, 1, 1]
            field_var_threshold = field_var.reshape(T, -1).median(dim=1).values.view(T, 1, 1)  # [T, 1, 1]
            motion_mask = (field_var_scalar < field_var_threshold).float()  # [T, 1, 1]

            delta_field = self.update_mlp(agg_individual) * motion_mask  # [T, 1, D]
            updated_group_field = group_field + delta_field.view(T, 1, 1, -1)
        else:
            agg_individual = constrained_micro.mean(dim=0, keepdim=True)
            field_var_scalar = group_field.var(dim=-1).mean()
            field_var_threshold = group_field.var(dim=-1).median()
            motion_mask = (field_var_scalar < field_var_threshold).float()

            delta_field = self.update_mlp(agg_individual) * motion_mask
            updated_group_field = group_field + delta_field.view(1, 1, -1)

        return constrained_micro, updated_group_field


# ─────────────────────────────────────────────────────────────
# 5. 隐式社交力建模 (物理势场)
# ─────────────────────────────────────────────────────────────
class ImplicitSocialForce(nn.Module):
    def __init__(self, safe_distance=1.5, epsilon=1e-5, k_neighbors=5):
        super(ImplicitSocialForce, self).__init__()
        self.safe_distance = safe_distance
        self.epsilon = epsilon
        self.k_neighbors = k_neighbors

        self.sensitivity_bias = nn.Parameter(torch.tensor(0.5))
        self.variance_factor = nn.Parameter(torch.tensor(1.0))

    def compute_social_sensitivity(self, adj_weights):
        if adj_weights.dim() == 3:
            mean_weights = adj_weights.mean(dim=2, keepdim=True).pow(2)  # [T, N, 1]
        else:
            mean_weights = adj_weights.mean(dim=1, keepdim=True).pow(2)  # [N, 1]
        sensitivity = torch.sigmoid(mean_weights + self.sensitivity_bias)
        return sensitivity

    def compute_distances(self, positions):
        if positions.dim() == 3:
            T, N, _ = positions.shape
            if N <= 1:
                return torch.ones(T, N, 1, device=positions.device, dtype=positions.dtype) * self.safe_distance

            dist_matrix = torch.cdist(positions, positions)  # [T, N, N]
            eye_mask = torch.eye(N, device=positions.device, dtype=torch.bool).unsqueeze(0)
            dist_matrix = dist_matrix.masked_fill(eye_mask, float('inf'))

            k = min(self.k_neighbors, N - 1)
            knn_dist, _ = dist_matrix.topk(k, largest=False, dim=-1)  # [T, N, k]
            avg_dist = knn_dist.mean(dim=-1, keepdim=True)  # [T, N, 1]
            return avg_dist

        N = positions.shape[0]
        if N <= 1:
            return torch.ones(N, 1, device=positions.device, dtype=positions.dtype) * self.safe_distance

        dist_matrix = torch.cdist(positions, positions)
        dist_matrix.fill_diagonal_(float('inf'))

        k = min(self.k_neighbors, N - 1)
        knn_dist, _ = dist_matrix.topk(k, largest=False)
        avg_dist = knn_dist.mean(dim=1, keepdim=True)
        return avg_dist

    def forward(self, adj_weights, sampled_feat, positions, prev_sampled_feat=None):
        if sampled_feat.dim() == 3:
            field_energy = torch.var(sampled_feat, dim=2, keepdim=True)  # [T, N, 1]
            field_energy.retain_grad()
        else:
            field_energy = torch.var(sampled_feat, dim=1, keepdim=True)  # [N, 1]
            field_energy.retain_grad()
        avg_dist = self.compute_distances(positions)  # [N, 1]

        #灵敏度
        sensitivity = self.compute_social_sensitivity(adj_weights)  # [N, 1]

        # 斥力
        f_rep = sensitivity * (1.0 / (field_energy + self.epsilon)) * \
                torch.exp(-avg_dist ** 2 / self.safe_distance ** 2)
        # 引力
        f_att = sensitivity * torch.sigmoid(field_energy) * \
                (1.0 - torch.exp(-avg_dist ** 2 / self.safe_distance ** 2))

        if prev_sampled_feat is None:
            if field_energy.dim() == 3:
                prev_energy = torch.zeros_like(field_energy)
                prev_energy[1:] = field_energy[:-1]
                prev_energy[0] = prev_energy[1] if field_energy.shape[0] > 1 else field_energy[0]
                delta_var = field_energy - prev_energy
            else:
                delta_var = field_energy
        else:
            delta_var = field_energy - prev_sampled_feat

        omega_t = torch.sigmoid(delta_var)
        implicit_force = omega_t * f_rep - (1.0 - omega_t) * f_att

        return implicit_force, field_energy


# ─────────────────────────────────────────────────────────────
# 6. 因果约束增强与特征融合
# ─────────────────────────────────────────────────────────────
class CausalConstraintEnhancement(nn.Module):
    def __init__(self, eps=1e-8):
        super(CausalConstraintEnhancement, self).__init__()
        self.loss_weights_logits = nn.Parameter(torch.zeros(4))
        self.eps = eps

    def _get_elementwise_grad_from_list(self, y_list, x_list):
        if len(y_list) != len(x_list):
            raise ValueError(f"`y_list` and `x_list` must have the same length, got {len(y_list)} vs {len(x_list)}")

        if len(y_list) > 0 and isinstance(y_list[0], (list, tuple)):
            if len(x_list) == 0:
                raise ValueError("`x_list` is empty while `y_list` is nested.")
            num_scenes = len(y_list[0])
            grads_all_scenes = []
            for s in range(num_scenes):
                y_scene = [y_list[t][s] for t in range(len(y_list))]
                x_scene = [x_list[t][s] for t in range(len(x_list))]
                grads_all_scenes.append(self._get_elementwise_grad_from_list(y_scene, x_scene))
            return torch.cat(grads_all_scenes, dim=0)

        valid_pairs = [(y, x) for y, x in zip(y_list, x_list) if y is not None and x is not None]
        valid_y = [p[0] for p in valid_pairs]
        valid_x = [p[1] for p in valid_pairs]

        template = None
        for t in y_list:
            if t is not None:
                template = t
                break
        if template is None:
            for t in x_list:
                if t is not None:
                    template = t
                    break

        if not valid_y:
            if template is None:
                raise ValueError("Both y_list and x_list are all None; cannot infer tensor shape.")
            N, D = template.shape[0], template.shape[-1]
            return torch.zeros(N, len(y_list), D, device=template.device, dtype=template.dtype)

        y_stacked = torch.stack(valid_y, dim=0)  # [T_valid, N, D]
        grads_list = torch.autograd.grad(
            outputs=y_stacked,
            inputs=valid_x,
            grad_outputs=torch.ones_like(y_stacked),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )
        grads_list_safe = []
        for g, x in zip(grads_list, valid_x):
            if g is None:
                grads_list_safe.append(torch.zeros_like(x))
            else:
                grads_list_safe.append(g)

        return torch.stack(grads_list_safe, dim=1)

    def implicit_social_constraint(self, social_force_list, sample_feat_list):
        # 瞬时梯度项
        inst_grad = self._get_elementwise_grad_from_list(social_force_list, sample_feat_list)
        # 势梯度项
        if len(sample_feat_list) > 0 and isinstance(sample_feat_list[0], (list, tuple)):
            sample_feat_seq_list = [torch.cat(sample_feat_list[t], dim=0) for t in range(len(sample_feat_list))]
            sample_feat_seq = torch.stack(sample_feat_seq_list, dim=1)
        else:
            sample_feat_seq = torch.stack(sample_feat_list, dim=1)
        evo_grad = torch.zeros_like(sample_feat_seq)
        evo_grad[:, 1:] = sample_feat_seq[:, 1:] - sample_feat_seq[:, :-1]
        evo_grad[:, 0] = evo_grad[:, 1]  # 边界填充

        loss = inst_grad * evo_grad
        loss = torch.log1p(loss.sum().abs())
        return loss

    def global_causal_constraint(self, fused_feat_list, macro_feat_list, social_force_list):
        # --- 宏观部分 (Macro) ---
        inst_grad_m = self._get_elementwise_grad_from_list(fused_feat_list, macro_feat_list)
        if len(macro_feat_list) > 0 and isinstance(macro_feat_list[0], (list, tuple)):
            macro_feat_seq_list = [torch.cat(macro_feat_list[t], dim=0) for t in range(len(macro_feat_list))]
            macro_feat_seq = torch.stack(macro_feat_seq_list, dim=1)
        else:
            macro_feat_seq = torch.stack(macro_feat_list, dim=1)
        evo_grad_m = torch.zeros_like(macro_feat_seq)
        evo_grad_m[:, 1:] = macro_feat_seq[:, 1:] - macro_feat_seq[:, :-1]
        term_macro = inst_grad_m * evo_grad_m

        # --- 社交部分 (Social Force) ---
        inst_grad_s = self._get_elementwise_grad_from_list(fused_feat_list, social_force_list)
        if len(social_force_list) > 0 and isinstance(social_force_list[0], (list, tuple)):
            social_force_seq_list = [torch.cat(social_force_list[t], dim=0) for t in range(len(social_force_list))]
            social_force_seq = torch.stack(social_force_seq_list, dim=1)
        else:
            social_force_seq = torch.stack(social_force_list, dim=1)
        evo_grad_s = torch.zeros_like(social_force_seq)
        evo_grad_s[:, 1:] = social_force_seq[:, 1:] - social_force_seq[:, :-1]
        term_social = inst_grad_s * evo_grad_s

        loss = term_macro + term_social
        loss = torch.log1p(loss.sum().abs())
        return loss

    def micro_causal_constraint(self, micro_feat_seq):

        N, T, _ = micro_feat_seq.shape
        if T < 4: return torch.tensor(0.0, device=micro_feat_seq.device, requires_grad=True)

        diff = torch.norm(micro_feat_seq[:, 1:] - micro_feat_seq[:, :-1], p=2, dim=-1)
        epsilon = 1 if (N / 50) > 1.2 else 2

        grad_ratio = diff[:, epsilon:] / (diff[:, :T - 1 - epsilon] + self.eps)

        t_idx = torch.arange(epsilon, T - 1, device=micro_feat_seq.device).float()
        w_t = torch.exp(-((t_idx - (T - 1) / 2.0) ** 2) / (2.0 * (T / 4.0) ** 2))

        z_curr = micro_feat_seq[:, epsilon: T - 1]
        z_prev = micro_feat_seq[:, epsilon - 1: T - 2]
        gamma_t = (F.cosine_similarity(z_curr, z_prev, dim=-1) + 1.0) / 2.0
        loss_init = (grad_ratio * w_t.unsqueeze(0) * gamma_t).sum()
        loss = torch.log1p(loss_init.abs())
        return loss

    def macro_causal_constraint(self, macro_feat_seq, adj_weights):
        if len(macro_feat_seq) > 0 and isinstance(macro_feat_seq[0], (list, tuple)):
            macro_feat_seq_list = [torch.cat(macro_feat_seq[t], dim=0) for t in range(len(macro_feat_seq))]
            macro_feat_seq = torch.stack(macro_feat_seq_list, dim=1)
        else:
            macro_feat_seq = torch.stack(macro_feat_seq, dim=1)

        m_mean = macro_feat_seq.mean(dim=1)
        m_norm = F.normalize(m_mean, p=2, dim=-1)
        sim_matrix = torch.mm(m_norm, m_norm.t())
        mask = (adj_weights > 0).float()
        return F.mse_loss(adj_weights * mask, sim_matrix * mask, reduction='sum') / (mask.sum() + self.eps)

    def forward(self, micro_feat_seq, macro_feat_seq, implicit_force_seq,
                fused_feat_seq, group_field_seq, adj_weights,sample_feat_seq):

        weights = F.softmax(self.loss_weights_logits, dim=0)

        l_micro = self.micro_causal_constraint(micro_feat_seq)  # Eq 30
        l_macro = self.macro_causal_constraint(macro_feat_seq, adj_weights)  # Eq 31
        l_social = self.implicit_social_constraint(implicit_force_seq, sample_feat_seq)  # Eq 32
        l_global = self.global_causal_constraint(fused_feat_seq, macro_feat_seq, implicit_force_seq)  # Eq 34

        loss = {
            'L_micro': weights[0] * l_micro,
            'L_macro': weights[1] * l_macro,
            'L_social': weights[2] * l_social,
            'L_global': weights[3] * l_global
        }
        return loss

# ─────────────────────────────────────────────────────────────
# 8. 主模型 CaDSL-Traj
# ─────────────────────────────────────────────────────────────
class CaDSLTraj(nn.Module):
    def __init__(self, args):
        super(CaDSLTraj, self).__init__()
        self.args = args

        input_dim       = self.args.input_size
        obs_length      = self.args.obs_length
        pred_length     = self.args.pred_length
        micro_dim       = self.args.hidden_size
        macro_dim       = self.args.hidden_size
        hidden_dim      = 2 * self.args.hidden_size
        field_dim       = self.args.hidden_size
        grid_size       = self.args.grid_size
        num_heads       = self.args.num_heads
        num_samples     = self.args.num_samples
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.num_samples = num_samples
        self.eff_nei_thre = self.args.eff_nei_thre
        self.lambda_loss = self.args.lambda_loss
        self.pos_embed = nn.Linear(input_dim, 5)

        self.iice_conv = IICEConv(in_channels=5, out_channels=micro_dim, kernel_size=3)
        self.gia = GroupInteractionAware(micro_feat_dim=micro_dim, group_stat_dim=3, hidden_dim=macro_dim, num_heads=num_heads, pos_fe_dim=5)
        self.field_gen = GroupFieldGenerator(micro_dim=micro_dim, macro_dim=macro_dim, field_dim=field_dim, grid_size=grid_size)
        self.grid_size = grid_size
        self.explicit_sf = ExplicitSocialForce(field_dim=field_dim, hidden_dim=hidden_dim)
        self.implicit_sf = ImplicitSocialForce(safe_distance=1.5, k_neighbors=5)
        self.causal_enhance = CausalConstraintEnhancement()
        self.predictor = MLPDecoder(self.args)
        self.mapping = nn.Linear(1,micro_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * micro_dim, 4 * micro_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(4 * micro_dim, micro_dim),
        )
        if self.args.ifGaussian:
            self.reg_loss = GaussianNLLLoss(reduction='mean')
        else:
            self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        self.use_torch_compile = getattr(args, 'use_torch_compile', False)

        if self.use_torch_compile and hasattr(torch, "compile"):
            try:
                self.iice_conv = torch.compile(self.iice_conv, mode="reduce-overhead")
                self.gia = torch.compile(self.gia, mode="reduce-overhead")
                self.field_gen = torch.compile(self.field_gen, mode="reduce-overhead")
                self.implicit_sf = torch.compile(self.implicit_sf, mode="reduce-overhead")
                self.fusion_mlp = torch.compile(self.fusion_mlp, mode="reduce-overhead")
            except Exception:
                self.use_torch_compile = False
    def normalize_positions(self, positions):
        pos_min = positions.min(dim=0, keepdim=True)[0]
        pos_max = positions.max(dim=0, keepdim=True)[0]
        pos_range = (pos_max - pos_min).clamp(min=1e-8)
        return (positions - pos_min) / pos_range

    def build_adjacency_seq(self, positions_seq, nei_list_scene_seq=None):
        T, n, _ = positions_seq.shape
        device = positions_seq.device

        if n == 0:
            return torch.zeros(T, 0, 0, device=device)

        if nei_list_scene_seq is not None:
            adj_seq = nei_list_scene_seq.float().to(device)

            if adj_seq.dim() == 3:
                adj_seq = adj_seq[:T, :n, :n]
            elif adj_seq.dim() == 2:
                adj_seq = adj_seq[:n, :n].unsqueeze(0).expand(T, -1, -1)
        else:
            adj_seq = torch.ones(T, n, n, device=device)

        eye_mask = torch.eye(n, device=device).unsqueeze(0).expand(T, -1, -1)
        adj_seq = adj_seq * (1.0 - eye_mask)

        if n > 1:
            dist_seq = torch.cdist(positions_seq, positions_seq)
            mean_dist = dist_seq.mean(dim=(1, 2), keepdim=True) + 1e-8
            dist_weight = torch.exp(-dist_seq / mean_dist)
            adj_seq = adj_seq * dist_weight

        return adj_seq

    def mdn_loss(self, y, pre_obs, y_prime, iftest):
        """
        y: [N, H, 2]  (GT 预测段轨迹)
        y_prime: (out_mu, out_sigma, out_pi)
          out_mu:   [F, N, H, 2]
          out_sigma:[F, N, H, 2]
          out_pi:   [N, F]
        """
        batch_size = y.shape[0]
        pre_obs = pre_obs.permute(1,0,2)
        out_mu, out_sigma, out_pi = y_prime
        y_hat = torch.cat((out_mu, out_sigma), dim=-1)
        loss, reg_loss, cls_loss = 0, 0, 0
        full_pre_tra = []
        l2_norm = (torch.norm(out_mu - y.unsqueeze(0), p=2, dim=-1)).sum(dim=-1)   # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(batch_size)]
        if not iftest:
            reg_loss += self.reg_loss(y_hat_best, y)
        soft_target = F.softmax(-l2_norm / self.args.pred_length, dim=0).t().detach() # [N, F]
        if not iftest:
            cls_loss += self.cls_loss(out_pi, soft_target)
            loss = reg_loss + cls_loss
        #best ADE
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((pre_obs,sample_k), axis=0))
        # best FDE
        l2_norm_FDE = (torch.norm(out_mu[:, :, -1, :] - y[:, -1, :].unsqueeze(0), p=2, dim=-1))  # [F, N]
        best_mode = l2_norm_FDE.argmin(dim=0)
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((pre_obs,sample_k), axis=0))
        return loss, full_pre_tra

    def forward(self, inputs_fw, batch, iftest=False, ifvisualize=False):
        batch_abs, batch_norm, nei_lists, nei_num, batch_split = inputs_fw
        device = batch_abs.device
        N, T_total, _ = batch_abs.shape
        T_obs = self.obs_length

        obs_abs = batch_abs[:, :T_obs, :]
        pred_abs = batch_abs[:, T_obs:, :]
        # 1. 微观因果序列
        vel_all = torch.zeros(N, T_obs, 2, device=device)
        vel_all[:, 1:] = obs_abs[:, 1:] - obs_abs[:, :-1]
        speed_all = vel_all.norm(dim=-1, keepdim=True)
        direction_all = torch.atan2(vel_all[..., 1:2], vel_all[..., 0:1] + 1e-8)
        time_steps_all = (torch.arange(T_obs, device=device).float() / T_obs).view(1, T_obs, 1).expand(N, -1, -1)

        feat_5d_all = torch.cat([obs_abs, speed_all, direction_all, time_steps_all], dim=-1)
        micro_feat_seq_all = self.iice_conv(feat_5d_all)

        rep_loss = torch.tensor(0.0, device=device)
        batch_micro_multi_list = []
        batch_macro_feat_list = []
        batch_adj_blocks = []
        batch_implicit_force_per_t = [[] for _ in range(T_obs)]
        batch_group_field_per_t = [[] for _ in range(T_obs)]
        batch_fused_feat_per_t = [[] for _ in range(T_obs)]
        batch_sample_feat_per_t = [[] for _ in range(T_obs)]
        batch_micro_single_list = []
        batch_fused_seq_list = []

        buckets = {}
        for scene_idx, (start, end) in enumerate(batch_split):
            n = end - start
            if n not in buckets:
                buckets[n] = []
            buckets[n].append((scene_idx, start, end))

        for n in sorted(buckets.keys()):
            scene_items = buckets[n]

            # 单人场景快速路径
            if n == 1:
                for _, start, end in scene_items:
                    scene_micro_seq = micro_feat_seq_all[start:end]
                    batch_micro_single_list.append(scene_micro_seq)
                    batch_fused_seq_list.append(scene_micro_seq)
                continue

            nei_bucket = []
            for scene_idx, _, _ in scene_items:
                if scene_idx < len(nei_lists):
                    nei_bucket.append(torch.as_tensor(nei_lists[scene_idx], device=device))
                else:
                    nei_bucket.append(torch.ones(T_obs, n, n, device=device))

            scene_cache = []
            for (_, start, end), nei_list_scene in zip(scene_items, nei_bucket):
                scene_micro_seq = micro_feat_seq_all[start:end]  # [n, T_obs, D]
                scene_pos_all = obs_abs[start:end]  # [n, T_obs, 2]
                scene_vel_all = vel_all[start:end]  # [n, T_obs, 2]
                scene_feat5d_all = feat_5d_all[start:end]  # [n, T_obs, 5]

                pos_min_t = scene_pos_all.min(dim=0, keepdim=True)[0]
                pos_max_t = scene_pos_all.max(dim=0, keepdim=True)[0]
                pos_range_t = (pos_max_t - pos_min_t).clamp(min=1e-8)
                norm_pos_all = (scene_pos_all - pos_min_t) / pos_range_t

                pos_seq_T_first = scene_pos_all.transpose(0, 1)
                adj_raw_seq = self.build_adjacency_seq(pos_seq_T_first, nei_list_scene)
                macro_feat, adj_updated, _, _, _, _ = self.gia(
                    scene_micro_seq,
                    scene_pos_all,
                    scene_vel_all,
                    scene_feat5d_all,
                    adj_raw_seq,
                    eff_nei_thre=self.eff_nei_thre,
                    adj_prev=None
                )
                scene_cache.append({
                    "scene_micro_seq": scene_micro_seq,
                    "scene_pos_all": scene_pos_all,
                    "norm_pos_all": norm_pos_all,
                    "macro_feat": macro_feat,
                    "adj_updated": adj_updated,
                })

            field_scene_micro_batch = int(getattr(self.args, 'field_scene_micro_batch', 4))
            field_scene_micro_batch = max(1, field_scene_micro_batch)
            D_micro = scene_cache[0]["scene_micro_seq"].shape[-1]
            D_macro = scene_cache[0]["macro_feat"].shape[-1]

            for chunk_start in range(0, len(scene_cache), field_scene_micro_batch):
                chunk = scene_cache[chunk_start: chunk_start + field_scene_micro_batch]
                valid_b = len(chunk)
                B = field_scene_micro_batch

                micro_pad = torch.zeros(B, n, T_obs, D_micro, device=device)
                norm_pos_pad = torch.zeros(B, n, T_obs, 2, device=device)
                macro_pad = torch.zeros(B, n, D_macro, device=device)
                agent_mask = torch.zeros(B, n, device=device, dtype=micro_pad.dtype)

                for bi, sc in enumerate(chunk):
                    micro_pad[bi] = sc["scene_micro_seq"]
                    norm_pos_pad[bi] = sc["norm_pos_all"]
                    macro_pad[bi] = sc["macro_feat"]
                    agent_mask[bi] = 1.0

                group_field_seq_batch = []
                sample_feat_seq_batch = []
                for t in range(T_obs):
                    t_m_feat = micro_pad[:, :, t, :]
                    t_norm_pos = norm_pos_pad[:, :, t, :]
                    _, t_field_b, t_sample_b = self.field_gen.forward_batched(
                        t_m_feat, macro_pad, t_norm_pos, agent_mask=agent_mask
                    )
                    group_field_seq_batch.append(t_field_b)
                    sample_feat_seq_batch.append(t_sample_b)

                for bi in range(valid_b):
                    sc = chunk[bi]
                    scene_micro_seq = sc["scene_micro_seq"]
                    scene_pos_all = sc["scene_pos_all"]
                    norm_pos_all = sc["norm_pos_all"]
                    macro_feat = sc["macro_feat"]
                    adj_updated = sc["adj_updated"]

                    group_field_seq = [group_field_seq_batch[t][bi] for t in range(T_obs)]
                    sample_feat_list = [sample_feat_seq_batch[t][bi] for t in range(T_obs)]

                    sample_feat_seq_t = torch.stack(sample_feat_list, dim=0)  # [T_obs, n, D]
                    scene_pos_all_t = scene_pos_all.transpose(0, 1)  # [T_obs, n, 2]
                    adj_updated_t = adj_updated.unsqueeze(0).expand(T_obs, -1, -1)  # [T_obs, n, n]
                    implicit_force_t, _ = self.implicit_sf(adj_updated_t, sample_feat_seq_t, scene_pos_all_t, prev_sampled_feat=None)
                    implicit_force_tensor = implicit_force_t.transpose(0, 1)  # [n, T_obs, 1]
                    implicit_force_list = [implicit_force_t[t] for t in range(T_obs)]  # 保持后续 list 结构

                    imp_force_map_tensor = self.mapping(implicit_force_tensor)  # [n, T_obs, D]
                    macro_feat_expanded = macro_feat.unsqueeze(1).expand(-1, T_obs, -1)  # [n, T_obs, D]

                    scene_micro_seq_t = scene_micro_seq.transpose(0, 1)  # [T_obs, n, D]
                    group_field_seq_t = torch.stack(group_field_seq, dim=0)  # [T_obs, G, G, D]
                    norm_pos_all_t = norm_pos_all.transpose(0, 1)  # [T_obs, n, 2]
                    time_index = torch.arange(T_obs, device=device, dtype=scene_micro_seq.dtype)

                    constrained_micro_t, updated_group_field_t = self.explicit_sf(
                        scene_micro_seq_t,
                        group_field_seq_t,
                        norm_pos_all_t,
                        self.grid_size,
                        t=time_index
                    )
                    constrained_micro_seq = constrained_micro_t.transpose(0, 1)  # [n, T_obs, D]
                    updated_group_field_seq = [updated_group_field_t[t] for t in range(T_obs)]

                    concat_feat_tensor = torch.cat([constrained_micro_seq, macro_feat_expanded, imp_force_map_tensor], dim=-1)
                    fused_feat_tensor = self.fusion_mlp(concat_feat_tensor)  # [n, T_obs, D]
                    fused_feat_list = [fused_feat_tensor[:, t, :] for t in range(T_obs)]

                    batch_micro_multi_list.append(constrained_micro_seq)
                    batch_macro_feat_list.append(macro_feat)
                    batch_adj_blocks.append(adj_updated)
                    for t in range(T_obs):
                        batch_implicit_force_per_t[t].append(implicit_force_list[t])
                        # 用更新后的场替代原 group_field_seq。
                        batch_group_field_per_t[t].append(updated_group_field_seq[t])
                        batch_fused_feat_per_t[t].append(fused_feat_list[t])
                        batch_sample_feat_per_t[t].append(sample_feat_list[t])

                    batch_fused_seq_list.append(fused_feat_tensor)

        if not iftest:
            if len(batch_micro_multi_list) > 0:
                micro_feat_batch = torch.cat(batch_micro_multi_list, dim=0)
                if len(batch_adj_blocks) == 1:
                    adj_weights_batch = batch_adj_blocks[0]
                else:
                    adj_weights_batch = torch.block_diag(*batch_adj_blocks)

                constraint_losses = self.causal_enhance(
                    micro_feat_seq=micro_feat_batch,
                    macro_feat_seq=[batch_macro_feat_list for _ in range(T_obs)],
                    implicit_force_seq=batch_implicit_force_per_t,
                    group_field_seq=batch_group_field_per_t,
                    fused_feat_seq=batch_fused_feat_per_t,
                    adj_weights=adj_weights_batch,
                    sample_feat_seq=batch_sample_feat_per_t 
                )
                rep_loss = rep_loss + sum([l for k, l in constraint_losses.items() if k.startswith('L_')])

            if len(batch_micro_single_list) > 0:
                micro_single_batch = torch.cat(batch_micro_single_list, dim=0)
                rep_loss = rep_loss + self.causal_enhance.micro_causal_constraint(micro_feat_seq=micro_single_batch)

        fused_feat_batch_all = torch.cat(batch_fused_seq_list, dim=0)  # [N, T_obs, D]
        last_fused_feat = fused_feat_batch_all[:, -1, :]  # [N, D]
        last_micro_feat = micro_feat_seq_all[:, -1, :]  # [N, D]

        mdn_out = self.predictor(last_fused_feat, last_micro_feat)
        pre_loss, scene_pred = self.mdn_loss(pred_abs, obs_abs, mdn_out, iftest)
        rep_loss = rep_loss / (len(batch_split) + 1e-8)
        pre_loss = pre_loss / (1.0 + 1e-8)
        lambda_loss = self.lambda_loss
        total_loss = lambda_loss * pre_loss + (1-lambda_loss) * rep_loss
        return total_loss, scene_pred