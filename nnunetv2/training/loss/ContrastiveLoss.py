import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SegContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0,
                 ignore_background=True,
                 sample_rate=0.001,
                 chunk_size=128,
                 max_points=1000,
                 use_fp16=True):
        """
        10GB显存优化的分割对比损失
        Args:
            margin: 负样本边界值
            ignore_background: 必须启用以节省显存
            sample_rate: 初始采样率(0.001~0.005)
            chunk_size: 分块计算大小(64~256)
            max_points: 最大采样点数(硬性限制)
            use_fp16: 启用FP16混合精度
        """
        super().__init__()
        self.margin = margin
        self.ignore_bg = ignore_background
        self.sample_rate = max(0.001, sample_rate)
        self.chunk_size = chunk_size
        self.max_points = max_points
        self.use_fp16 = use_fp16

    def _dynamic_sampling(self, feats, labels):
        B, C, H, W = feats.size()
        total_pixels = H * W
        target_samples = min(self.max_points, int(total_pixels * self.sample_rate))
        target_samples = max(1, target_samples)  # 确保至少采样1个点 [3,7](@ref)

        # 网格采样
        grid_step = max(2, int(math.sqrt(total_pixels / target_samples)))
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, grid_step),
            torch.arange(0, W, grid_step),
            indexing='ij'
        )
        grid_idx = (grid_y * W + grid_x).flatten()

        # 随机采样补足
        rand_samples = target_samples - len(grid_idx)
        if rand_samples > 0:
            rand_idx = torch.randint(0, total_pixels, (rand_samples,))
            total_idx = torch.cat([grid_idx, rand_idx])
        else:
            total_idx = grid_idx[:target_samples]

        flat_feats = feats.view(B, C, -1)[..., total_idx.to(feats.device)]
        flat_labels = labels.view(B, 1, -1)[..., total_idx.to(labels.device)]
        return flat_feats, flat_labels

    def _compute_chunk_loss(self, anchor_feat, anchor_label, global_feats, global_labels):
        """Inf-CL分块计算策略核心"""
        # 计算当前锚点与全局特征距离
        dist = torch.norm(global_feats - anchor_feat, p=2, dim=1)  # [N]

        # 正负样本掩码
        pos_mask = (global_labels == anchor_label)
        neg_mask = ~pos_mask

        # 排除自身
        self_mask = torch.isclose(dist, torch.tensor(0.0, device=dist.device))
        pos_mask = pos_mask & ~self_mask

        # 计算损失分量
        pos_loss = 0.5 * (dist[pos_mask] ** 2).sum() if pos_mask.any() else 0.0
        neg_loss = 0.5 * F.relu(self.margin - dist[neg_mask]).pow(2).sum() if neg_mask.any() else 0.0

        # 有效样本对数
        valid_pairs = pos_mask.sum().item() + neg_mask.sum().item()
        return pos_loss + neg_loss, valid_pairs

    def forward(self, inputs, labels):
        # 混合精度上下文
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            if self.use_fp16:
                inputs = inputs.half()

            # 1. 动态空间采样
            feats, labels = self._dynamic_sampling(inputs, labels)  # [B, C, S]
            feats = feats.permute(0, 2, 1)  # [B, S, C]

            # 2. 背景过滤
            B, S, C = feats.size()
            if self.ignore_bg:
                non_bg_mask = (labels.squeeze(1) != 0)  # [B, S]
            else:
                non_bg_mask = torch.ones(B, S, dtype=torch.bool, device=feats.device)

            total_loss = 0
            valid_batch_count = 0

            # 3. 逐样本处理
            for b in range(B):
                # 提取有效点
                cur_mask = non_bg_mask[b]  # [S]
                if cur_mask.sum() < 2:
                    continue

                cur_feats = feats[b][cur_mask]  # [N, C]
                cur_labels = labels[b, 0, cur_mask]  # [N]
                N = len(cur_feats)

                # 4. Inf-CL分块计算
                chunk_loss = 0
                total_valid_pairs = 0

                # 分块处理锚点
                for i in range(0, N, self.chunk_size):
                    chunk_end = min(i + self.chunk_size, N)
                    for j in range(i, chunk_end):
                        # 单锚点计算（避免矩阵存储）
                        anchor_feat = cur_feats[j].unsqueeze(0)  # [1, C]
                        anchor_label = cur_labels[j].unsqueeze(0)  # [1]

                        # 计算当前锚点损失
                        batch_loss, valid_pairs = self._compute_chunk_loss(
                            anchor_feat, anchor_label, cur_feats, cur_labels
                        )
                        chunk_loss += batch_loss
                        total_valid_pairs += valid_pairs

                # 聚合批次损失
                if total_valid_pairs > 0:
                    total_loss += chunk_loss / total_valid_pairs
                    valid_batch_count += 1

            return total_loss / valid_batch_count if valid_batch_count > 0 else 0.0