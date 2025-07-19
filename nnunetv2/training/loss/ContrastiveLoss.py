import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleContrastiveLoss(nn.Module):
    """
    简化版对比学习损失模块
    核心功能：通过InfoNCE损失拉近正样本距离，推开负样本距离
    """

    def __init__(self, temperature=0.5, min_views=2):
        """
        :param temperature: 温度系数，控制相似度分布的尖锐程度（值越小对困难样本越敏感）
        :param min_views: 每类最少采样数，避免样本不足的类别
        """
        super().__init__()
        self.temperature = temperature
        self.min_views = min_views
        self.eps = 1e-10  # 数值稳定常数

    def forward(self, features, labels):
        """
        :param features: 特征张量 (B, C, H, W)
        :param labels: 类别标签 (B, H, W)
        :return: 对比损失值
        """
        # 1. 降采样标签匹配特征尺寸
        scale = features.shape[2] / labels.shape[1]
        labels_ds = F.interpolate(
            labels.unsqueeze(1).float(),
            scale_factor=scale,
            mode='nearest'
        ).long().squeeze(1)

        # 2. 采样锚点特征
        sampled_feats, sampled_labels = self._sample_anchors(features, labels_ds)

        # 3. 计算InfoNCE损失
        return self._compute_infonce(sampled_feats, sampled_labels)

    def _sample_anchors(self, features, labels):
        """采样每类特征向量"""
        B, C, H, W = features.shape
        features_flat = features.view(B, C, -1)  # (B, C, H*W)
        labels_flat = labels.view(B, -1)  # (B, H*W)

        sampled_feats = []
        sampled_labels = []

        # 遍历batch内每张图像的类别
        for b in range(B):
            unique_classes = torch.unique(labels_flat[b])
            for cls in unique_classes:
                if cls == -1: continue  # 跳过忽略类

                # 获取当前类别的所有特征
                cls_mask = (labels_flat[b] == cls)
                cls_feats = features_flat[b, :, cls_mask]

                # 确保满足最小采样数
                if cls_feats.shape[1] < self.min_views:
                    continue

                # 随机采样特征
                rand_idx = torch.randperm(cls_feats.shape[1])[:self.min_views]
                sampled_feats.append(cls_feats[:, rand_idx])
                sampled_labels.append(cls.expand(self.min_views))

        return torch.cat(sampled_feats, dim=1).permute(1, 0), torch.cat(sampled_labels)

    def _compute_infonce(self, feats, labels):
        """核心InfoNCE损失计算"""
        # 特征归一化 (L2归一化)
        feats = F.normalize(feats, p=2, dim=1)  # (N, C)

        # 计算相似度矩阵
        sim_matrix = torch.mm(feats, feats.T) / self.temperature  # (N, N)

        # 构建正负样本掩码
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # (N, N)
        diag_mask = ~torch.eye(len(labels), dtype=torch.bool, device=feats.device)
        pos_mask = label_matrix & diag_mask  # 排除自身

        # 计算InfoNCE损失
        max_val = sim_matrix.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim_matrix - max_val)  # 数值稳定

        pos_sum = (exp_sim * pos_mask).sum(dim=1)  # 正样本相似度和
        neg_sum = (exp_sim * ~label_matrix).sum(dim=1)  # 负样本相似度和

        # 最终损失计算
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + self.eps))
        return loss.mean()


class MultiScaleContrastiveLoss(nn.Module):
    """
    处理多尺度特征的对比损失计算
    对每个尺度的特征和标签分别计算SimpleContrastiveLoss后求平均
    """

    def __init__(self, temperature=0.5, min_views=2):
        super().__init__()
        self.base_loss = SimpleContrastiveLoss(temperature, min_views)

    def forward(self, features_list, labels_list):
        """
        :param features_list: 多尺度特征列表 (L, B, C, H_l, W_l)
        :param labels_list: 多尺度标签列表 (L, B, 1, H_l, W_l)
        :return: 平均对比损失值
        """
        total_loss = 0.0
        valid_scales = 0

        for feats, lbls in zip(features_list, labels_list):
            # 调整标签张量维度 (B, 1, H, W) -> (B, H, W)
            lbls_adjusted = lbls.squeeze(1)

            # 确保标签与特征空间尺寸匹配
            if lbls_adjusted.shape[1:] != feats.shape[2:]:
                scale_factor = feats.shape[2] / lbls_adjusted.shape[1]
                lbls_ds = F.interpolate(
                    lbls_adjusted.float().unsqueeze(1),
                    scale_factor=scale_factor,
                    mode='nearest'
                ).squeeze(1).long()
            else:
                lbls_ds = lbls_adjusted

            # 计算当前尺度的对比损失
            loss = self.base_loss(feats, lbls_ds)
            total_loss += loss
            valid_scales += 1

        return total_loss / valid_scales if valid_scales > 0 else torch.tensor(0.0)