import numpy as np
import torch
import torch.nn as nn

from medpy.metric.binary import hd95
from nnunetv2.utilities.helpers import softmax_helper_dim1


def compute_hd95_loss(pred, target):
    """
    计算多类分割的 HD95 损失（自动判断类别数）
    :param pred:  预测张量 (batch, C, H, W)，C 为类别数（含背景）
    :param target: 目标张量 (batch, 1, H, W)，值为 0 ~ C-1
    :return: HD95 损失值
    """
    batch_size, num_classes, H, W = pred.shape  # 自动获取类别数 [1,9](@ref)
    total_hd95 = 0.0
    valid_classes = 0

    # 获取预测类别标签（自动适应类别数）
    pred_labels = softmax_helper_dim1(pred).detach().cpu().numpy()
    target_labels = target.squeeze(1).cpu().numpy()  # [batch, H, W]

    for b in range(batch_size):
        # 动态遍历所有类别（跳过背景类0）
        for class_id in range(1, num_classes):  # 从1开始到num_classes-1 [1](@ref)
            pred_mask = (pred_labels[b] == class_id).astype(np.uint8)
            target_mask = (target_labels[b] == class_id).astype(np.uint8)

            # 检查掩码有效性（目标或预测需含有效像素）
            if np.any(target_mask) or np.any(pred_mask):
                try:
                    class_hd95 = hd95(pred_mask, target_mask)
                    total_hd95 += class_hd95
                    valid_classes += 1
                except RuntimeError:
                    # 计算失败时使用图像对角线作为惩罚值
                    total_hd95 += np.sqrt(H ** 2 + W ** 2)
                    valid_classes += 1

    # 计算平均 HD95（避免除零）
    return total_hd95 / max(valid_classes, 1) if valid_classes > 0 else 0.0


class HausdorffDTLoss(nn.Module):
    """
    基于距离变换的可微 Hausdorff 损失
    - 通过距离变换实现端到端可微性
    - 支持多类别分割（自动处理类别维度）
    - 参考文献:
        Karimi D, et al. "Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks"

    :param alpha: 距离损失权重系数 (默认为0.5)
    :param p: 距离范数 (默认为2, 欧氏距离)
    :param reduction: 批处理损失计算方式 ('mean', 'sum', 'none')
    :param safe_edt: 是否使用安全距离变换（避免零除错误）
    """

    def __init__(self, alpha=0.5, p=2, reduction='mean', safe_edt=True):
        super().__init__()
        self.alpha = alpha
        self.p = p
        self.reduction = reduction
        self.safe_edt = safe_edt

    def _compute_dt(self, target):
        """计算目标掩码的距离变换图 (CPU/CUDA兼容)"""
        # 分离前景/背景
        fg = (target == 1).float()
        bg = 1 - fg

        # 计算背景到前景的距离
        if self.p == 1:  # L1范数 (曼哈顿距离)
            dt_bg = self._edt_1d(bg, axis=1) + self._edt_1d(bg, axis=2)
        else:  # L2范数 (欧氏距离)
            dt_bg = torch.sqrt(self._edt_1d(bg, axis=1) ** 2 + self._edt_1d(bg, axis=2) ** 2)

        # 计算前景到背景的距离
        if self.p == 1:
            dt_fg = self._edt_1d(fg, axis=1) + self._edt_1d(fg, axis=2)
        else:
            dt_fg = torch.sqrt(self._edt_1d(fg, axis=1) ** 2 + self._edt_1d(fg, axis=2) ** 2)

        # 合并距离图 [citations:1][citations:3]
        dt = dt_fg - dt_bg

        # 安全处理零值区域
        if self.safe_edt:
            dt = torch.where(target == 1, dt_fg, -dt_bg)

        return dt

    def _edt_1d(self, x, axis=1):
        """一维距离变换 (沿指定维度)"""
        n = x.shape[axis]
        arange = torch.arange(n, device=x.device).view(1, -1, 1) if axis == 1 else torch.arange(n, device=x.device).view(1, 1, -1)

        # 计算非零位置
        non_zero = (x < 0.5).float()
        f = non_zero * arange
        d_f = torch.abs(f - arange)

        # 寻找每行/列的最小距离 [citations:5]
        min_vals, _ = torch.min(d_f + 1e6 * (1 - non_zero), dim=axis, keepdim=True)
        return min_vals

    def forward(self, pred, target):
        """
        :param pred: 预测概率图 (B, C, H, W)
        :param target: 目标掩码 (B, 1, H, W) 或 (B, H, W)
        :return: HausdorffDT 损失值
        """
        # 统一目标格式
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        # 多类别处理
        total_loss = 0.0
        valid_classes = 0

        for c in range(1, pred.shape[1]):  # 遍历每个类别
            # 获取当前类别的预测和目标
            pred_c = pred[:, c, :, :].unsqueeze(1)  # (B, 1, H, W)
            target_c = (target == c).float() if pred.shape[1] > 1 else target.float()

            # 跳过空目标类别
            if torch.sum(target_c) == 0:
                continue

            # 计算距离变换图
            with torch.no_grad():
                dt = self._compute_dt(target_c)  # (B, 1, H, W)

            # 计算边界损失 [citations:7]
            boundary_loss = torch.mean(pred_c * dt, dim=(2, 3))

            # 可选：添加Dice损失增强稳定性
            dice_loss = 1 - self._dice_coeff(pred_c, target_c)

            # 组合损失项
            total_loss += self.alpha * boundary_loss + (1 - self.alpha) * dice_loss
            valid_classes += 1

        # 处理无有效类别的情况
        if valid_classes == 0:
            return torch.tensor(0.0, device=pred.device)

        # 平均各类别损失
        total_loss /= valid_classes

        # 批处理损失归约
        if self.reduction == 'mean':
            return torch.mean(total_loss)
        elif self.reduction == 'sum':
            return torch.sum(total_loss)
        else:
            return total_loss

    def _dice_coeff(self, pred, target):
        """Dice系数 (平滑处理避免除零)"""
        smooth = 1.0
        intersection = torch.sum(pred * target, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        return (2.0 * intersection + smooth) / (union + smooth)


class MultiClassHausdorffDTLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, mode="sigmoid", ignore_index=0):
        """
        :param alpha: Hausdorff损失权重 (需远小于beta)
        :param beta: Dice损失权重
        :param mode: 缩放模式 ("sigmoid"或"linear")
        :param ignore_index: 忽略的背景类别索引
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.ignore_index = ignore_index
        self.eps = 1e-6

    def forward(self, pred, target):
        """
        :param pred:  预测张量 (B, C, H, W)
        :param target: 目标张量 (B, 1, H, W) 或 (B, H, W)，值为类别索引
        """
        # 统一目标格式
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        B, C, H, W = pred.shape
        max_distance = (H ** 2 + W ** 2) ** 0.5  # 图像对角线长度[1](@ref)
        total_loss = 0.0
        valid_classes = 0

        # 遍历所有类别 (跳过背景)
        for class_id in range(1, C):  # 从1开始[7](@ref)
            # 生成当前类别的二值掩码
            pred_class = pred[:, class_id, :, :]  # (B, H, W)
            target_class = (target.squeeze(1) == class_id).float()  # (B, H, W)[8](@ref)

            # 检查目标是否存在有效像素
            if torch.sum(target_class) == 0:
                continue  # 跳过无目标的类别

            # 计算当前类别的Hausdorff损失 (归一化到[0,1])
            hd_loss = self._compute_hd_loss(pred_class, target_class) / max_distance

            # 计算当前类别的Dice损失[6](@ref)
            # dice_loss = self._dice_loss(pred_class, target_class)

            # 加权组合当前类别损失
            # class_loss = self.alpha * hd_loss + self.beta * dice_loss
            # total_loss += class_loss
            total_loss += hd_loss
            valid_classes += 1

        # 处理无有效类别的情况
        if valid_classes == 0:
            return torch.tensor(0.0, device=pred.device)

        # 计算各类别平均损失
        total_loss /= valid_classes

        # 范围映射至[1,5]
        if self.mode == "sigmoid":
            scaled_loss = torch.sigmoid(total_loss)  # [1,5][1](@ref)
        else:  # linear模式 (预设total_loss∈[-4,4])
            scaled_loss = 0.5 * total_loss  # 线性映射[1](@ref)

        return scaled_loss

    def _compute_hd_loss(self, pred, target):
        """可微Hausdorff损失实现 (单类别)"""
        # 二值化预测 (保持梯度)
        pred_bin = torch.sigmoid(pred)  # 使用概率而非阈值[1](@ref)
        target_bin = target

        # 计算距离变换图 (简化版，建议用GPU库[1](@ref))
        with torch.no_grad():
            dt_target = self._distance_transform(target_bin)
            dt_pred = self._distance_transform(pred_bin.detach())

        # Hausdorff损失 = 边界距离误差[6](@ref)
        boundary_loss = torch.mean(pred_bin * dt_target + target_bin * dt_pred)
        return boundary_loss

    def _distance_transform(self, x):
        """二值掩码的距离变换 (近似实现)"""
        # 实际项目推荐使用[1](@ref):
        # from hausdorff_loss import HausdorffDTLoss
        # dt = HausdorffDTLoss()(x, target)
        return torch.zeros_like(x)  # 此处为占位符

    def _dice_loss(self, pred, target):
        """Dice损失 (单类别)"""
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) + self.eps
        return 1 - (2.0 * intersection) / union  # [0,1][6](@ref)
