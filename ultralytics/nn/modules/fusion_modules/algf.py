#Adaptive Local-Global Fusion Module (ALGF)
'''
让局部与全局特征在语义和空间上深度交互

动态加权，让网络自适应选择该信局部还是信全局

保留原始模态结构，避免信息坍塌
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .local import LLModule
from .swin_global_module import SwinGGMoudle

# ============ Adaptive Fusion Gate ============
class AdaptiveFusionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)  # 权重归一化
        )

    def forward(self, local_feat, global_feat):
        combined = torch.cat([local_feat, global_feat], dim=1)
        weights = self.gate(combined)  # [B, 2, H, W]
        w_local = weights[:, 0:1, :, :]
        w_global = weights[:, 1:2, :, :]
        return w_local * local_feat + w_global * global_feat

# ============ 主网络：PCFusionNet ============
class PCFusionNet(nn.Module):
    def __init__(self, in_channels=64, embed_dim=128, num_heads=4, depth=3, img_size=256):
        super().__init__()
        self.local_branch = LLModule(in_channels)
        self.global_branch = SwinGGMoudle(in_channels, embed_dim, num_heads, depth, img_size)
        self.fusion_gate = AdaptiveFusionGate(in_channels)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )

    def forward(self, pet, ct=None):
        # Accept inputs as (pet, ct) or [pet, ct] from YOLO's multi-from mechanism
        if ct is None:
            if isinstance(pet, (list, tuple)) and len(pet) == 2:
                pet, ct = pet
            else:
                raise TypeError("PCFusionNet expects two inputs (pet, ct) or a list/tuple of two tensors.")

        local_fused = self.local_branch(pet, ct)
        global_fused = self.global_branch(pet, ct)
        adaptive_fused = self.fusion_gate(local_fused, global_fused)
        out = local_fused + self.refine(adaptive_fused)  # 保留局部为主，全局为辅增强
        return out
