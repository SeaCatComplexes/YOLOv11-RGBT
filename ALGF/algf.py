#Adaptive Local-Global Fusion Module (ALGF)
'''
让局部与全局特征在语义和空间上深度交互

动态加权，让网络自适应选择该信局部还是信全局

保留原始模态结构，避免信息坍塌
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile 
# from .global_global import GGModule
# from .local import LLModule
# from .swin_global_module import SwinGGMoudle
from global_global import GGModule
from local import LLModule
from swin_global_module import SwinGGMoudle

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

    def forward(self, pet, ct):
        local_fused = self.local_branch(pet, ct)
        global_fused = self.global_branch(pet, ct)
        adaptive_fused = self.fusion_gate(local_fused, global_fused)
        out = local_fused + self.refine(adaptive_fused)  # 保留局部为主，全局为辅增强
        return out

# ============ 测试代码 ============
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PCFusionNet(
        in_channels=1,
        embed_dim=128,
        num_heads=2,
        depth=2,
        img_size=128
    ).to(device)

    pet = torch.randn(1, 1, 128, 128).to(device)
    ct = torch.randn(1, 1, 128, 128).to(device)

    # 计算GFLOPs和参数量
    flops, params = profile(
        model,
        inputs=(pet, ct),  # 传入模型的输入（元组形式）
        verbose=False  # 关闭详细日志
    )

    # 转换单位（FLOPs -> GFLOPs，params -> 百万参数）
    gflops = flops / 1e9  # 1 GFLOPs = 10^9 FLOPs
    m_params = params / 1e6  # 1M = 10^6 参数

    # 测试模型输出
    output = model(pet, ct)
    print("✅ PCFusionNet 测试通过！")
    print(f"输入 PET shape: {pet.shape}")
    print(f"输入 CT shape: {ct.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"总参数量: {m_params:.2f} M")  # 以百万为单位
    print(f"GFLOPs: {gflops:.4f}")  # 浮点运算量（每秒千兆次）