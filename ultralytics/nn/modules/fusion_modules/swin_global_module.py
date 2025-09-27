# swin_global_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    """
    将特征图划分为不重叠的窗口
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将窗口还原为特征图
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size: int
        H, W: 原始高宽
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H // window_size * W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """窗口多头自注意力，带相对位置偏置"""
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # 相对位置偏置表 [ (2*Wh-1) * (2*Ww-1), nH ]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # 计算相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)

        # 初始化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        # x: [B*num_windows, N, C] where N = window_size * window_size
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, C//num_heads]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B_, num_heads, N, N]

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block: W-MSA + MLP"""
    def __init__(self, dim, num_heads, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        shortcut = x

        # 转成 [B, H, W, C] 便于窗口划分
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # 划分窗口
        x_windows = window_partition(x, self.window_size)  # [B*num_windows, win_h, win_w, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [B*N, win^2, C]

        # Window Attention
        attn_windows = self.attn(x_windows)  # [B*N, win^2, C]

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H, W, C]

        # LayerNorm + Residual (注意维度转换)
        x = x.permute(0, 3, 1, 2)  # 转回 [B, C, H, W] 用于 CNN 风格操作
        x = shortcut + self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # 残差连接

        # MLP + 残差
        shortcut2 = x
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x + self.norm2(self.mlp(x))
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = shortcut2 + x

        return x

class CrossModalityAttention(nn.Module):
    """跨模态注意力：用模态A的Query去查询模态B的Key/Value"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_query, x_kv):
        # x_query, x_kv: [B, N, C]
        B, N, C = x_query.shape

        q = self.q_proj(x_query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(x_kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, num_heads, N, C//h]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B, h, N, N]
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SwinGGMoudle(nn.Module):
    """
    Swin-based Global Guidance Module with Cross-Modality Interaction
    输入：双模态图像（如 PET + CT）
    输出：融合后的全局增强特征图
    """
    def __init__(self, in_channels, embed_dim, num_heads, depth, img_size=256, window_size=8, downsample_factor=2):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.window_size = window_size
        self.downsample_factor = max(1, int(downsample_factor))

        # 1x1 卷积投影到嵌入空间
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    # 2D 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, img_size, img_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Swin Transformer Blocks
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(dim=embed_dim, num_heads=num_heads, window_size=8)
            for _ in range(depth)
        ])

        # 跨模态注意力层
        self.cross_attn = CrossModalityAttention(embed_dim, num_heads)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 最终投影回原始通道数
        self.final_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    @staticmethod
    def _pad_to_multiple(x, multiple):
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        # F.pad: (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    @staticmethod
    def _unpad(x, pads):
        pad_h, pad_w = pads
        if pad_h:
            x = x[:, :, : x.shape[2] - pad_h, :]
        if pad_w:
            x = x[:, :, :, : x.shape[3] - pad_w]
        return x

    def forward(self, x1, x2):
        """
        Args:
            x1: [B, C_in, H, W] - 模态1 (e.g., PET)
            x2: [B, C_in, H, W] - 模态2 (e.g., CT)
        Returns:
            out: [B, C_in, H, W] - 融合增强后的特征图
        """
        # 投影到嵌入空间
        x1 = self.proj(x1)
        x2 = self.proj(x2)

        # 动态适配位置编码到当前特征图分辨率（例如构建 stride 时使用 256 输入 -> P3=32）
        _, _, H, W = x1.shape
        if self.pos_embed.shape[-2:] != (H, W):
            pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        else:
            pos = self.pos_embed

        # 添加位置编码
        x1 = x1 + pos  # [B, C, H, W]
        x2 = x2 + pos

        # 分别通过 Swin Blocks 提取全局特征
        for blk in self.swin_blocks:
            x1 = blk(x1)
            x2 = blk(x2)

        # 窗口化的跨模态注意（可选下采样 + pad/unpad + window_partition/reverse）
        B, C, H, W = x1.shape
        H0, W0 = H, W

        # 注意力前 2x 下采样（可选，AvgPool）
        if self.downsample_factor > 1:
            x1_d = F.avg_pool2d(x1, kernel_size=self.downsample_factor, stride=self.downsample_factor)
            x2_d = F.avg_pool2d(x2, kernel_size=self.downsample_factor, stride=self.downsample_factor)
        else:
            x1_d, x2_d = x1, x2

        # pad 到 window_size 的倍数
        win = self.window_size
        x1_p, pads = self._pad_to_multiple(x1_d, win)
        x2_p, _ = self._pad_to_multiple(x2_d, win)
        Hp, Wp = x1_p.shape[2], x1_p.shape[3]

        # [B,C,H,W] -> [B,H,W,C]
        A = x1_p.permute(0, 2, 3, 1).contiguous()
        Bm = x2_p.permute(0, 2, 3, 1).contiguous()

        # 划分窗口 -> [B*nW, win, win, C]
        wA = window_partition(A, win)
        wB = window_partition(Bm, win)
        # 展平 -> [B*nW, win^2, C]
        NA = wA.view(-1, win * win, C)
        NB = wB.view(-1, win * win, C)

        # 双向跨模态注意（窗口粒度）
        NA2 = self.cross_attn(NA, NB)
        NB2 = self.cross_attn(NB, NA)

        # 还原窗口 -> [B*nW, win, win, C]
        wA2 = NA2.view(-1, win, win, C)
        wB2 = NB2.view(-1, win, win, C)

        # 还原特征图 -> [B,Hp,Wp,C]
        A2 = window_reverse(wA2, win, Hp, Wp)
        B2 = window_reverse(wB2, win, Hp, Wp)

        # 回到 [B,C,Hp,Wp]
        x1_enhanced = A2.permute(0, 3, 1, 2).contiguous()
        x2_enhanced = B2.permute(0, 3, 1, 2).contiguous()

        # 去 pad
        x1_enhanced = self._unpad(x1_enhanced, pads)
        x2_enhanced = self._unpad(x2_enhanced, pads)

        # 上采样回 P3 分辨率
        if self.downsample_factor > 1:
            x1_enhanced = F.interpolate(x1_enhanced, size=(H0, W0), mode='bilinear', align_corners=False)
            x2_enhanced = F.interpolate(x2_enhanced, size=(H0, W0), mode='bilinear', align_corners=False)

        # 拼接 + 融合
        fused = torch.cat([x1_enhanced, x2_enhanced], dim=1)  # [B, 2C, H, W]
        fused = self.fusion(fused)  # [B, C, H, W]

        # 最终投影
        out = self.final_proj(fused)  # [B, in_channels, H, W]
        return out
