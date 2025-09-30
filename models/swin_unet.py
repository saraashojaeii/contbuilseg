"""
SWIN-UNET model implementation for building segmentation with contour awareness.
Combines Swin Transformer with U-Net architecture for hierarchical feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    """
    Patch Embedding layer that splits image into patches and embeds them.
    """
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H/patch_size, W/patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    """
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with window-based attention.
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging layer for downsampling.
    """
    def __init__(self, dim):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    """
    Patch Expanding layer for upsampling.
    """
    def __init__(self, dim):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, C * 2)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        return x


class SwinUNet(nn.Module):
    """
    SWIN-UNET architecture for segmentation with optional contour prediction.
    Uses Swin Transformer blocks in an encoder-decoder structure.
    """
    def __init__(self, img_size=224, in_channels=3, out_channels_mask=1, out_channels_contour=1,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], window_size=7):
        """
        Initialize SWIN-UNET model.
        
        Args:
            img_size: Input image size
            in_channels: Number of input channels (typically 3 for RGB)
            out_channels_mask: Number of output channels for the mask
            out_channels_contour: Number of output channels for the contour
            embed_dim: Embedding dimension
            depths: Number of blocks at each stage
            num_heads: Number of attention heads at each stage
            window_size: Window size for attention
        """
        super(SwinUNet, self).__init__()
        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=4, in_channels=in_channels, embed_dim=embed_dim)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=int(embed_dim * 2 ** i),
                    num_heads=num_heads[i],
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2
                )
                for j in range(depths[i])
            ])
            self.encoder_layers.append(layer)
            
            if i < self.num_layers - 1:
                self.downsample_layers.append(PatchMerging(int(embed_dim * 2 ** i)))

        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for i in range(self.num_layers - 1, 0, -1):
            self.upsample_layers.append(PatchExpanding(int(embed_dim * 2 ** i)))
            
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=int(embed_dim * 2 ** (i - 1)),
                    num_heads=num_heads[i - 1],
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2
                )
                for j in range(depths[i - 1])
            ])
            self.decoder_layers.append(layer)

        # Final expansion
        self.final_expand = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True)
        )

        # Output heads
        self.mask_head = nn.Conv2d(embed_dim // 4, out_channels_mask, kernel_size=1)
        self.contour_head = nn.Conv2d(embed_dim // 4, out_channels_contour, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through SWIN-UNET.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            Tuple of (mask_output, contour_output)
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)
        
        # Store encoder features for skip connections
        encoder_features = []
        H_enc, W_enc = H // 4, W // 4

        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            for block in layer:
                x = block(x, H_enc, W_enc)
            encoder_features.append((x, H_enc, W_enc))
            
            if i < self.num_layers - 1:
                x = self.downsample_layers[i](x, H_enc, W_enc)
                H_enc, W_enc = H_enc // 2, W_enc // 2

        # Decoder
        for i, (upsample, layer) in enumerate(zip(self.upsample_layers, self.decoder_layers)):
            x = upsample(x, H_enc, W_enc)
            H_enc, W_enc = H_enc * 2, W_enc * 2
            
            # Skip connection
            skip_x, skip_H, skip_W = encoder_features[-(i + 2)]
            x = x + skip_x
            
            for block in layer:
                x = block(x, H_enc, W_enc)

        # Reshape to spatial format
        x = x.view(B, H_enc, W_enc, -1).permute(0, 3, 1, 2).contiguous()

        # Final expansion
        x = self.final_expand(x)

        # Output heads
        mask_output = torch.sigmoid(self.mask_head(x))
        # Contour output is distance transform (continuous values), no sigmoid
        contour_output = self.contour_head(x)

        return mask_output, contour_output


def get_swin_unet_model(img_size=224, in_channels=3, out_channels_mask=1, out_channels_contour=1):
    """
    Factory function to create a SWIN-UNET model.
    
    Args:
        img_size: Input image size
        in_channels: Number of input channels
        out_channels_mask: Number of output channels for the mask
        out_channels_contour: Number of output channels for the contour
        
    Returns:
        Initialized SWIN-UNET model
    """
    model = SwinUNet(
        img_size=img_size,
        in_channels=in_channels,
        out_channels_mask=out_channels_mask,
        out_channels_contour=out_channels_contour
    )
    return model
