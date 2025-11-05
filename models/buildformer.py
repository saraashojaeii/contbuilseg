"""
BuildFormer model for building segmentation.
Based on "Building Extraction with Vision Transformer" (Wang et al., IEEE TGRS 2022)
https://github.com/WangLibo1995/BuildFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class RPE(nn.Module):
    """Relative Position Encoding"""
    def __init__(self, dim):
        super().__init__()
        self.rpe_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.rpe_norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        return x + self.rpe_norm(self.rpe_conv(x))


class Stem(nn.Module):
    def __init__(self, img_dim=3, out_dim=64, rpe=True):
        super(Stem, self).__init__()
        self.conv1 = ConvBNAct(img_dim, out_dim//2, kernel_size=3, stride=2, inplace=True)
        self.conv2 = ConvBNAct(out_dim//2, out_dim, kernel_size=3, stride=2, inplace=True)
        self.rpe = rpe
        if self.rpe:
            self.proj_rpe = RPE(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.rpe:
            x = self.proj_rpe(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, 
                 norm_layer=nn.BatchNorm2d, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNAct(in_features, hidden_features, kernel_size=1)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features),
            norm_layer(hidden_features),
            act_layer()
        )
        self.fc3 = ConvBN(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x


class LWMSA(nn.Module):
    """Lightweight Multi-head Self-Attention"""
    def __init__(self, dim=16, num_heads=8, window_size=16, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.eps = 1e-6
        self.ws = window_size
        self.qkv = Conv(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.proj = ConvBN(dim, dim, kernel_size=1)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps))
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps))
        return x

    def l2_norm(self, x):
        return torch.einsum("bhcn, bhn->bhcn", x, 1 / torch.norm(x, p=2, dim=-2))

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        hh, ww = Hp//self.ws, Wp//self.ws
        
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h d (ws1 ws2)',
                            b=B, h=self.num_heads, d=C//self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)
        
        q = self.l2_norm(q).permute(0, 1, 3, 2)
        k = self.l2_norm(k)
        
        tailor_sum = 1 / (self.ws * self.ws + torch.einsum("bhnc, bhc->bhn", q, torch.sum(k, dim=-1) + self.eps))
        attn = torch.einsum('bhmn, bhcn->bhmc', k, v)
        attn = torch.einsum("bhnm, bhmc->bhcn", q, attn)
        
        v = torch.einsum("bhcn->bhc", v).unsqueeze(-1)
        v = v.expand(B*hh*ww, self.num_heads, C//self.num_heads, self.ws * self.ws)
        attn = attn + v
        attn = torch.einsum("bhcn, bhn->bhcn", attn, tailor_sum)
        attn = rearrange(attn, '(b hh ww) h d (ws1 ws2) -> b (h d) (hh ws1) (ww ws2)',
                         b=B, h=self.num_heads, d=C // self.num_heads, ws1=self.ws, ws2=self.ws,
                         hh=Hp // self.ws, ww=Wp // self.ws)
        attn = attn[:, :, :H, :W]
        return attn


class Block(nn.Module):
    def __init__(self, dim=16, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=16):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.ws = window_size
        self.attn = LWMSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, 
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.BatchNorm2d, rpe=True):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, out_dim, 2, 2, 0, bias=False)
        self.rpe = rpe
        if self.rpe:
            self.proj_rpe = RPE(out_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        if self.rpe:
            x = self.proj_rpe(x)
        return x


class StageModule(nn.Module):
    def __init__(self, num_layers=2, in_dim=96, out_dim=96, num_heads=8, mlp_ratio=4., qkv_bias=False, use_pm=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=-1):
        super().__init__()
        self.use_pm = use_pm
        if self.use_pm:
            self.patch_partition = PatchMerging(in_dim, out_dim)

        self.layers = nn.ModuleList([])
        for idx in range(num_layers):
            self.layers.append(Block(dim=out_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, drop=drop,
                                     drop_path=drop_path, act_layer=act_layer, window_size=window_size,
                                     norm_layer=norm_layer))

    def forward(self, x):
        if self.use_pm:
            x = self.patch_partition(x)
        for block in self.layers:
            x = block(x)
        return x


class BuildFormerBackbone(nn.Module):
    """BuildFormer Encoder with Vision Transformer blocks"""
    def __init__(self, img_dim=3, mlp_ratio=4., window_sizes=[16, 16, 16, 16],
                 layers=[2, 2, 2, 2], num_heads=[4, 8, 16, 32], dims=[64, 128, 256, 512],
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3):
        super().__init__()
        self.stem = Stem(img_dim=img_dim, out_dim=dims[0], rpe=True)
        self.encoder_channels = dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]
        self.stage1 = StageModule(layers[0], dims[0], dims[0], num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  use_pm=False, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[0], window_size=window_sizes[0])
        self.stage2 = StageModule(layers[1], dims[0], dims[1], num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  use_pm=True, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[1], window_size=window_sizes[1])
        self.stage3 = StageModule(layers[2], dims[1], dims[2], num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  use_pm=True, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[2], window_size=window_sizes[2])
        self.stage4 = StageModule(layers[3], dims[2], dims[3], num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  use_pm=True, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr[3], window_size=window_sizes[3])

    def forward(self, x):
        features = []
        x = self.stem(x)
        x = self.stage1(x)
        features.append(x)
        x = self.stage2(x)
        features.append(x)
        x = self.stage3(x)
        features.append(x)
        x = self.stage4(x)
        features.append(x)
        return features


class DetailPath(nn.Module):
    """Detail path for capturing fine-grained features"""
    def __init__(self, embed_dim=64):
        super().__init__()
        dim1 = embed_dim // 4
        dim2 = embed_dim // 2
        self.dp1 = nn.Sequential(ConvBNAct(3, dim1, stride=2, inplace=False),
                                 ConvBNAct(dim1, dim1, stride=1, inplace=False))
        self.dp2 = nn.Sequential(ConvBNAct(dim1, dim2, stride=2, inplace=False),
                                 ConvBNAct(dim2, dim2, stride=1, inplace=False))
        self.dp3 = nn.Sequential(ConvBNAct(dim2, embed_dim, stride=1, inplace=False),
                                 ConvBNAct(embed_dim, embed_dim, stride=1, inplace=False))

    def forward(self, x):
        feats = self.dp1(x)
        feats = self.dp2(feats)
        feats = self.dp3(feats)
        return feats


class FPN(nn.Module):
    """Feature Pyramid Network decoder"""
    def __init__(self, encoder_channels=(64, 128, 256, 512), decoder_channels=256):
        super().__init__()
        self.pre_conv0 = Conv(encoder_channels[0], decoder_channels, kernel_size=1)
        self.pre_conv1 = Conv(encoder_channels[1], decoder_channels, kernel_size=1)
        self.pre_conv2 = Conv(encoder_channels[2], decoder_channels, kernel_size=1)
        self.pre_conv3 = Conv(encoder_channels[3], decoder_channels, kernel_size=1)

        self.post_conv3 = nn.Sequential(ConvBNAct(decoder_channels, decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNAct(decoder_channels, decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNAct(decoder_channels, decoder_channels))
        self.post_conv2 = nn.Sequential(ConvBNAct(decoder_channels, decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNAct(decoder_channels, decoder_channels))
        self.post_conv1 = ConvBNAct(decoder_channels, decoder_channels)
        self.post_conv0 = ConvBNAct(decoder_channels, decoder_channels)

    def upsample_add(self, up, x):
        up = F.interpolate(up, x.size()[-2:], mode='nearest')
        up = up + x
        return up

    def forward(self, x0, x1, x2, x3):
        x3 = self.pre_conv3(x3)
        x2 = self.pre_conv2(x2)
        x1 = self.pre_conv1(x1)
        x0 = self.pre_conv0(x0)

        x2 = self.upsample_add(x3, x2)
        x1 = self.upsample_add(x2, x1)
        x0 = self.upsample_add(x1, x0)

        x3 = self.post_conv3(x3)
        x3 = F.interpolate(x3, x0.size()[-2:], mode='bilinear', align_corners=False)
        x2 = self.post_conv2(x2)
        x2 = F.interpolate(x2, x0.size()[-2:], mode='bilinear', align_corners=False)
        x1 = self.post_conv1(x1)
        x1 = F.interpolate(x1, x0.size()[-2:], mode='bilinear', align_corners=False)
        x0 = self.post_conv0(x0)

        x0 = x3 + x2 + x1 + x0
        return x0


class DualHeadBuildFormer(nn.Module):
    """
    BuildFormer with dual heads for mask and contour prediction.
    Based on "Building Extraction with Vision Transformer" (Wang et al., IEEE TGRS 2022)
    """
    def __init__(self, decoder_channels=256, dims=[64, 128, 256, 512],
                 window_sizes=[16, 16, 16, 16], num_labels=1):
        super().__init__()
        self.backbone = BuildFormerBackbone(layers=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
                                            dims=dims, window_sizes=window_sizes)
        encoder_channels = self.backbone.encoder_channels
        self.dp = DetailPath(embed_dim=decoder_channels)
        self.fpn = FPN(encoder_channels, decoder_channels)
        
        # Mask head
        self.mask_head = nn.Sequential(
            ConvBNAct(decoder_channels, encoder_channels[0]),
            nn.Dropout(0.1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            Conv(encoder_channels[0], num_labels, kernel_size=1)
        )
        
        # Contour head
        self.contour_head = nn.Sequential(
            ConvBNAct(decoder_channels, encoder_channels[0]),
            nn.Dropout(0.1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            Conv(encoder_channels[0], 1, kernel_size=1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        sz = x.size()[-2:]
        
        # Pad input to be divisible by 64 for BuildFormer's hierarchical architecture
        # Total downsampling: stem(4x) + 3 stages(2x each) = 32x, plus window_size=16
        h, w = x.shape[-2:]
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            # Debug: uncomment to see padding
            # print(f"Padded from {sz} to {x.shape[-2:]}")
        
        dp = self.dp(x)
        x0, x1, x2, x3 = self.backbone(x)
        x = self.fpn(x0, x1, x2, x3)
        x = x + dp
        
        mask_logits = self.mask_head(x)
        mask_logits = F.interpolate(mask_logits, sz, mode='bilinear', align_corners=False)
        
        contour_map = self.contour_head(x)
        contour_map = F.interpolate(contour_map, sz, mode='bilinear', align_corners=False)
        
        return mask_logits, contour_map


class BuildFormerModel:
    """Wrapper for BuildFormer model"""
    def __init__(self, decoder_channels=256, dims=[64, 128, 256, 512],
                 window_sizes=[16, 16, 16, 16], num_labels=1):
        self.model = DualHeadBuildFormer(
            decoder_channels=decoder_channels,
            dims=dims,
            window_sizes=window_sizes,
            num_labels=num_labels
        )
        # BuildFormer uses standard image normalization
        self.image_processor = None
    
    def get_model(self):
        return self.model
    
    def get_image_processor(self):
        return self.image_processor


def get_buildformer_model(decoder_channels=256, dims=[64, 128, 256, 512],
                          window_sizes=[16, 16, 16, 16], num_labels=1, **kwargs):
    """
    Factory function to create a BuildFormer model.
    
    Args:
        decoder_channels: Number of channels in decoder (default: 256)
        dims: Channel dimensions for each stage (default: [64, 128, 256, 512])
        window_sizes: Window sizes for attention (default: [16, 16, 16, 16])
        num_labels: Number of output classes (default: 1 for binary)
        
    Returns:
        BuildFormerModel wrapper
    """
    return BuildFormerModel(
        decoder_channels=decoder_channels,
        dims=dims,
        window_sizes=window_sizes,
        num_labels=num_labels
    )
