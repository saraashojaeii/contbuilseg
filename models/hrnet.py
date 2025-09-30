"""
HRNet model implementation for building segmentation with contour awareness.
High-Resolution Network that maintains high-resolution representations throughout the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic residual block for HRNet.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    """
    High-Resolution Module that processes multi-resolution features in parallel.
    """
    def __init__(self, num_branches, num_blocks, num_channels, block=BasicBlock):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels

        self.branches = nn.ModuleList()
        for i in range(num_branches):
            self.branches.append(self._make_branch(i, block, num_blocks, num_channels))

        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_branch(self, branch_index, block, num_blocks, num_channels):
        """Create a branch of the module."""
        layers = []
        for i in range(num_blocks):
            layers.append(block(num_channels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        """Create fusion layers to exchange information between branches."""
        if self.num_branches == 1:
            return None

        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    # Upsample
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_channels[j], self.num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(self.num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # Downsample
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(self.num_channels[j], self.num_channels[i], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(self.num_channels[i])
                            ))
                        else:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(self.num_channels[j], self.num_channels[j], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(self.num_channels[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward pass through the high-resolution module."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # Process each branch
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        # Fuse branches
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNet(nn.Module):
    """
    High-Resolution Network for segmentation with optional contour prediction.
    Maintains high-resolution representations throughout the network.
    """
    def __init__(self, in_channels=3, out_channels_mask=1, out_channels_contour=1):
        """
        Initialize HRNet model.
        
        Args:
            in_channels: Number of input channels (typically 3 for RGB)
            out_channels_mask: Number of output channels for the mask
            out_channels_contour: Number of output channels for the contour
        """
        super(HRNet, self).__init__()

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)

        # Transition 1: 1 -> 2 branches
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 2
        self.stage2 = HighResolutionModule(2, 4, [32, 64])

        # Transition 2: 2 -> 3 branches
        self.transition2 = nn.ModuleList([
            None,
            None,
            nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 3
        self.stage3 = HighResolutionModule(3, 4, [32, 64, 128])

        # Transition 3: 3 -> 4 branches
        self.transition3 = nn.ModuleList([
            None,
            None,
            None,
            nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 4
        self.stage4 = HighResolutionModule(4, 3, [32, 64, 128, 256])

        # Upsampling and fusion for final output
        self.final_layer = nn.Sequential(
            nn.Conv2d(32 + 64 + 128 + 256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Output heads
        self.mask_head = nn.Conv2d(256, out_channels_mask, kernel_size=1, stride=1, padding=0)
        self.contour_head = nn.Conv2d(256, out_channels_contour, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, in_channels, out_channels, blocks):
        """Create a layer with multiple blocks."""
        layers = []
        layers.append(block(in_channels, out_channels))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through HRNet.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            Tuple of (mask_output, contour_output)
        """
        input_size = x.shape[-2:]

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)

        # Transition 1
        x_list = []
        for i in range(2):
            x_list.append(self.transition1[i](x))

        # Stage 2
        x_list = self.stage2(x_list)

        # Transition 2
        x_list_new = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list_new.append(self.transition2[i](x_list[-1]))
            else:
                x_list_new.append(x_list[i])
        x_list = x_list_new

        # Stage 3
        x_list = self.stage3(x_list)

        # Transition 3
        x_list_new = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list_new.append(self.transition3[i](x_list[-1]))
            else:
                x_list_new.append(x_list[i])
        x_list = x_list_new

        # Stage 4
        x_list = self.stage4(x_list)

        # Upsample all branches to the highest resolution
        x0_h, x0_w = x_list[0].size(2), x_list[0].size(3)
        x1 = F.interpolate(x_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=False)

        # Concatenate all features
        x = torch.cat([x_list[0], x1, x2, x3], dim=1)

        # Final layer
        x = self.final_layer(x)

        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        # Output heads
        mask_output = torch.sigmoid(self.mask_head(x))
        # Contour output is distance transform (continuous values), no sigmoid
        contour_output = self.contour_head(x)

        return mask_output, contour_output


def get_hrnet_model(in_channels=3, out_channels_mask=1, out_channels_contour=1):
    """
    Factory function to create an HRNet model.
    
    Args:
        in_channels: Number of input channels
        out_channels_mask: Number of output channels for the mask
        out_channels_contour: Number of output channels for the contour
        
    Returns:
        Initialized HRNet model
    """
    model = HRNet(
        in_channels=in_channels,
        out_channels_mask=out_channels_mask,
        out_channels_contour=out_channels_contour
    )
    return model
