import torch
from torch import nn
from torch.nn import functional as F
from models.model.modules.backbone import Residual
from models.model.modules.naive_init import naive_init_module

class InstanceEmbedding(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding, self).__init__()
        self.neck = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms)
        naive_init_module(self.me)
        naive_init_module(self.neck)

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)


# 2D lane detection auxiliary head
class LaneHeadResidual_Instance(nn.Module):
    def __init__(self, output_size, input_channel=256):
        """

        :param output_size:
        :param input_channel:
        """
        super(LaneHeadResidual_Instance, self).__init__()

        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 60x 24
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(scale_factor=2),  # 120 x 48
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 32, 1),
            ),

            nn.Upsample(size=output_size),  # 300 x 120
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                )
            ),
        )

        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up)

    def forward(self, bev_x):
        bev_feat = self.bev_up(bev_x)
        return self.head(bev_feat)


# Depth auxiliary head
class DepthHeadUNet(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, base_channels=256, output_raw=True):
        super(DepthHeadUNet, self).__init__()

        self.output_raw = output_raw  # Flag to control output

        # Decoder layers
        self.up1 = self.up_block(in_channels, base_channels)  # From layer4 to layer3
        self.conv1 = self.conv_block(base_channels * 2, base_channels)

        self.up2 = self.up_block(base_channels, base_channels // 2)  # From layer3 to layer2
        self.conv2 = self.conv_block(base_channels, base_channels // 2)

        self.up3 = self.up_block(base_channels // 2, base_channels // 4)  # From layer2 to layer1
        self.conv3 = self.conv_block(base_channels // 2, base_channels // 4)

        self.up4 = self.up_block(base_channels // 4, base_channels // 8)  # From layer1 to after maxpool
        self.conv4 = self.conv_block(base_channels // 8 + 64, base_channels // 8)  # +64 channels from after maxpool

        # Final upsampling to reach (576, 1024)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)  # From (B, 32, 288, 512) to (B, 32, 576, 1024)
        self.final_conv = nn.Conv2d(base_channels // 8, out_channels, kernel_size=1)
        naive_init_module(self.final_conv)

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skips):
        """
        x: Input feature map from backbone (B, 512, H/32, W/32)
        skips: List of feature maps from backbone for skip connections
               [after maxpool, layer1, layer2, layer3, layer4]
        """
        # Decoder step 1: Upsample layer4 to layer3's spatial size
        d1 = self.up1(x)  # (B, 256, H/16, W/16)
        s1 = skips[3]  # Corresponding skip from layer3 (B, 256, H/16, W/16)
        d1 = torch.cat([d1, s1], dim=1)  # (B, 512, H/16, W/16)
        d1 = self.conv1(d1)  # (B, 256, H/16, W/16)

        # Decoder step 2: Upsample to layer2's spatial size
        d2 = self.up2(d1)  # (B, 128, H/8, W/8)
        s2 = skips[2]  # Corresponding skip from layer2 (B, 128, H/8, W/8)
        d2 = torch.cat([d2, s2], dim=1)  # (B, 256, H/8, W/8)
        d2 = self.conv2(d2)  # (B, 128, H/8, W/8)

        # Decoder step 3: Upsample to layer1's spatial size
        d3 = self.up3(d2)  # (B, 64, H/4, W/4)
        s3 = skips[1]  # Corresponding skip from layer1 (B, 64, H/4, W/4)
        d3 = torch.cat([d3, s3], dim=1)  # (B, 128, H/4, W/4)
        d3 = self.conv3(d3)  # (B, 64, H/4, W/4)

        # Decoder step 4: Upsample to after maxpool's spatial size
        d4 = self.up4(d3)  # (B, 32, H/2, W/2)
        s4 = skips[0]  # Corresponding skip from after maxpool (B, 64, H/4, W/4)
        # To match spatial dimensions, upsample s4
        s4 = F.interpolate(s4, size=d4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, s4], dim=1)  # (B, 96, H/2, W/2)
        d4 = self.conv4(d4)  # (B, 32, H/2, W/2)

        # Final upsampling
        out = self.final_up(d4)  # (B, 32, H, W)
        logits = self.final_conv(out)  # (B, 1, H, W)

        if self.output_raw:
            depth_logit = logits  # Raw logits for distillation
        else:
            depth_map = torch.sigmoid(logits)  # Normalized depth map
            depth_logit = depth_map

        return depth_logit  # Return logits or depth_map based on flag


