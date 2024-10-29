from torch import nn
from .naive_init import naive_init_module
from .backbone import Residual
from .tcn import SimpleTCN


class InstanceEmbedding_offset_y_z(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding_offset_y_z, self).__init__()
        self.neck_new = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_offset_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_z = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        naive_init_module(self.m_z)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(feat), self.m_z(feat)



class LaneHeadResidual_Instance_with_offset_z(nn.Module):
    def __init__(self, bev_shape, input_channel=512, temporal=False, temporal_length=4):

        super(LaneHeadResidual_Instance_with_offset_z, self).__init__()

        self.temporal = temporal
        self.temporal_length = temporal_length if temporal else 1  # 如果不使用时序，则时序长度为1

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上采样
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
            nn.Upsample(size=bev_shape, mode='bilinear', align_corners=True),  # 调整到目标大小
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                ),
                downsample=nn.Conv2d(128, 64, 1),
            ),
        )

        # 如果使用时序模块，添加 SimpleTCN
        if self.temporal and self.temporal_length > 1:
            self.tcn = SimpleTCN(
                num_inputs=64,
                num_channels=[64, 64],  # 减少层数和通道数以保持轻量
                kernel_size=3,
                dropout=0.2
            )

        # 定义高度预测头
        self.head = InstanceEmbedding_offset_y_z(64, 2)

        # 初始化模块
        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)
        if self.temporal and self.temporal_length > 1:
            naive_init_module(self.tcn)

    def forward(self, bev_x):
        """
        车道检测头的前向传播。

        Args:
            bev_x (torch.Tensor): BEV 特征图，形状为 (B, C, H, W)。

        Returns:
            tuple: 包含置信度 (ms)、嵌入 (me)、偏移_y 和 高度 (z) 的预测。
        """
        bev_feat = self.bev_up_new(bev_x)  # (B, C_new, H, W)

        if self.temporal and self.temporal_length > 1:
            B, C_new, H, W = bev_feat.shape
            B_seq = B // self.temporal_length
            bev_feat = bev_feat.view(B_seq, self.temporal_length, C_new, H, W)  # (B_seq, T, C_new, H, W)

            # 转换为 (B_seq * H * W, C_new, T)
            bev_feat = bev_feat.permute(0, 3, 4, 1, 2).contiguous()  # (B_seq, H, W, T, C_new)
            bev_feat = bev_feat.view(B_seq * H * W, self.temporal_length, C_new)  # (B_seq * H * W, T, C_new)
            bev_feat = bev_feat.permute(0, 2, 1).contiguous()  # (B_seq * H * W, C_new, T)

            # 通过 SimpleTCN
            bev_feat = self.tcn(bev_feat)  # (B_seq * H * W, C_new, T)

            # 转换回 (B_seq, H, W, C_new, T)
            bev_feat = bev_feat.permute(0, 2, 1).contiguous()  # (B_seq * H * W, T, C_new)
            bev_feat = bev_feat.view(B_seq, H, W, self.temporal_length, C_new)  # (B_seq, H, W, T, C_new)

            # 转换为 (B_seq * T, C_new, H, W)
            bev_feat = bev_feat.permute(0, 4, 3, 1, 2).contiguous()  # (B_seq, C_new, T, H, W)
            bev_feat = bev_feat.view(B_seq * self.temporal_length, C_new, H, W)  # (B_seq * T, C_new, H, W)

        # 通过预测头
        return self.head(bev_feat)