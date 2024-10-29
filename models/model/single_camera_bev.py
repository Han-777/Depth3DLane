# single_camera_bev.py

import torch
from torch import nn
import torch.nn.functional as F
from .modules.dpt import DepthAnythingV2
from .modules.naive_init import naive_init_module
from .modules.backbone import Residual, ResNet34_Backbone
from .modules.neck import FCTransform_
from .modules.outputHead import LaneHeadResidual_Instance_with_offset_z
from .modules.auxiliaryHead import LaneHeadResidual_Instance, DepthHeadUNet
from .modules.depthStudentBranch import StudentS32, StudentS64  # Import the new student models
from models.util.blocks import FeatureFusionBlock  # Updated import path if necessary


class BEV_LaneDet(nn.Module):
    def __init__(self, bev_shape, output_2d_shape, train=True, fusion_type='concat', sequence_height_head = False, temporal_length=1, **depth_distillation_settings):
        """
        Initializes the BEV_LaneDet model with configurable feature fusion strategies.

        Args:
            bev_shape (tuple): Shape of the BEV feature map.
            output_2d_shape (tuple): Shape of the 2D output feature map.
            train (bool): Flag indicating training mode.
            fusion_type (str): Type of feature fusion ('concat' or 'feature_fusion_block').
            **depth_distillation_settings: Additional settings for depth distillation.
        """
        super(BEV_LaneDet, self).__init__()

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Fusion type parameter
        self.fusion_type = fusion_type.lower()
        assert self.fusion_type in ['concat', 'feature_fusion_block'], \
            "fusion_type must be either 'concat' or 'feature_fusion_block'"

        # Custom ResNet34 Backbone with Skip Connections
        self.backbone = ResNet34_Backbone(pretrained=True).to(self.device)

        # Spatial Transformers to BEV
        self.down = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(1024)
                ),
                downsample=nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            )
        ).to(self.device)

        # Transformers for different scales
        self.s32transformer = FCTransform_((512, 18, 32), (256, 25, 5)).to(self.device)
        self.s64transformer = FCTransform_((1024, 9, 16), (256, 25, 5)).to(self.device)

        # Lane detection head
        # 时序高度预测头配置
        self.sequence_height_head = sequence_height_head
        self.temporal_length = temporal_length if sequence_height_head else 1  # 如果不启用时序，则时序长度为1

        if self.sequence_height_head:
            self.lane_head = LaneHeadResidual_Instance_with_offset_z(
                bev_shape,
                input_channel=512,
                temporal=True,
                temporal_length=self.temporal_length
            ).to(self.device)
        else:
            self.lane_head = LaneHeadResidual_Instance_with_offset_z(
                bev_shape,
                input_channel=512,
                temporal=False
            ).to(self.device)

        # Depth auxiliary head settings
        self.is_train = train
        self.depth_label_pred = depth_distillation_settings.get("response_distillation", True)
        self.depth_feature_pred = depth_distillation_settings.get("feature_distillation", True)
        print(f"Fusion Type: {self.fusion_type}")
        print(f"Depth Label Prediction: {self.depth_label_pred}, Depth Feature Prediction: {self.depth_feature_pred}")

        if self.is_train:
            # 2D lane detection auxiliary head
            self.lane_head_2d = LaneHeadResidual_Instance(output_2d_shape, input_channel=512).to(self.device)

            # Depth auxiliary head
            if self.depth_feature_pred or self.depth_label_pred:
                self.depth_anything = DepthAnythingV2(
                    encoder='vitl',
                    features=256,
                    out_channels=[256, 512, 1024, 1024],
                    use_bn=False,
                    use_clstoken=False
                )
                self.depth_anything.load_state_dict(
                    torch.load('/mnt/d/github/depth3dlane/models/pretrained/depth_anything_v2_vitl.pth', map_location=self.device)
                )
                self.depth_anything = self.depth_anything.to(self.device).eval()

                if self.depth_label_pred:
                    self.depth_head = DepthHeadUNet(in_channels=512, out_channels=1, base_channels=256).to(self.device)

        if self.depth_feature_pred:
            self.channel_reduction = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512)
            )
            # Initialize new student models
            self.student_s32 = StudentS32(in_channels=512, out_channels=512).to(self.device)
            self.student_s64 = StudentS64(in_channels=512, out_channels=1024).to(self.device)

            if self.fusion_type == 'concat':
                # Initialize fusion convolution layers
                self.fuse_conv_s32 = nn.Sequential(
                    nn.Conv2d(512 * 2, 512, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=False if self.is_train else True)
                ).to(self.device)

                self.fuse_conv_s64 = nn.Sequential(
                    nn.Conv2d(1024 * 2, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=False if self.is_train else True)
                ).to(self.device)
            elif self.fusion_type == 'feature_fusion_block':
                # Initialize Feature Fusion Blocks
                self.fuse_block_s32 = FeatureFusionBlock(
                    features=512,  # 512
                    activation=nn.ReLU(inplace=False if self.is_train else True),
                    deconv=False,
                    bn=True,
                    expand=False,  # To reduce channels back to 512
                    align_corners=True,
                    size=(18, 32)  # Default scaling
                ).to(self.device)

                self.fuse_block_s64 = FeatureFusionBlock(
                    features=1024,  # 1024
                    activation=nn.ReLU(inplace=False if self.is_train else True),
                    deconv=False,
                    bn=True,
                    expand=False,  # To reduce channels back to 1024
                    align_corners=True,
                    size=(9, 16)  # Default scaling
                ).to(self.device)
            else:
                raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

    def forward(self, img):

        # Move input to device
        img = img.to(self.device)

        # Backbone feature extraction with skip connections
        skips = self.backbone(img)  # List of feature maps [After maxpool, layer1, layer2, layer3, layer4]

        # Original img_s32 and img_s64 feature extraction
        img_s32 = skips[-1]  # (B, 512, 18, 32)
        img_s64 = self.down(img_s32)  # (B, 1024, 9, 16)

        if self.depth_feature_pred:
            distillation_feature = []
            # Pass skips[-1] (img_s32) through the student models
            student_features_s32 = self.student_s32(skips[-1])
            distillation_feature.append(student_features_s32)
            student_features_s64 = self.student_s64(student_features_s32)
            distillation_feature.append(student_features_s64)

            if self.fusion_type == "concat":
                # Concatenation followed by convolution to reduce channels
                fused_s32 = torch.cat([img_s32, student_features_s32], dim=1)  # (B, 1024, 18, 32)
                fused_s64 = torch.cat([img_s64, student_features_s64], dim=1)  # (B, 2048, 9, 16)

                # Apply convolution to fuse the concatenated features
                img_s32 = self.fuse_conv_s32(fused_s32)  # (B, 512, 18, 32)
                img_s64 = self.fuse_conv_s64(fused_s64)  # (B, 1024, 9, 16)

            elif self.fusion_type == "feature_fusion_block":
                # Use FeatureFusionBlock for feature fusion
                img_s32 = self.fuse_block_s32(img_s32, student_features_s32)  # (B, 512, 18, 32)
                img_s64 = self.fuse_block_s64(img_s64, student_features_s64)  # (B, 1024, 9, 16)
            else:
                raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        else:
            distillation_feature = None

        # Spatial transformers to BEV
        bev_32 = self.s32transformer(img_s32)  # (B, 256, 25, 5)
        bev_64 = self.s64transformer(img_s64)  # (B, 256, 25, 5)

        # Concatenate BEV features
        bev = torch.cat([bev_64, bev_32], dim=1)  # (B, 512, 25, 5)

        # Lane head predictions
        lane_outputs = self.lane_head(bev)  # (pred, emb, offset_y, z)

        if self.is_train:
            teacher_depth_label = None
            if self.depth_label_pred or self.depth_feature_pred:
                # for depth_feature_pred
                with torch.no_grad():  # with 里面的部分可以做并行计算
                    resized_img = F.interpolate(img, size=(574, 1022), mode='bilinear', align_corners=False)  # (B, C, 576, 1024) -> (B, C, 574, 1022)

                    # 获取 teacher_depth_label 和 teacher_features
                    teacher_depth_label, teacher_features = self.depth_anything(resized_img)  # teacher_features: List of feature maps

                    # Process teacher_depth_label
                    teacher_depth_label = teacher_depth_label.unsqueeze(1)  # Add a channel dimension (B, 1, 574, 1022)
                    teacher_depth_label = F.interpolate(teacher_depth_label, size=(576, 1024), mode='bilinear', align_corners=True)  # (B, 1, 576, 1024)

                    # teacher_depth_label = torch.sigmoid(teacher_depth_label)  # Normalize depth label
                if self.depth_feature_pred:
                    # 2. 从 teacher_features 中提取 s32 和 s64 特征
                    teacher_feature_s32 = teacher_features[-2]
                    teacher_feature_s32 = self.channel_reduction(teacher_feature_s32)
                    teacher_feature_s32 = F.interpolate(teacher_feature_s32, size=(18, 32), mode='bilinear',align_corners=True) # (B, 512, 18, 32)
                    distillation_feature.append(teacher_feature_s32)

                    teacher_feature_s64 = teacher_features[-1]
                    teacher_feature_s64 = F.interpolate(teacher_feature_s64, size=(9, 16), mode='bilinear',align_corners=True) # (B, 1024, 9, 16)
                    distillation_feature.append(teacher_feature_s64)

            # for 2d axuliary head
            lane_2d_outputs = self.lane_head_2d(img_s32)  # (pred_2d, emb_2d)

            if self.depth_label_pred:
                # Depth prediction with skip connections
                depth_map = self.depth_head(img_s32, skips)  # (B, 1, 576, 1024)
            else:
                depth_map = None

            # 构建输出字典
            outputs_dict = {
                'lane_outputs': lane_outputs,
                'lane_2d_outputs': lane_2d_outputs,
                'depth_map': depth_map,
                'teacher_depth_label': teacher_depth_label,
                'distillation_feature': distillation_feature
            }

            return outputs_dict
        else:
            return {'lane_outputs': lane_outputs}
