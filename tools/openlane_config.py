import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from loader.bev_road.openlane_data import OpenLane_dataset_with_offset, OpenLane_dataset_with_offset_val
from models.model.single_camera_bev import BEV_LaneDet

base_dir = "/media/bluewolf/Data/bluewolf/projs/distillation/Depth3DLane"
''' data split '''
train_gt_paths = os.path.join(base_dir, 'data/lane3d_1000/training')
train_image_paths = os.path.join(base_dir, 'data/images/training')

val_gt_paths = os.path.join(base_dir, 'data/lane3d_1000/validation')
val_image_paths = os.path.join(base_dir, 'data/images/validation')

model_save_path = os.path.join(base_dir, 'results/openlane')

input_shape = (576, 1024)
output_2d_shape = (144, 256)

dpt_path = os.path.join(base_dir, r"models/pretrained/depth_anything_v2_vitl.pth")

''' BEV Range Configuration '''
x_range = (3, 103)
y_range = (-12, 12)
meter_per_pixel = 0.5  # Grid size
bev_shape = (
    int((x_range[1] - x_range[0]) / meter_per_pixel),
    int((y_range[1] - y_range[0]) / meter_per_pixel)
)

loader_args = dict(
    batch_size=4,
    num_workers=12,
    shuffle=True
)

''' Virtual Camera Configuration '''
vc_config = {}
vc_config['use_virtual_camera'] = True
vc_config['vc_intrinsic'] = np.array([
    [2081.5212033927246, 0.0, 934.7111248349433],
    [0.0, 2081.5212033927246, 646.3389987785433],
    [0.0, 0.0, 1.0]
])
vc_config['vc_extrinsics'] = np.array([
    [-0.002122161262459438, 0.010697496358766389, 0.9999405282331697, 1.5441039498273286],
    [-0.9999378331046326, -0.010968621415360667, -0.0020048117763292747, -0.023774034344867204],
    [0.010946522625388108, -0.9998826195688676, 0.01072010851209982, 2.1157397903843567],
    [0.0, 0.0, 0.0, 1.0]
])
vc_config['vc_image_shape'] = (1920, 1280)

''' Model Definition '''
def model():
    fusion_type = 'feature_fusion_block'  # or 'concat'
    # fusion_type = 'concat'
    return BEV_LaneDet(
        bev_shape=bev_shape,
        output_2d_shape=output_2d_shape,
        train=True,
        sequence_height_head=False,
        temporal_length=4,
        dpt_path=dpt_path,
        fusion_type=fusion_type,
        response_distillation=False,    # Set to True if response distillation is needed
        feature_distillation=True      # Enable feature distillation
    )

''' Optimizer and Scheduler Configuration '''
epochs = 50
optimizer = AdamW
optimizer_params = dict(
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,
    amsgrad=False
)
scheduler = CosineAnnealingLR

''' Training Dataset Configuration '''
def train_dataset():
    train_trans = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.MotionBlur(p=0.2),
        A.RandomBrightnessContrast(),
        A.ColorJitter(p=0.1),
        A.Normalize(),
        ToTensorV2()
    ])
    train_data = OpenLane_dataset_with_offset(
        image_paths=train_image_paths,
        gt_paths=train_gt_paths,
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        data_trans=train_trans,
        output_2d_shape=output_2d_shape,
        virtual_camera_config=vc_config
    )
    return train_data

''' Validation Dataset Configuration '''
def val_dataset():
    trans_image = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(),
        ToTensorV2()
    ])
    val_data = OpenLane_dataset_with_offset_val(
        image_paths=val_image_paths,
        gt_paths=val_gt_paths,
        data_trans=trans_image,
        virtual_camera_config=vc_config
    )
    return val_data