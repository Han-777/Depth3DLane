import sys
sys.path.append('/media/bluewolf/Data/bluewolf/projs/Depth3DLane')
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from models.loss import IoULoss, NDPushPullLoss
from utils.config_util import load_config_module
from sklearn.metrics import f1_score
import numpy as np
import torch.nn.functional as F
from models.model.dpt import DepthAnythingV2

class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None, image_gt_segment=None,
                image_gt_instance=None, train=True):
        # 完整传递所有参数给模型
        res = self.model(inputs, gt_seg, gt_instance, gt_offset_y, gt_z, image_gt_segment, image_gt_instance)
        # 假设模型的输出顺序为：
        # res = (lane_output, lane_output_2d, img_s32, img_s64, depth_pred, z_pred)
        # 根据您的 BEV_LaneDet 的定义，请确保正确解析
        lane_output, lane_output_2d, img_s32, img_s64, depth_pred = res  # 根据模型实际输出调整

        if train:
            # 3D 损失
            loss_seg = self.bce(lane_output, gt_seg) + self.iou_loss(torch.sigmoid(lane_output), gt_seg)
            loss_emb = self.poopoo(emb, gt_instance)  # 确保 'emb' 已定义
            loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y)
            loss_z = self.mse_loss(z_pred, gt_z)  # 使用 MSELoss 计算 z 预测损失

            loss_total = 3 * loss_seg + 0.5 * loss_emb
            loss_total = loss_total.unsqueeze(0)
            loss_offset = 60 * loss_offset.unsqueeze(0)
            loss_z = 30 * loss_z.unsqueeze(0)

            # 2D 损失
            loss_seg_2d = self.bce(lane_output_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(lane_output_2d), image_gt_segment)
            loss_emb_2d = self.poopoo(emb_2d, image_gt_instance)  # 确保 'emb_2d' 已定义
            loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d
            loss_total_2d = loss_total_2d.unsqueeze(0)

            return lane_output, loss_total, loss_total_2d, loss_offset, loss_z, depth_pred
        else:
            return lane_output, depth_pred


def initialize_teacher_model(device):
    teacher_model = DepthAnythingV2(
        encoder='vitl',
        features=256,
        out_channels=[256,512,1024,1024],
        use_bn=False,
        use_clstoken=False
    )
    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model

def train_epoch(model, dataset, optimizer, configs, epoch, teacher_model, device, lambda_distill=0.5):
    """
    修改后的训练循环，包含蒸馏损失。

    参数:
    - model: Combine_Model_and_Loss 包装后的 BEV_LaneDet 模型
    - dataset: 训练数据集
    - optimizer: 优化器
    - configs: 配置参数
    - epoch: 当前轮数
    - teacher_model: DPT 教师模型
    - device: 计算设备
    - lambda_distill: 蒸馏损失的权重
    """
    model.train()
    teacher_model.eval()  # 确保教师模型在评估模式
    losses_avg = {}
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''

    for idx, (
        input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance
    ) in enumerate(dataset):
        # 将数据迁移到设备
        input_data = input_data.to(device)
        gt_seg_data = gt_seg_data.to(device)
        gt_emb_data = gt_emb_data.to(device)
        offset_y_data = offset_y_data.to(device)
        z_data = z_data.to(device)
        image_gt_segment = image_gt_segment.to(device)
        image_gt_instance = image_gt_instance.to(device)

        # 前向传播：获取 BEV_LaneDet 的输出，包括深度预测
        prediction, loss_total_bev, loss_total_2d, loss_offset, loss_z, depth_pred = model(
            input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data,
            image_gt_segment, image_gt_instance, train=True
        )

        # 计算总损失（不包括蒸馏损失）
        loss_back_total = loss_total_bev.mean() + 0.5 * loss_total_2d.mean() + loss_offset.mean() + loss_z.mean()

        # 获取教师模型的深度预测（无需计算梯度）
        with torch.no_grad():
            depth_teacher, _ = teacher_model(input_data)
            depth_teacher = depth_teacher.to(device)

        # 计算蒸馏损失（例如，均方误差损失）
        distillation_loss = F.mse_loss(depth_pred, depth_teacher)

        # 组合总损失
        total_loss = loss_back_total + lambda_distill * distillation_loss

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 记录和打印损失
        if idx % 50 == 0:
            print(f"Epoch [{epoch}], Iter [{idx}], BEV Loss: {loss_total_bev.mean().item():.4f}, "
                  f"2D Loss: {loss_total_2d.mean().item():.4f}, Offset Loss: {loss_offset.mean().item():.4f}, "
                  f"Z Loss: {loss_z.mean().item():.4f}, Distillation Loss: {distillation_loss.item():.4f}, "
                  f"Total Loss: {total_loss.item():.4f}")

        if idx % 300 == 0:
            target = gt_seg_data.detach().cpu().numpy().ravel()
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
            f1_bev_seg = f1_score(
                (target > 0.5).astype(np.int64),
                (pred > 0.5).astype(np.int64),
                zero_division=1
            )
            loss_iter = {
                "BEV Loss": loss_total_bev.mean().item(),
                '2D Loss': loss_total_2d.mean().item(),
                'Offset Loss': loss_offset.mean().item(),
                'Z Loss': loss_z.mean().item(),
                "F1_BEV_seg": f1_bev_seg,
                "Distillation Loss": distillation_loss.item(),
                "Total Loss": total_loss.item()
            }
            print(f"Epoch [{epoch}], Iter [{idx}], Losses: {loss_iter}")



def worker_function(config_file, gpu_id, checkpoint_path=None):
    print('use gpu ids is '+','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)

    ''' models and optimizer '''
    model = configs.model()
    model = Combine_Model_and_Loss(model)
    device = torch.device(f'cuda:{gpu_id[0]}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_id)
    optimizer = configs.optimizer(filter(lambda p: p.requires_grad, model.parameters()), **configs.optimizer_params)
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)
    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            resume_training(checkpoint_path, model.module, optimizer, scheduler)
        else:
            load_checkpoint(checkpoint_path, model.module, None)

    ''' Initialize teacher model '''
    teacher_model = initialize_teacher_model(device)

    ''' dataset '''
    Dataset = getattr(configs, "train_dataset", None)
    if Dataset is None:
        Dataset = configs.training_dataset
    train_loader = DataLoader(Dataset(), **configs.loader_args, pin_memory=True)

    ''' get validation '''
    # if configs.with_validation:
    #     val_dataset = Dataset(**configs.val_dataset_args)
    #     val_loader = DataLoader(val_dataset, **configs.val_loader_args, pin_memory=True)
    #     val_loss = getattr(configs, "val_loss", loss)
    #     if eval_only:
    #         loss_mean = val_dp(model, val_loader, val_loss)
    #         print(loss_mean)
    #         return

    for epoch in range(configs.epochs):
        print('*' * 100, epoch)
        train_epoch(model, train_loader, optimizer, configs, epoch, teacher_model, device)
        scheduler.step()
        save_model_dp(model, optimizer, configs.model_save_path, f'ep{epoch:03d}.pth')
        save_model_dp(model, None, configs.model_save_path, 'latest.pth')


# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    worker_function('./openlane_config.py', gpu_id=[0])
