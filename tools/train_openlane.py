# train_openlane.py
import sys
sys.path.append(r"/media/bluewolf/Data/bluewolf/projs/distillation/Depth3DLane")
import os
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
from models.loss.depth_combined_loss import CombinedDepthLoss
from models.loss.feature_distillation_loss import FeatureDistillationLoss
import matplotlib.pyplot as plt
import json


class LossCalculator(nn.Module):
    def __init__(self, device='cuda'):
        super(LossCalculator, self).__init__()
        self.device = device

        # 损失函数
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(self.device))
        self.iou_loss = IoULoss().to(self.device)
        self.poopoo = NDPushPullLoss(1.0, 1.0, 1.0, 5.0, 200).to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.bce_loss = nn.BCELoss().to(self.device)

        # 深度损失
        self.combined_depth_loss = CombinedDepthLoss(alpha=0.5, beta=0.3, gamma=0.2).to(self.device)

        # 特征蒸馏损失
        self.depth_feature_distillation_loss = FeatureDistillationLoss().to(self.device)

    def forward(self, outputs, gt, is_train=True):
        """
        计算所有相关的损失。

        参数：
            outputs (dict): 模型输出，包含预测结果。
            gt (dict): 真实标签张量。
            is_train (bool): 是否为训练模式。

        返回：
            dict: 包含各个损失组件和总损失的字典。
        """
        if not is_train:
            return {}

        # 解包输出
        lane_outputs = outputs['lane_outputs']  # (pred, emb, offset_y, z)
        lane_2d_outputs = outputs['lane_2d_outputs']  # (pred_2d, emb_2d)
        depth_map = outputs.get('depth_map', None)  # (B, 1, H, W) 或 None
        teacher_depth_label = outputs.get('teacher_depth_label', None)  # 教师深度标签
        distillation_feature = outputs.get('distillation_feature', None)  # 教师特征

        pred, emb, offset_y, z = lane_outputs
        pred_2d, emb_2d = lane_2d_outputs

        # 3D损失
        # 分割损失
        loss_seg = self.bce(pred, gt['gt_seg']) + self.iou_loss(torch.sigmoid(pred), gt['gt_seg'])

        # 嵌入损失
        loss_emb = self.poopoo(emb, gt['gt_instance'])

        # 偏移损失
        loss_offset = self.bce_loss(torch.sigmoid(offset_y) * gt['gt_seg'], gt['gt_offset_y'])

        # 高度损失
        loss_z = self.mse_loss(torch.sigmoid(z) * gt['gt_seg'], gt['gt_z'])

        # 综合BEV损失
        loss_total = 3.0 * loss_seg + 0.5 * loss_emb

        # 2D损失
        # 2D分割损失
        loss_seg_2d = self.bce(pred_2d, gt['image_gt_segment']) + self.iou_loss(torch.sigmoid(pred_2d),
                                                                                gt['image_gt_segment'])

        # 2D嵌入损失
        loss_emb_2d = self.poopoo(emb_2d, gt['image_gt_instance'])

        # 综合2D损失
        loss_total_2d = 3.0 * loss_seg_2d + 0.5 * loss_emb_2d

        # 深度损失
        loss_depth = self.combined_depth_loss(depth_map, teacher_depth_label) if depth_map is not None else 0.0

        # 蒸馏损失
        distillation_loss = self.depth_feature_distillation_loss(distillation_feature[0], distillation_feature[1],
                                                                 distillation_feature[2], distillation_feature[3]) if distillation_feature is not None else 0.0

        # 总综合损失
        loss_total_combined = (
                loss_total +
                0.5 * loss_total_2d +
                60 * loss_offset +
                30 * loss_z +
                loss_depth +
                distillation_loss
        )

        # 汇总所有损失
        loss_dict = {
            "loss_total_combined": loss_total_combined,
            "loss_total_2d": loss_total_2d,
            "loss_offset": loss_offset,
            "loss_z": loss_z,
            "loss_depth": loss_depth,
            "distillation_loss": distillation_loss,
        }

        return loss_dict


def train_epoch(model, loss_calculator, dataloader, optimizer, configs, epoch, device, model_save_path):
    """
    训练模型一个epoch，保存损失曲线和模型。

    参数：
        model (torch.nn.Module): BEV_LaneDet模型。
        loss_calculator (LossCalculator): 损失计算模块。
        dataloader (DataLoader): 训练数据的DataLoader。
        optimizer (torch.optim.Optimizer): 优化器。
        configs: 包含训练参数的配置模块。
        epoch (int): 当前epoch编号。
        device (torch.device): 计算设备。
        model_save_path (str): 保存模型和图像的路径。

    返回：
        dict: 包含epoch平均损失组件的字典。
    """
    model.train()

    # 初始化损失累加器
    total_loss_total_combined = 0.0
    total_loss_total_2d = 0.0
    total_loss_offset = 0.0
    total_loss_z = 0.0
    total_loss_depth = 0.0
    total_distillation_loss = 0.0
    num_batches = 0

    # 初始化每批次损失列表
    loss_total_combined_list = []
    loss_total_2d_list = []
    loss_offset_list = []
    loss_z_list = []
    loss_depth_list = []
    distillation_loss_list = []

    for idx, batch in enumerate(dataloader):
        # 解包批次数据
        input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance = batch

        # 将数据移动到设备上
        input_data = input_data.to(device)
        gt_seg_data = gt_seg_data.to(device)
        gt_emb_data = gt_emb_data.to(device)
        offset_y_data = offset_y_data.to(device)
        z_data = z_data.to(device)
        image_gt_segment = image_gt_segment.to(device)
        image_gt_instance = image_gt_instance.to(device)

        # 准备真实标签字典
        gt = {
            'gt_seg': gt_seg_data,
            'gt_instance': gt_emb_data,
            'gt_offset_y': offset_y_data,
            'gt_z': z_data,
            'image_gt_segment': image_gt_segment,
            'image_gt_instance': image_gt_instance,
        }

        # 模型前向传播
        outputs = model(input_data)

        # 计算损失
        loss_dict = loss_calculator(outputs, gt, is_train=True)

        # 提取各个损失
        loss_total_combined = loss_dict["loss_total_combined"]
        loss_total_2d = loss_dict["loss_total_2d"]
        loss_offset = loss_dict["loss_offset"]
        loss_z = loss_dict["loss_z"]
        loss_depth = loss_dict["loss_depth"]
        distillation_loss = loss_dict["distillation_loss"]

        # 反向传播和优化
        optimizer.zero_grad()
        loss_total_combined.backward()
        optimizer.step()

        # 累加损失
        total_loss_total_combined += loss_total_combined.item()
        total_loss_total_2d += loss_total_2d.item()
        total_loss_offset += loss_offset.item()
        total_loss_z += loss_z.item()
        if isinstance(loss_depth, torch.Tensor):
            total_loss_depth += loss_depth.item()
        else:
            total_loss_depth += loss_depth
        if isinstance(distillation_loss, torch.Tensor):
            total_distillation_loss += distillation_loss.item()
        else:
            total_distillation_loss += distillation_loss
        num_batches += 1

        # 将每批次损失添加到列表
        loss_total_combined_list.append(loss_total_combined.item())
        loss_total_2d_list.append(loss_total_2d.item())
        loss_offset_list.append(loss_offset.item())
        loss_z_list.append(loss_z.item())
        if isinstance(loss_depth, torch.Tensor):
            loss_depth_list.append(loss_depth.item())
        else:
            loss_depth_list.append(loss_depth)
        if isinstance(distillation_loss, torch.Tensor):
            distillation_loss_list.append(distillation_loss.item())
        else:
            distillation_loss_list.append(distillation_loss)

        # 日志记录
        if idx % 50 == 0:
            print(
                f"Epoch [{epoch + 1}], Step [{idx}/{len(dataloader)}], "
                f"BEV Loss: {loss_total_combined.item():.4f}, "
                f"Depth Loss: {loss_depth.item() if isinstance(loss_depth, torch.Tensor) else loss_depth:.4f}, "
                f"Distillation Loss: {distillation_loss.item() if isinstance(distillation_loss, torch.Tensor) else distillation_loss:.4f}, "
            )

        if idx % 300 == 0:
            # 计算BEV分割的F1得分
            with torch.no_grad():
                target = gt['gt_seg'].detach().cpu().numpy().ravel()
                pred_prob = torch.sigmoid(outputs['lane_outputs'][0]).detach().cpu().numpy().ravel()
                pred_label = (pred_prob > 0.5).astype(np.int64)
                target_label = (target > 0.5).astype(np.int64)
                f1_bev_seg = f1_score(target_label, pred_label, zero_division=1)

            loss_iter = {
                "BEV Loss": loss_total_combined.item(),
                "Offset Loss": loss_offset.item(),
                "Z Loss": loss_z.item(),
                "Depth Loss": loss_depth.item() if isinstance(loss_depth, torch.Tensor) else loss_depth,
                "Distillation Loss": distillation_loss.item() if isinstance(distillation_loss, torch.Tensor) else distillation_loss,
                "F1_BEV_seg": f1_bev_seg
            }

    # 计算epoch的平均损失
    avg_losses = {
        "loss_total_combined": total_loss_total_combined / num_batches,
        "loss_total_2d": total_loss_total_2d / num_batches,
        "loss_offset": total_loss_offset / num_batches,
        "loss_z": total_loss_z / num_batches,
        "loss_depth": total_loss_depth / num_batches,
        "distillation_loss": total_distillation_loss / num_batches,
    }

    # 为当前epoch创建一个文件夹
    folder_name = os.path.join(model_save_path, f'epoch_{epoch + 1}')
    os.makedirs(folder_name, exist_ok=True)

    # 将每批次损失保存为JSON
    losses_dict = {
        "loss_total_combined": loss_total_combined_list,
        "loss_total_2d": loss_total_2d_list,
        "loss_offset": loss_offset_list,
        "loss_z": loss_z_list,
        "loss_depth": loss_depth_list,
        "distillation_loss": distillation_loss_list,
    }
    json_path = os.path.join(folder_name, f'losses_epoch_{epoch + 1}.json')
    with open(json_path, 'w') as f:
        json.dump(losses_dict, f)
    print(f"Saved per-batch losses for epoch {epoch + 1} at {json_path}")

    # 绘制每批次损失曲线
    plt.figure(figsize=(10, 6))
    for loss_name, loss_values in losses_dict.items():
        plt.plot(range(1, len(loss_values) + 1), loss_values, label=loss_name)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curves for Epoch {epoch + 1}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(folder_name, f'loss_epoch_{epoch + 1}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot for epoch {epoch + 1} at {plot_path}")

    # 将平均损失保存为JSON
    avg_losses_path = os.path.join(folder_name, f'avg_losses_epoch_{epoch + 1}.json')
    with open(avg_losses_path, 'w') as f:
        json.dump(avg_losses, f)
    print(f"Saved average losses for epoch {epoch + 1} at {avg_losses_path}")

    # 将模型检查点保存到此文件夹
    save_model_dp(model, optimizer, folder_name, f'ep{epoch + 1:03d}.pth')
    print(f"Saved model checkpoint for epoch {epoch + 1} at {folder_name}")

    return avg_losses  # 返回平均损失


class WorkerFunction:
    def __init__(self, config_file, gpu_id, checkpoint_path=None):
        self.config_file = config_file
        self.gpu_id = gpu_id
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
        self.configs = load_config_module(config_file)

        ''' 初始化模型 '''
        self.model = self.configs.model().to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=gpu_id)

        ''' 初始化损失计算器 '''
        self.loss_calculator = LossCalculator(device=self.device)

        ''' 初始化优化器和调度器 '''
        self.optimizer = self.configs.optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.configs.optimizer_params
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.configs.epochs)

        ''' 加载检查点（如果提供） '''
        if self.checkpoint_path:
            if getattr(self.configs, "load_optimizer", True):
                resume_training(self.checkpoint_path, self.model.module, self.optimizer, self.scheduler)
            else:
                load_checkpoint(self.checkpoint_path, self.model.module, None)

        ''' 初始化数据集和DataLoader '''
        Dataset = getattr(self.configs, "train_dataset", None)
        if Dataset is None:
            Dataset = self.configs.train_dataset
        self.train_loader = DataLoader(Dataset(), **self.configs.loader_args, pin_memory=True)

        ''' 初始化用于绘图的损失历史记录 '''
        self.loss_history = {
            "loss_total_combined": [],
            "loss_total_2d": [],
            "loss_offset": [],
            "loss_z": [],
            "loss_depth": [],
            "distillation_loss": [],
        }

    def train(self):
        ''' 训练循环 '''
        for epoch in range(self.configs.epochs):
            print('*' * 100, f"Epoch {epoch + 1}/{self.configs.epochs}")
            avg_losses = train_epoch(
                model=self.model,
                loss_calculator=self.loss_calculator,
                dataloader=self.train_loader,
                optimizer=self.optimizer,
                configs=self.configs,
                epoch=epoch,
                device=self.device,
                model_save_path=self.configs.model_save_path  # 传递模型保存路径
            )
            self.scheduler.step()

            # 将平均损失添加到历史记录
            for key in self.loss_history:
                self.loss_history[key].append(avg_losses[key])

            # 保存最新的模型检查点
            save_model_dp(self.model, None, self.configs.model_save_path, 'latest.pth')

    def plot_losses(self, epoch, folder_name):
        """
        绘制损失曲线并保存图像。

        参数：
            epoch (int): 当前epoch编号。
            folder_name (str): 保存图像的文件夹。
        """
        plt.figure(figsize=(10, 6))
        for loss_name, loss_values in self.loss_history.items():
            plt.plot(range(1, epoch + 1), loss_values, label=loss_name)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 将图像保存到指定文件夹
        plot_path = os.path.join(folder_name, f'loss_epoch_{epoch}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved loss plot for epoch {epoch} at {plot_path}")


def worker_function(config_file, gpu_id, checkpoint_path=None):
    """
    工作函数，初始化并开始训练。

    参数：
        config_file (str): 配置文件的路径。
        gpu_id (list): 使用的GPU ID列表。
        checkpoint_path (str, optional): 用于恢复训练的检查点路径。
    """
    print('Using GPU IDs:', ','.join([str(i) for i in gpu_id]))
    trainer = WorkerFunction(config_file, gpu_id, checkpoint_path)
    trainer.train()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    worker_function('./openlane_config.py', gpu_id=[0])
