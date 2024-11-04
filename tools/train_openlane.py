# train_openlane.py

import sys
import os

# Add the directory containing the 'models' module to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Print the system path for debugging
print("System path:", sys.path)

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
import matplotlib.pyplot as plt  # Added for plotting


class LossCalculator(nn.Module):
    def __init__(self, device="cuda"):
        super(LossCalculator, self).__init__()
        self.device = device

        # Loss functions
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(self.device))
        self.iou_loss = IoULoss().to(self.device)
        self.poopoo = NDPushPullLoss(1.0, 1.0, 1.0, 5.0, 200).to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.bce_loss = nn.BCELoss().to(self.device)

        # Combined depth loss
        self.combined_depth_loss = CombinedDepthLoss(alpha=0.5, beta=0.3, gamma=0.2).to(
            self.device
        )

        # Depth Feature distillation loss
        self.depth_feature_distillation_loss = FeatureDistillationLoss().to(self.device)

    def forward(self, outputs, gt, is_train=True):
        """
        Computes all relevant losses.

        Parameters:
            outputs (dict): Model outputs containing predictions.
            gt (dict): Ground truth tensors.
            is_train (bool): Flag indicating training mode.

        Returns:
            dict: Dictionary containing individual loss components and total loss.
        """
        if not is_train:
            return {}

        # Unpack outputs
        lane_outputs = outputs["lane_outputs"]  # (pred, emb, offset_y, z)
        lane_2d_outputs = outputs["lane_2d_outputs"]  # (pred_2d, emb_2d)
        depth_map = outputs.get("depth_map", None)  # (B, 1, H, W) or None
        teacher_depth_label = outputs.get(
            "teacher_depth_label", None
        )  # Teacher's depth labels
        distillation_feature = outputs.get(
            "distillation_feature", None
        )  # Teacher's features

        pred, emb, offset_y, z = lane_outputs
        pred_2d, emb_2d = lane_2d_outputs

        # 3D Losses
        # Segmentation Loss
        loss_seg = self.bce(pred, gt["gt_seg"]) + self.iou_loss(
            torch.sigmoid(pred), gt["gt_seg"]
        )

        # Embedding Loss
        loss_emb = self.poopoo(emb, gt["gt_instance"])

        # Offset Loss
        loss_offset = self.bce_loss(
            torch.sigmoid(offset_y) * gt["gt_seg"], gt["gt_offset_y"]
        )

        # Height Loss
        loss_z = self.mse_loss(torch.sigmoid(z) * gt["gt_seg"], gt["gt_z"])

        # Combined BEV Loss
        loss_total = 3.0 * loss_seg + 0.5 * loss_emb

        # 2D Losses
        # 2D Segmentation Loss
        loss_seg_2d = self.bce(pred_2d, gt["image_gt_segment"]) + self.iou_loss(
            torch.sigmoid(pred_2d), gt["image_gt_segment"]
        )

        # 2D Embedding Loss
        loss_emb_2d = self.poopoo(emb_2d, gt["image_gt_instance"])

        # Combined 2D Loss
        loss_total_2d = 3.0 * loss_seg_2d + 0.5 * loss_emb_2d

        # Depth Loss
        loss_depth = (
            self.combined_depth_loss(depth_map, teacher_depth_label)
            if depth_map is not None
            else 0.0
        )

        # Distillation Loss
        distillation_loss = (
            self.depth_feature_distillation_loss(
                distillation_feature[0],
                distillation_feature[1],
                distillation_feature[2],
                distillation_feature[3],
            )
            if distillation_feature is not None
            else 0.0
        )

        # Total combined loss
        loss_total_combined = (
            loss_total
            + 0.5 * loss_total_2d
            + 60 * loss_offset
            + 30 * loss_z
            + loss_depth
            + distillation_loss
        )

        # Aggregate all losses
        loss_dict = {
            "loss_total_combined": loss_total_combined,
            "loss_total_2d": loss_total_2d,
            "loss_offset": loss_offset,
            "loss_z": loss_z,
            "loss_depth": loss_depth,
            "distillation_loss": distillation_loss,
        }

        return loss_dict


def train_epoch(model, loss_calculator, dataloader, optimizer, configs, epoch, device):
    """
    Trains the model for one epoch.

    Parameters:
        model (torch.nn.Module): The BEV_LaneDet model.
        loss_calculator (LossCalculator): The loss computation module.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        configs: Configuration module containing training parameters.
        epoch (int): Current epoch number.
        device (torch.device): Device to run computations on.

    Returns:
        dict: Dictionary containing average loss components for the epoch.
    """
    model.train()

    # Initialize accumulators for losses
    total_loss_total_combined = 0.0
    total_loss_total_2d = 0.0
    total_loss_offset = 0.0
    total_loss_z = 0.0
    total_loss_depth = 0.0
    total_distillation_loss = 0.0
    num_batches = 0

    for idx, batch in enumerate(dataloader):
        # Unpack batch
        (
            input_data,
            gt_seg_data,
            gt_emb_data,
            offset_y_data,
            z_data,
            image_gt_segment,
            image_gt_instance,
        ) = batch

        # Move data to device
        input_data = input_data.to(device)
        gt_seg_data = gt_seg_data.to(device)
        gt_emb_data = gt_emb_data.to(device)
        offset_y_data = offset_y_data.to(device)
        z_data = z_data.to(device)
        image_gt_segment = image_gt_segment.to(device)
        image_gt_instance = image_gt_instance.to(device)

        # Prepare ground truth dictionary
        gt = {
            "gt_seg": gt_seg_data,
            "gt_instance": gt_emb_data,
            "gt_offset_y": offset_y_data,
            "gt_z": z_data,
            "image_gt_segment": image_gt_segment,
            "image_gt_instance": image_gt_instance,
        }

        # Forward pass through the model
        outputs = model(input_data)

        # Add additional outputs if necessary
        # For example, if model outputs 'teacher_features', include them in outputs

        # Compute losses
        loss_dict = loss_calculator(outputs, gt, is_train=True)

        # Extract individual losses
        loss_total_combined = loss_dict["loss_total_combined"]
        loss_total_2d = loss_dict["loss_total_2d"]
        loss_offset = loss_dict["loss_offset"]
        loss_z = loss_dict["loss_z"]
        loss_depth = loss_dict["loss_depth"]
        distillation_loss = loss_dict["distillation_loss"]

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_total_combined.backward()
        optimizer.step()

        # Accumulate losses
        total_loss_total_combined += loss_total_combined.item()
        total_loss_total_2d += loss_total_2d.item()
        total_loss_offset += loss_offset.item()
        total_loss_z += loss_z.item()
        total_loss_depth += (
            loss_depth.item() if isinstance(loss_depth, torch.Tensor) else loss_depth
        )
        total_distillation_loss += (
            distillation_loss.item()
            if isinstance(distillation_loss, torch.Tensor)
            else distillation_loss
        )
        num_batches += 1

        # Logging
        if idx % 50 == 0:
            print(
                f"Epoch [{epoch + 1}], Step [{idx}/{len(dataloader)}], "
                f"BEV Loss: {loss_total_combined.item():.4f}, "
                f"Depth Loss: {loss_depth.item() if isinstance(loss_depth, torch.Tensor) else loss_depth:.4f}, "
                f"Distillation Loss: {distillation_loss.item() if isinstance(distillation_loss, torch.Tensor) else distillation_loss:.4f}, "
            )

        if idx % 300 == 0:
            # Calculate F1 Score for BEV segmentation
            with torch.no_grad():
                target = gt["gt_seg"].detach().cpu().numpy().ravel()
                pred_prob = (
                    torch.sigmoid(outputs["lane_outputs"][0])
                    .detach()
                    .cpu()
                    .numpy()
                    .ravel()
                )
                pred_label = (pred_prob > 0.5).astype(np.int64)
                target_label = (target > 0.5).astype(np.int64)
                f1_bev_seg = f1_score(target_label, pred_label, zero_division=1)

            loss_iter = {
                "BEV Loss": loss_total_combined.item(),
                "Offset Loss": loss_offset.item(),
                "Z Loss": loss_z.item(),
                "Depth Loss": (
                    loss_depth.item()
                    if isinstance(loss_depth, torch.Tensor)
                    else loss_depth
                ),
                "Distillation Loss": (
                    distillation_loss.item()
                    if isinstance(distillation_loss, torch.Tensor)
                    else distillation_loss
                ),
                "F1_BEV_seg": f1_bev_seg,
            }
            print(f"Epoch [{epoch + 1}], Step [{idx}], Losses: {loss_iter}")

    # Compute average losses for the epoch
    avg_losses = {
        "loss_total_combined": total_loss_total_combined / num_batches,
        "loss_total_2d": total_loss_total_2d / num_batches,
        "loss_offset": total_loss_offset / num_batches,
        "loss_z": total_loss_z / num_batches,
        "loss_depth": total_loss_depth / num_batches,
        "distillation_loss": total_distillation_loss / num_batches,
    }

    return avg_losses  # Return average losses


class WorkerFunction:
    def __init__(self, config_file, gpu_id, checkpoint_path=None):
        self.config_file = config_file
        self.gpu_id = gpu_id
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(
            f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu"
        )
        self.configs = load_config_module(config_file)

        """ Initialize Model """
        self.model = self.configs.model().to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=gpu_id)

        """ Initialize Loss Calculator """
        self.loss_calculator = LossCalculator(device=self.device)

        """ Initialize Optimizer and Scheduler """
        self.optimizer = self.configs.optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.configs.optimizer_params,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.configs.epochs)

        """ Load Checkpoint if Provided """
        if self.checkpoint_path:
            if getattr(self.configs, "load_optimizer", True):
                resume_training(
                    self.checkpoint_path,
                    self.model.module,
                    self.optimizer,
                    self.scheduler,
                )
            else:
                load_checkpoint(self.checkpoint_path, self.model.module, None)

        """ Initialize Dataset and DataLoader """
        Dataset = getattr(self.configs, "train_dataset", None)
        if Dataset is None:
            Dataset = self.configs.train_dataset
        self.train_loader = DataLoader(
            Dataset(), **self.configs.loader_args, pin_memory=True
        )

        """ Initialize Loss History for Plotting """
        self.loss_history = {
            "loss_total_combined": [],
            "loss_total_2d": [],
            "loss_offset": [],
            "loss_z": [],
            "loss_depth": [],
            "distillation_loss": [],
        }

        """ Initialize Plot Directory """
        self.plot_dir = os.path.join(self.configs.model_save_path, "loss_plots")
        os.makedirs(
            self.plot_dir, exist_ok=True
        )  # Create directory if it doesn't exist

    def train(self):
        """Training Loop"""
        for epoch in range(self.configs.epochs):
            print("*" * 100, f"Epoch {epoch + 1}/{self.configs.epochs}")
            avg_losses = train_epoch(
                model=self.model,
                loss_calculator=self.loss_calculator,
                dataloader=self.train_loader,
                optimizer=self.optimizer,
                configs=self.configs,
                epoch=epoch,
                device=self.device,
            )
            self.scheduler.step()

            # Append average losses to history
            for key in self.loss_history:
                self.loss_history[key].append(avg_losses[key])

            # Plot and save loss diagrams
            self.plot_losses(epoch + 1)

            # Save model checkpoints
            save_model_dp(
                self.model,
                self.optimizer,
                self.configs.model_save_path,
                f"ep{epoch + 1:03d}.pth",
            )
            save_model_dp(self.model, None, self.configs.model_save_path, "latest.pth")

    def plot_losses(self, epoch):
        """
        Plots the loss curves and saves the figure.

        Parameters:
            epoch (int): The current epoch number.
        """
        plt.figure(figsize=(10, 6))
        for loss_name, loss_values in self.loss_history.items():
            plt.plot(range(1, epoch + 1), loss_values, label=loss_name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(self.plot_dir, f"loss_epoch_{epoch}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved loss plot for epoch {epoch} at {plot_path}")


def worker_function(config_file, gpu_id, checkpoint_path=None):
    """
    Worker function to initialize and start training.

    Parameters:
        config_file (str): Path to the configuration file.
        gpu_id (list): List of GPU IDs to use.
        checkpoint_path (str, optional): Path to a checkpoint to resume training from.
    """
    print("Using GPU IDs:", ",".join([str(i) for i in gpu_id]))
    trainer = WorkerFunction(config_file, gpu_id, checkpoint_path)
    trainer.train()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    worker_function("./openlane_config.py", gpu_id=[0])
