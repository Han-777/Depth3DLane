import torch
import torch.nn as nn
# import pytorch_ssim
from .SiLogLoss import SiLogLoss
from .pytorch_ssim import SSIM
"""
loss for depth auxiliary head 


"""
class GradientLoss(nn.Module):
    """
    Computes the gradient loss between two depth maps.
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Define Sobel kernels for gradient computation
        self.grad_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.grad_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # # Initialize Sobel kernels
        # sobel_x = torch.tensor([[[[-1, 0, 1],
        #                           [-2, 0, 2],
        #                           [-1, 0, 1]]]], dtype=torch.float32)
        # sobel_y = torch.tensor([[[[-1, -2, -1],
        #                           [0, 0, 0],
        #                           [1, 2, 1]]]], dtype=torch.float32)
        #
        # # Assign Sobel kernels to convolution layers
        # self.grad_x.weight = nn.Parameter(sobel_x, requires_grad=False)
        # self.grad_y.weight = nn.Parameter(sobel_y, requires_grad=False)

        # # Define L1 loss for gradients
        # self.l1 = nn.L1Loss()

        # Initialize Scharr kernels
        scharr_x = torch.tensor([[[[-3, 0, 3],
                                   [-10, 0, 10],
                                   [-3, 0, 3]]]], dtype=torch.float32)
        scharr_y = torch.tensor([[[[-3, -10, -3],
                                   [0, 0, 0],
                                   [3, 10, 3]]]], dtype=torch.float32)
        # Assign Scharr kernels to convolution layerss
        self.grad_x.weight = nn.Parameter(scharr_x, requires_grad=False)
        self.grad_y.weight = nn.Parameter(scharr_y, requires_grad=False)

        self.huber = nn.SmoothL1Loss()


    def forward(self, pred, target):
        # Compute gradients for predictions and targets
        pred_grad_x = self.grad_x(pred)
        pred_grad_y = self.grad_y(pred)
        target_grad_x = self.grad_x(target)
        target_grad_y = self.grad_y(target)

        # # Compute L1 loss between gradients
        # loss_x = self.l1(pred_grad_x, target_grad_x)
        # loss_y = self.l1(pred_grad_y, target_grad_y)

        # Compute Huber loss between gradients
        loss_x = self.huber(pred_grad_x, target_grad_x)
        loss_y = self.huber(pred_grad_y, target_grad_y)

        return loss_x + loss_y


class CombinedDepthLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.2, gamma=0.05, delta=1): # 这里的可以根据效果修改，三种损失函数的比例
        """
        Initializes the CombinedDepthLoss.

        Parameters:
        - alpha (float): Weight for L1 loss.
        - beta (float): Weight for SSIM loss.
        - gamma (float): Weight for Gradient loss.
        """
        super(CombinedDepthLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        # self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        # self.gradient_loss = GradientLoss()
        self.SilogLoss = SiLogLoss()

    def forward(self, pred, target):
        """
        Computes the combined depth loss.

        Parameters:
        - pred (torch.Tensor): Predicted depth map (B, 1, H, W).
        - target (torch.Tensor): Ground truth depth map (B, 1, H, W).

        Returns:
        - torch.Tensor: Combined depth loss.
        """
        # Ensure the depth maps are in the same range
        pred = torch.sigmoid(pred)  # Normalize to [0, 1] if not already
        target = torch.sigmoid(target)
        valid_mask = target > 0
        loss_silog = self.SilogLoss(pred, target, valid_mask)
        # pred = torch.sigmoid(pred)  # Normalize to [0, 1] if not already
        # target = torch.sigmoid(target)
        # Compute individual loss components
        # loss_l1 = self.l1_loss(pred, target)
        loss_ssim = 1 - self.ssim_loss(pred, target)
        # loss_grad = self.gradient_loss(pred, target)

        # Combine losses with respective weights
        combined_loss = self.beta * loss_ssim + self.delta * loss_silog
        # combined_loss = self.delta * loss_silog
        return combined_loss


