import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillationLoss(nn.Module):
    def __init__(self, device='cuda', lambda_mse=1.0, lambda_cos=0.5):
        super(FeatureDistillationLoss, self).__init__()
        self.device = device
        self.lambda_mse = lambda_mse
        self.lambda_cos = lambda_cos
        self.mse_loss = nn.MSELoss().to(self.device)
        self.cosine_loss = nn.CosineEmbeddingLoss().to(self.device)

    def forward(self, student_features_s32, student_features_s64, teacher_feature_s32, teacher_feature_s64):

        # Calculate MSE Loss
        loss_feat_s32_mse = self.mse_loss(student_features_s32, teacher_feature_s32)
        loss_feat_s64_mse = self.mse_loss(student_features_s64, teacher_feature_s64)

        # Flatten feature maps for cosine similarity loss
        student_s32_flat = student_features_s32.view(student_features_s32.size(0), student_features_s32.size(1), -1)
        teacher_s32_flat = teacher_feature_s32.view(teacher_feature_s32.size(0), teacher_feature_s32.size(1), -1)
        student_s64_flat = student_features_s64.view(student_features_s64.size(0), student_features_s64.size(1), -1)
        teacher_s64_flat = teacher_feature_s64.view(teacher_feature_s64.size(0), teacher_feature_s64.size(1), -1)

        # Define separate targets for s32 and s64
        target_s32 = torch.ones(student_s32_flat.size(0) * student_s32_flat.size(2)).to(self.device)
        target_s64 = torch.ones(student_s64_flat.size(0) * student_s64_flat.size(2)).to(self.device)

        # Calculate Cosine Similarity Loss for s32
        loss_feat_s32_cos = self.cosine_loss(
            student_s32_flat.permute(0, 2, 1).reshape(-1, student_s32_flat.size(1)),
            teacher_s32_flat.permute(0, 2, 1).reshape(-1, teacher_s32_flat.size(1)),
            target_s32
        )

        # Calculate Cosine Similarity Loss for s64
        loss_feat_s64_cos = self.cosine_loss(
            student_s64_flat.permute(0, 2, 1).reshape(-1, student_s64_flat.size(1)),
            teacher_s64_flat.permute(0, 2, 1).reshape(-1, teacher_s64_flat.size(1)),
            target_s64
        )

        # Calculate total distillation loss
        distillation_loss = (
            self.lambda_mse * (loss_feat_s32_mse + loss_feat_s64_mse) +
            self.lambda_cos * (loss_feat_s32_cos + loss_feat_s64_cos)
        )

        return distillation_loss
