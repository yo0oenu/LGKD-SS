import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_gram_matrix(features):
    """
    Compute Channel similarity matrix from features map
    Args:
        features: (B, C, H, W)
    Returns:
        gram_matrix: (B, C, C)
    """
    B, C, H, W = features.shape
    features = features.view(B, C, H*W)  # (B, C, H*W)
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = F.normalize(gram, p=2, dim=2)

    
    #features = F.normalize(features, p=2, dim=2)
    #gram = torch.bmm(features, features.transpose(1, 2))  #(B, C, C)
    return gram

def gram_matrix_loss(student_features, teacher_features, reduction='mean'):
    student_gram = compute_gram_matrix(student_features)
    teacher_gram = compute_gram_matrix(teacher_features)

    loss = F.mse_loss(student_gram, teacher_gram, reduction=reduction)
    return loss

class GramMatrixLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GramMatrixLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, student_features, teacher_features):
        loss = gram_matrix_loss(student_features, teacher_features, reduction=self.reduction)
        return loss*self.loss_weight