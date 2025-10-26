import torch
import torch.nn as nn
import torch.nn.functional as F

class SP(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''
    def __init__(self):
        super(SP, self).__init__()
    
    def forward(self, fm_s, fm_t):
        B = fm_s.shape[0]
        fm_s = fm_s.view(B, -1)
        m_s = torch.mm(fm_s, fm_s.t())
        norm_s = F.normalize(m_s, p=2, dim=1)

        fm_t = fm_t.view(B, -1)
        m_t = torch.mm(fm_t, fm_t.t())
        norm_t = F.normalize(m_t, p=2, dim=1)
        
        loss = F.mse_loss(norm_s, norm_t)
        return loss


class AT(nn.Module):
    '''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Networks via Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
    def __init__(self, loss_weight=1, p=2, eps=1e-6):
        super(AT, self).__init__()
        self.loss_weight = loss_weight
        self.p = p
        self.eps = eps
    
    def forward(self, fm_s, fm_t):
        # Get attention maps
        s_attention = self.attention_map(fm_s)
        t_attention = self.attention_map(fm_t)
        
        loss =F.mse_loss(s_attention, t_attention)

        loss = loss*self.loss_weight
        return loss
    
    def attention_map(self, fm):
        """
        Compute normalized attention map
        """
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)  # (B, 1, H, W)
        
        B, _, H, W = fm.shape
        am = am.reshape(B, -1)
        am = F.normalize(am, p=2, dim=1, eps=self.eps)
        am = am.reshape(B, 1, H, W)

        return am
    
class PRloss(nn.Module):
    '''
    Understanding the Role of the Projector in Knowledge Distillation
    https://arxiv.org/pdf/2303.11098
    '''
    def __init__(self):
        super(PRloss, self).__init__()
        self.proj = nn.Conv2d(in_channels=11, out_channels=11, kernel_size=1, bias=False)
        self.bn_s = nn.BatchNorm2d(11, eps=0.0001, affine=False)
        self.bn_t = nn.BatchNorm2d(11, eps=0.0001, affine=False)

    def forward(self, fm_s, fm_t):
        fm_s = self.bn_s(self.proj(fm_s))
        fm_t = self.bn_t(fm_t)

        c_diff = fm_s - fm_t
        c_diff = torch.abs(c_diff)
        c_diff = c_diff.pow(4.0)

        loss = c_diff.view(c_diff.size(0), -1).sum(dim=1)
        loss = torch.log(loss+0.000001).mean()
        return loss

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