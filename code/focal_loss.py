import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)  
        probs = torch.exp(log_probs)              

        targets = targets.view(-1)               
        one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()  

        pt = (probs * one_hot).sum(dim=1)         
        log_pt = (log_probs * one_hot).sum(dim=1) 
        focal_weight = (1 - pt) ** self.gamma     

        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha_t = torch.tensor(self.alpha).to(inputs.device)[targets]  
            else:
                alpha_t = self.alpha
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
