import torch.nn as nn
import torch

def tversky(output, target, alpha = 0.7, smooth = 1):
    output = torch.nn.functional.sigmoid(output)
    output = torch.flatten(output)
    target = torch.flatten(target)
    true_pos = torch.sum(target * output)
    false_neg = torch.sum(target * (1-output))
    false_pos = torch.sum((1-target) * output)
    return (true_pos + smooth) / (true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tverskyşloss(output, target):
    return 1-tversky(output, target)

def FocalTversky(output, target, gamma = 0.75):
    tv = tversky(output, target)
    return torch.pow((1-tv), gamma)

class FocalTverskyLoss(nn.Module):
    def __init__(self, gamma = 0.75):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, output, target):
        loss = FocalTversky(output, target, self.gamma)
        return loss