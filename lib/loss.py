import torch
import torch.nn as nn
from torch.nn import functional as F


class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, temperature=1.0, use_softmax = True):
        super(CriterionKD, self).__init__()
        self.temperature = temperature
        self.criterion_kd = torch.nn.KLDivLoss()
        self.use_softmax = use_softmax

    def forward(self, pred, soft):

        soft.detach()
        scale_pred = pred
        scale_soft = soft
        if self.use_softmax:
            loss = self.criterion_kd(F.log_softmax(scale_pred / self.temperature, dim=1), F.softmax(scale_soft / self.temperature, dim=1))
        else:
            loss = self.criterion_kd((scale_pred / self.temperature), (scale_soft / self.temperature))
        return loss