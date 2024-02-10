import torch
import torch.nn as nn
import torch.nn.functional as F

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.bn = nn.BatchNorm1d(ft_in)

    def forward(self, seq):
        ret = self.fc(self.bn(seq))
        return ret
