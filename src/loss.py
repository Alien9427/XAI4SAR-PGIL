import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


class SoftConstraintLoss(nn.Module):
    def __init__(self, nonzero_prob):
        super(SoftConstraintLoss, self).__init__()
        self.nonzero_prob = nonzero_prob #
        return

    def forward(self, feat, attr):
        loss_mask = torch.zeros(size=attr.shape)
        with torch.no_grad():
            for kk in range(attr.shape[0]):
                nonzeros = torch.nonzero(attr[kk, ].cpu() >= 0.1)[:, 0]  # alpha = 0.1
                nonzeros_sel_ind = np.random.choice(nonzeros, int(round(len(nonzeros) * self.nonzero_prob)), False)
                loss_mask[kk, nonzeros_sel_ind] = 1

        loss = -attr * loss_mask.cuda() * func.log_softmax(feat, 1)
        return loss.sum(axis=1).mean()