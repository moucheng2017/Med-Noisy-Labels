import torch
import torch.nn as nn
import torch.nn.functional as F
# =============================

def contrastive_loss(Fxlr, label, margin=0.05):
    #
    contrastive_loss = 1e-8
    #
    for index, fxlr in enumerate(Fxlr):
        #
        if fxlr == Fxlr[-1]:
            euclidean_dist = torch.pow(fxlr - Fxlr[0], 2)
        else:
            euclidean_dist = torch.pow(fxlr - Fxlr[index + 1], 2)
        #
        loss_ = torch.mean((1 - label) * torch.pow(euclidean_dist, 2) +
                           (label) * torch.pow(torch.clamp(margin - euclidean_dist, min=0.0), 2))
        #
        contrastive_loss += loss_
    #
    return max(1 / contrastive_loss, 1e-8)