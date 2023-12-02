import torch.nn as nn
import torch
from config import CFG

def criterion(logits, targets, activated=False):
    if activated:
        losses = nn.BCELoss(reduction='none')(logits.view(-1), targets.view(-1))
    else:
        losses = CFG.bce(logits.view(-1), targets.view(-1))
    losses[targets.view(-1) > 0] *= 2.
    norm = torch.ones(logits.view(-1).shape[0]).to(CFG.device)
    norm[targets.view(-1) > 0] *= 2
    return losses.sum() / norm.sum()
