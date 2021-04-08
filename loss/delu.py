import torch
import numpy as np

def bwLoss(expls, gts):
    cntZero = 0
    cntOne = 0
    remember = []
    zero = torch.tensor([0.0]).double()
    for gt, expl in zip(gts, expls):
        for zero_xy in (gt == 0).nonzero().tolist():
            row, column = zero_xy
            result = torch.max(zero, (expl[row,column] - 0.5))
            cntZero += result
        for one_xy in (gt == 255).nonzero().tolist():
            row, column = one_xy
            result = torch.max(zero, (0.9 - expl[row,column])*torch.tensor([14.0]))
            cntOne += result
        output = (cntZero + cntOne) / (gt.shape[0]*gt.shape[1])
        remember.append(output)
        cntZero = 0
        cntOne = 0
    summation = sum(remember)
    loss = summation / len(gts)
    return loss
