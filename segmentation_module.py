import torch
import torch.nn as nn
from torch import distributed
import torch.nn.functional as functional

from functools import partial, reduce


from model.IncrementalBiSeNet import IncrementalBiSeNet


def make_model(opts, classes=None):
    
    norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex

    model = IncrementalBiSeNet(classes=classes, context_path=opts.backbone)

    return model
