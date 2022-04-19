#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:06:07 2022

@author: leilei
"""

import random
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers, _mobilenet_extractor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
class FeatureExtractor(nn.Module):
    """
    ResNet backbone with two heads for SESEMI training.
    """

    def __init__(
        self,
        n_sup_classes: int,
        n_unsup_classes: int,
        pretrained: bool = True
    ):
        super().__init__()
        self.stem = torchvision.models.resnet50(pretrained=pretrained)
        
        self.fc_out = 256
        self.stem.fc = nn.Linear(self.stem.fc.in_features, self.fc_out)
        self.sup_fc = nn.Linear(self.fc_out, n_sup_classes)
        self.selfsup_fc = nn.Linear(self.fc_out, n_unsup_classes)

    def forward(self, x: torch.Tensor, x_selfsup: Optional[torch.Tensor] = None):
        x = self.stem(x)
        x = self.sup_fc(x)
        if x_selfsup is not None:
            x_selfsup = self.stem(x_selfsup)
            x_selfsup = self.selfsup_fc(x_selfsup)
            return x, x_selfsup
        else:
            return x
        
class UnsuperModel(nn.Module):
    def __init__(self, n_unsup_classes: int, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc_out = 512
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.fc_out)
        self.unsup_fc = nn.Linear(self.fc_out, n_unsup_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.unsup_fc(x)
        
        return x
        
def BackboneToFastRcnn(n_sup_classess: int, backbone, trainable_backbone_layers=None):
    trainable_backbone_layers = _validate_trainable_layers(True, trainable_backbone_layers, 5, 3)  
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, n_sup_classess)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_sup_classess)   
    
    return model

# class SuperModel(nn.Module):
#     def __init__(
#             self, 
#             n_sup_classess: int, 
#             backbone, 
#             trainable_backbone_layers=None):
#         super().__init__()
#         self.trainable_backbone_layers = _validate_trainable_layers(
#             True, trainable_backbone_layers, 5, 3)        
#         self.backbone = backbone
#         self.backbone = _resnet_fpn_extractor(self.backbone, trainable_backbone_layers)
#         self.FasterRCNN = FasterRCNN(self.backbone, n_sup_classess)
#         in_features = self.FasterRCNN.roi_heads.box_predictor.cls_score.in_features
#         self.FasterRCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_sup_classess)
    
#     def forward(self, imgs, targets):
        