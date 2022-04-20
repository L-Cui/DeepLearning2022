#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:19:36 2022

@author: leilei
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T
import torchvision.transforms as TorchTransform
import utils
from engine import train_one_epoch_sup, train_one_epoch_unsup, evaluate
import FeatureExtractor as FcExt


from dataset import UnlabeledDataset, LabeledDataset
import matplotlib.pyplot as plt
import numpy as np

def imshow(img: torch.Tensor):
    """
    Display a single image.
    """

    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_transform(sup=True):
    transforms = []
    
    if sup: # for supervised learning 
        transforms.append(T.ToTensor())
        # transforms.append(TorchTransform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)    
    else:
        transforms.append(TorchTransform.ToTensor())
        transforms.append(TorchTransform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return TorchTransform.Compose(transforms)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train_dataset_unsup = UnlabeledDataset(root='./data/unlabeled_data', transforms=get_transform(sup=False))
    train_loader_unsup = torch.utils.data.DataLoader(train_dataset_unsup, batch_size=2, shuffle=True, num_workers=2)
    
    n_unsup_classes = len(train_dataset_unsup.sesemi_transforms.classes)
    backbone = torchvision.models.resnet50(pretrained=False)
    model_unsup = FcExt.UnsuperModel(n_unsup_classes, backbone)
    model_unsup.to(device)

    params = [p for p in model_unsup.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criteria = nn.CrossEntropyLoss()
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 50 iterations
        train_one_epoch_unsup(model_unsup, optimizer, criteria, train_loader_unsup, device, epoch, print_freq=200)
        # update the learning rate
        lr_scheduler.step()

    train_dataset = LabeledDataset(root='./data/labeled', split="training", transforms=get_transform(sup=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='./data/labeled', split="validation", transforms=get_transform(sup=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)
    
    n_sup_classess = 100
    model_sup = FcExt.BackboneToFastRcnn(n_sup_classess, model_unsup.backbone, trainable_backbone_layers=2)
    model_sup.to(device)
    for epoch in range(num_epochs):
        # train for one epoch, printing every 50 iterations
        train_one_epoch_sup(model_sup, optimizer, train_loader, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model_sup, valid_loader, device=device)       
    
    print("That's it!")
    
if __name__=="__main__":
    main()