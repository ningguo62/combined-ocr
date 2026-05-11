# %%
import os
import shutil
import warnings
import platform
from glob import glob
from itertools import chain
from collections import Counter
from dataclasses import dataclass
 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
# from sklearn.utils.class_weight import compute_class_weight
 
# Next, we have our usual torch and torchvision imports.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
 
import torchvision
import torchvision.transforms as TF
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.ops import sigmoid_focal_loss


def resize_and_pad_collate(batch, target_height=64):
    """
    batch: list of tuples (image_tensor, label)
    image_tensor: expected shape (C, H, W)
    """
    resized_images = []
    labels = []

    for img, label in batch:
        # 1. Calculate new width to maintain aspect ratio
        _, h, w = img.shape
        scale_factor = target_height / h
        target_width = int(w * scale_factor)

        # 2. Resize to target height (requires 4D input: B x C x H x W)
        img_resized = F.interpolate(
            img.unsqueeze(0), 
            size=(target_height, target_width), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)

        resized_images.append(img_resized)
        labels.append(label)

    # 3. Find max width in this specific batch for padding
    max_batch_width = max(img.shape[2] for img in resized_images)

    # 4. Pad the width of each image to match max_batch_width
    padded_images = []
    for img in resized_images:
        current_width = img.shape[2]
        pad_width = max_batch_width - current_width
        
        # F.pad expects (left, right, top, bottom) for the last two dims
        padded_img = F.pad(img, (0, pad_width, 0, 0), mode='constant', value=0)
        padded_images.append(padded_img)

    return torch.stack(padded_images), torch.tensor(labels)

device = 'cuda:0'
#device = 'cpu'
batch_size = 16
num_epochs = 100
transform = TF.Compose([TF.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root='./data/train/single', transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root='./data/valid/single', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=resize_and_pad_collate, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=resize_and_pad_collate, shuffle=True)

network = torchvision.models.efficientnet_b3().to(device)

criterion = torch.nn.CrossEntropyLoss()
# Freeze original weights if you only want to train the new head
#for param in network.parameters():
#    param.requires_grad = False

# Replace the classifier (usually the last layer in model.classifier)
num_ftrs = network.classifier[1].in_features
network.classifier = torch.nn.Linear(num_ftrs, 10).to(device) # Example: 10 classes

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.1, patience=3)

accuracy_max = 0
for epoch in range(num_epochs):
    network.train()
    loss_all = 0
    for inputs, labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = network(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    print(loss_all)

    network.eval()  # Set to evaluation mode
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            labels = labels.to(device)
            outputs = network(inputs.to(device))
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    print(f'Val Loss: {avg_loss:.4f} | Val Acc: {accuracy:.2f}%')
    if accuracy > accuracy_max:
        print(f"At epoch {epoch}, accuracy is {accuracy:.4f}%, better than the record of {accuracy_max:.4f}%")
        torch.save(network.state_dict(), 'best_classifier.pth')
        accuracy_max = accuracy
    scheduler.step(accuracy)

# %%
