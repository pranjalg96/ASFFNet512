import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19

import os
import cv2
import numpy as np

from dataset.BFRestoreData import BFRestoreDataset
from torch.utils.data import DataLoader

from utils.train_utils import get_lmarks_from_tensor
from utils.WLS import guidance_selection
from models.ASFFNet import ASFFNet

from torchvision.transforms.functional import normalize
import face_alignment


# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the loss functions
mse_loss = nn.MSELoss()
# vgg_model = vgg19(pretrained=True).features[:35].to(device).eval()
# for param in vgg_model.parameters():
#     param.requires_grad = False

# Define the perceptual loss function
# def perceptual_loss(fake, real):
#     loss = 0
#     for layer in [1, 6, 11, 20]:
#         fake_feat = vgg_model[:layer](fake)
#         real_feat = vgg_model[:layer](real)
#         loss += mse_loss(fake_feat, real_feat)
#     return loss

# # Define the style loss function
# def gram_matrix(x):
#     b, c, h, w = x.size()
#     features = x.view(b, c, h * w)
#     gram = torch.bmm(features, features.transpose(1, 2))
#     gram = gram / (c * h * w)
#     return gram

# def style_loss(fake, real):
#     loss = 0
#     for layer in [1, 6, 11, 20]:
#         fake_feat = vgg_model[:layer](fake)
#         real_feat = vgg_model[:layer](real)
#         fake_gram = gram_matrix(fake_feat)
#         real_gram = gram_matrix(real_feat)
#         loss += mse_loss(fake_gram, real_gram)
#     return loss

# Training parameters
batch_size = 1
lr = 2e-4
num_epochs = 1
checkpoint_interval = 10
lambda_mse = 1.0
lambda_perc = 1.0
lambda_style = 0.001
lambda_adv = 0.001

# Create data loader
train_dir = 'TrainExamples'

train_dataset = BFRestoreDataset(data_dir=train_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Define the ASFFNet model
asffnet = ASFFNet().to(device)

# Define optimizers
optimizer_G = optim.Adam(asffnet.parameters(), lr=lr)
optimizer_D = optim.Adam(asffnet.parameters(), lr=lr)

# Load face detection module
FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

# Start training
for epoch in range(num_epochs):
    for i, (lq_image_batch, hq_images_batch, hq_lmarks_batch, gt_image_batch) in enumerate(train_loader):

        lq_image, hq_images, hq_lmarks, gt_image = lq_image_batch[0], hq_images_batch[0], hq_lmarks_batch[0], gt_image_batch[0]

        # lq image landmark detection
        lq_lmarks = get_lmarks_from_tensor(lq_image, FaceDetection)
        if lq_lmarks is None:
            print(f'No landmarks found in lq image of training example {i}. Skipping...')
            continue

        # Guidance selection
        hq_selected_idx = guidance_selection(lq_lmarks, hq_lmarks)

        hq_selected_image = hq_images_batch[:, hq_selected_idx]
        hq_selected_lmark = hq_lmarks_batch[:, hq_selected_idx]

#         lq_imgs, id_imgs, lq_landmarks, id_landmarks = lq_imgs.to(device), id_imgs.to(device), lq_landmarks.to(device), id_landmarks.to(device)

#         # Forward pass
#         fake_id_imgs = asffnet(lq_imgs, id_imgs, lq_landmarks, id_landmarks)

#         # Compute reconstruction loss
#         mse_loss_val = mse_loss(fake_id_imgs, id_imgs)
#         perc_loss_val = perceptual_loss(fake_id_imgs, id_imgs)
#         reconstruction_loss = lambda_mse * mse_loss_val + lambda_perc * perc_loss_val

#         # Compute photo-realistic loss
#         style_loss_val = style_loss(fake_id_imgs, id_imgs)
#         adv_loss_val = -torch.mean(asffnet(id_imgs, id_imgs, id_landmarks, id_landmarks))

#         photo_realistic_loss = lambda_style * style_loss_val + lambda_adv * adv_loss_val

#         # Compute the overall loss
#         total_loss = reconstruction_loss + photo_realistic_loss

#         # Backpropagation
#         optimizer_G.zero_grad()
#         total_loss.backward()
#         optimizer_G.step()

#         # Print progress
#         if i % 10 == 0:
#             print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(train_loader)}] Loss: {total_loss.item()}")

#     # Save model checkpoints
#     if (epoch + 1) % checkpoint_interval == 0:
#         checkpoint_path = f"asffnet_epoch_{epoch + 1}.pth"
#         torch.save(asffnet.state_dict(), checkpoint_path)
#         print(f"Saved checkpoint at {checkpoint_path}")

# print("Training completed.")
