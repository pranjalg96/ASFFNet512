import argparse
import torch
import numpy as np

from dataset.BFRestoreData import BFRestoreDataset
from torch.utils.data import DataLoader

from utils.train_utils import get_lmarks_from_tensor, generate_lq_point_mask
from utils.WLS import guidance_selection
from utils.losses import mse_loss, perceptual_loss, style_loss

from models.ASFFNet import ASFFNet
from models.SNGAN_disc import SNGANDiscriminator

import face_alignment

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', type=str, default='./TrainExamples', help='input path of dataset')
parser.add_argument('-s', '--save_dir', type=str, default='./saved_models', help='save path of models')
args = parser.parse_args()

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
batch_size = 1
asff_lr = 2e-4
disc_lr = 2e-4
num_epochs = 10
report_interval = 2
checkpoint_interval = 5

# Loss coefficients
lambda_mse = 10.0
lambda_perc = 5.0
lambda_style = 2.0
lambda_adv = 1.0

# Create data loader
train_dir = args.input_path

train_dataset = BFRestoreDataset(data_dir=train_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Define the ASFFNet model
ASFFNet512 = ASFFNet().to(device).train()

# Define the SNGAN discriminator model (For real images, output should be close to 1. For fake images, output should be close to -1)
SNGANDisc = SNGANDiscriminator().to(device).train()

# Define optimizers
asff_opt = torch.optim.Adam(ASFFNet512.parameters(), lr=asff_lr, betas=(0.5, 0.999))
sngan_disc_opt = torch.optim.Adam(SNGANDisc.parameters(), lr=disc_lr, betas=(0.5, 0.999))

# Load face detection module
FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

# Start training
for epoch in range(num_epochs):
    for i, (lq_image_batch, hq_images_batch, hq_lmarks_batch, gt_image_batch) in enumerate(train_loader):
        # Extract items from batch
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

        # Generate Landmark mask from lq landmarks
        lq_landmarks_mask = generate_lq_point_mask(lq_lmarks)

        # Move all tensors to device
        lq_image_batch, hq_selected_image, hq_selected_lmark, gt_image_batch, lq_landmarks_mask = lq_image_batch.to(device), hq_selected_image.to(device), hq_selected_lmark.to(device), gt_image_batch.to(device), lq_landmarks_mask.to(device)
        lq_lmarks = torch.from_numpy(lq_lmarks).unsqueeze(0).to(device)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Training ASFFNet
        asff_opt.zero_grad()

        # Forward pass through ASFFNet
        restored_img_batch = ASFFNet512(lq_image_batch, hq_selected_image, lq_landmarks_mask, lq_lmarks, hq_selected_lmark)

        mse_loss_val = mse_loss(restored_img_batch, gt_image_batch)
        perc_loss_val = perceptual_loss(restored_img_batch, gt_image_batch)
        
        # Compute reconstruction loss
        reconstruction_loss = lambda_mse * mse_loss_val + lambda_perc * perc_loss_val

        style_loss_val = style_loss(restored_img_batch, gt_image_batch)
        fake_logits = SNGANDisc(restored_img_batch)
        adv_loss_val = -torch.mean(fake_logits)

        # Compute photo-realistic loss
        photo_realistic_loss = lambda_style * style_loss_val + lambda_adv * adv_loss_val

        # Compute the overall loss
        total_asffnet_loss = reconstruction_loss + photo_realistic_loss

        # Backpropagation through ASFFNet
        total_asffnet_loss.backward()
        asff_opt.step()
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Training SNGAN Discriminator
        sngan_disc_opt.zero_grad()

        # Forward pass through ASFFNet. Detach the tensor after since only discriminator is trained
        restored_img_batch = ASFFNet512(lq_image_batch, hq_selected_image, lq_landmarks_mask, lq_lmarks, hq_selected_lmark).detach()

        fake_logits = SNGANDisc(restored_img_batch)
        real_logits = SNGANDisc(gt_image_batch)

        # Hinge version of adversarial loss for discriminator
        d_loss_real = torch.mean(torch.nn.ReLU()(1.0 - real_logits))
        d_loss_fake = torch.mean(torch.nn.ReLU()(1.0 + fake_logits))
        d_total_loss = d_loss_real + d_loss_fake

        # Backpropogation through SNGAN Discriminator
        d_total_loss.backward()
        sngan_disc_opt.step()
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Print progress
        if (i + 1) % report_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(train_loader)}] ASFF_Loss: {total_asffnet_loss.item()} (MSE_Loss: {mse_loss_val.item()} Perc_Loss: {perc_loss_val.item()} Style_Loss: {style_loss_val.item()} Adv_Loss: {adv_loss_val.item()}) Disc_Loss: {d_total_loss.item()}")

    # Save model checkpoints
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path_asff = f"{args.save_dir}/asffnet_adv_epoch_{epoch + 1}.pth"
        checkpoint_path_disc = f"{args.save_dir}/sngan_disc_epoch_{epoch + 1}.pth"

        torch.save(ASFFNet512.state_dict(), checkpoint_path_asff)
        torch.save(SNGANDisc.state_dict(), checkpoint_path_disc)

        print(f"Saved checkpoints at {checkpoint_path_asff}, {checkpoint_path_disc}")

print("Training completed.")
