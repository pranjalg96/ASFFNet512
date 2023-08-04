import argparse
import torch
import numpy as np

from dataset.BFRestoreData import BFRestoreDataset
from torch.utils.data import DataLoader

from utils.train_utils import get_lmarks_from_tensor, generate_lq_point_mask
from utils.WLS import guidance_selection
from models.ASFFNet import ASFFNet

from utils.losses import mse_loss, perceptual_loss, style_loss

from torchvision.transforms.functional import normalize
import face_alignment

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', type=str, default='./TrainExamples', help='input path of dataset')
parser.add_argument('-s', '--save_dir', type=str, default='./saved_models', help='save path of restoration result')
args = parser.parse_args()

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
batch_size = 1
lr = 2e-4
num_epochs = 10
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

# Define optimizers
asff_opt = torch.optim.Adam(ASFFNet512.parameters(), lr=lr)
# optimizer_D = optim.Adam(ASFFNet512.parameters(), lr=lr)

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

        # Forward pass
        restored_img_batch = ASFFNet512(lq_image_batch, hq_selected_image, lq_landmarks_mask, lq_lmarks, hq_selected_lmark)

        # Compute reconstruction loss
        mse_loss_val = mse_loss(restored_img_batch, gt_image_batch)
        perc_loss_val = perceptual_loss(restored_img_batch, gt_image_batch)

        reconstruction_loss = lambda_mse * mse_loss_val + lambda_perc * perc_loss_val

        # Compute photo-realistic loss
        style_loss_val = style_loss(restored_img_batch, gt_image_batch)
#         adv_loss_val = -torch.mean(asffnet(id_imgs, id_imgs, id_landmarks, id_landmarks))

        photo_realistic_loss = lambda_style * style_loss_val  # + lambda_adv * adv_loss_val

        # Compute the overall loss
        total_loss = reconstruction_loss + photo_realistic_loss

        # Backpropagation
        asff_opt.zero_grad()
        total_loss.backward()
        asff_opt.step()

        # Print progress
        if (i + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(train_loader)}] Loss: {total_loss.item()} MSE_Loss: {mse_loss_val.item()} Perc_Loss: {perc_loss_val.item()} Style_Loss: {style_loss_val.item()}")

    # Save model checkpoints
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = f"{args.save_dir}/asffnet_epoch_{epoch + 1}.pth"
        torch.save(ASFFNet512.state_dict(), checkpoint_path)
        print(f"Saved checkpoint at {checkpoint_path}")

print("Training completed.")
