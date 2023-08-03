import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize
import numpy as np

class BFRestoreDataset(Dataset):
    def __init__(self, data_dir):
        """
        Initialize the dataset
        :param: data_dir - Directory that contains the dataset

        """
        self.data_dir = data_dir
        self.image_list = self._load_image_list()

    def _load_image_list(self):
        """
        Load a list of example folders from the dataset

        returns - A list of example folders from the dataset
        """
        image_list = []
        for example_folder in os.listdir(self.data_dir):
            example_folder_path = os.path.join(self.data_dir, example_folder)
            if os.path.isdir(example_folder_path):
                image_list.append(example_folder)

        return image_list

    def _load_image(self, folder_path, image_type):
        """
        Load image and preprocess to RGB with pixel values between -1 to 1
        :param: folder_path - Path to an example folder in train dataset
        :param: image_type - Type of image to load (lq or gt)

        returns - A preprocessed batch of image tensor
        """

        folder_type = os.path.join(folder_path, image_type)
        img_name = os.listdir(folder_type)[0]
        image_path = os.path.join(folder_type, img_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = normalize(torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        return image


    def _load_image_and_landmarks(self, folder_path):
        """
        Load image and corresponding landmarks from folder and return as a torch tensor. The landmarks
        of each image is in a txt file with same name as the image
        :param: folder_path - Path to an example folder in train dataset

        returns - Batch of tensors of preprocessed images and numpy array of corresponding landmarks
        """

        images = []
        landmarks = []

        hq_folder_path = os.path.join(folder_path, "hq")
        img_names = os.listdir(hq_folder_path)

        for img_name in img_names:
            # load image
            image_path = os.path.join(hq_folder_path, img_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            image = normalize(torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            images.append(image)

            # load landmarks
            landmarks_path = os.path.join(folder_path, "hq_lmarks", f"{img_name}.txt")
            with open(landmarks_path, "r") as file:
                lines = file.readlines()

            img_lmarks = []
            for line in lines:
                x, y = line.split(' ')
                x, y = float(x), float(y)
                img_lmarks.append([x, y])

            np_img_lmarks = np.array(img_lmarks)
            landmarks.append(np_img_lmarks)

        batch_images = torch.stack(images, dim=0)
        batch_landmarks = np.array(landmarks)

        return batch_images, batch_landmarks

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        example_folder = self.image_list[idx]
        example_folder_path = os.path.join(self.data_dir, example_folder)

        # Load low-quality (degraded) image
        lq_image = self._load_image(example_folder_path, "lq")

        # Load high-quality (guidance) image
        hq_images, hq_landmarks = self._load_image_and_landmarks(example_folder_path)

        # Load ground truth restored image
        gt_image = self._load_image(example_folder_path, "gt")

        return lq_image, hq_images, hq_landmarks, gt_image
