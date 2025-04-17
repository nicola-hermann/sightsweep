from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable
import cv2
from copy import deepcopy
import numpy as np
import random
import torch


class SightSweepDataset(Dataset):
    def __init__(
        self,
        data_folder: Path,
        augmentation_fn: Callable = None,
        min_blobs: int = 1,
        max_blobs: int = 4,
    ):
        self.data_folder = data_folder
        self.augmentation_fn = augmentation_fn
        self.data_paths = self.get_data_paths()
        self.min_blobs = min_blobs
        self.max_blobs = max_blobs

    def get_data_paths(self):
        # Glob for all jpg files in the data folder
        data_paths = list(self.data_folder.rglob("*.jpg"))
        if not data_paths:
            raise ValueError(f"No jpg files found in {self.data_folder}")
        return data_paths

    def apply_random_mask(self, image: cv2.Mat) -> tuple[cv2.Mat, cv2.Mat]:
        # Create a random ellipse mask
        height, width = image.shape[:2]
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)  # Initialize with white

        num_blobs = random.randint(self.min_blobs, self.max_blobs)

        for _ in range(num_blobs):
            c_x = random.randint(width // 10, 9 * width // 10)
            c_y = random.randint(height // 10, 9 * height // 10)
            axes_x = random.randint(min(width, height) // 16, min(width, height) // 5)
            axes_y = random.randint(min(width, height) // 16, min(width, height) // 5)
            angle = random.randint(0, 180)  # Angle of rotation of the ellipse

            temp_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.ellipse(temp_mask, (c_x, c_y), (axes_x, axes_y), angle, 0, 360, 255, -1)
            binary_blob_inverted = cv2.bitwise_not(temp_mask)
            mask = cv2.bitwise_and(mask, binary_blob_inverted)

        masked_image = image.copy()
        masked_image[mask == 0] = 0

        return masked_image, mask

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load the image
        img_path = self.data_paths[idx]
        image = cv2.imread(str(img_path))
        label = deepcopy(image)

        blob_img, mask = self.apply_random_mask(image)

        # Apply augmentation if provided
        if self.augmentation_fn:
            image = self.augmentation_fn(image)

        # Add padding to match 1024x1024 with white color
        def pad_to_1024(image: np.ndarray) -> np.ndarray:
            height, width = image.shape[:2]
            pad_h = max(0, 1024 - height)
            pad_w = max(0, 1024 - width)
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        blob_img = pad_to_1024(blob_img)
        label = pad_to_1024(label)
        mask = pad_to_1024(mask)

        blob_img = torch.from_numpy(blob_img).permute(2, 0, 1).float() / 255.0  # Convert to (C, H, W) and normalize
        label = torch.from_numpy(label).permute(2, 0, 1).float() / 255.0  # Normalize
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0  # Convert to (1, H, W) and normalize
        return blob_img, label, mask


if __name__ == "__main__":
    # Example usage
    dataset = SightSweepDataset(data_folder=Path(r"data\train"), augmentation_fn=None)
    print(f"Number of images in dataset: {len(dataset)}")
    blob_img, label, mask = dataset[0]

    # Convert tensors to NumPy arrays for OpenCV
    blob_img = blob_img.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)
    label = label.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)
    mask = mask.squeeze(0).numpy()  # Convert (1, H, W) to (H, W)

    # Display the first image and its label
    cv2.imshow("Image", blob_img)
    cv2.waitKey(0)
    cv2.imshow("Label", label)
    cv2.waitKey(0)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
