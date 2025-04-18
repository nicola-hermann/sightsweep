from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torchvision import transforms


class SightSweepDataset(Dataset):
    def __init__(
        self,
        data_folder: Path,
        augmentation_fn: Callable = None,
        min_blobs: int = 1,
        max_blobs: int = 4,
        max_img_dim: int = 800,
        max_length: int = None,
    ):
        self.data_folder = data_folder
        self.augmentation_fn = augmentation_fn
        self.data_paths = self.get_data_paths()
        self.min_blobs = min_blobs
        self.max_blobs = max_blobs
        self.max_img_dim = max_img_dim
        self.max_length = max_length
        self.to_tensor = transforms.ToTensor()

    def get_data_paths(self):
        data_paths = list(self.data_folder.rglob("*.jpg"))
        if not data_paths:
            raise ValueError(f"No jpg files found in {self.data_folder}")
        return data_paths

    def apply_random_mask(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        [_, height, width] = image.shape
        mask = torch.ones((height, width), dtype=torch.float32)

        num_blobs = torch.randint(self.min_blobs, self.max_blobs, (1,)).item()

        for _ in range(num_blobs):
            c_x = torch.randint(width // 10, 9 * width // 10, (1,)).item()
            c_y = torch.randint(height // 10, 9 * height // 10, (1,)).item()
            axes_x = torch.randint(min(width, height) // 16, min(width, height) // 5, (1,)).item()
            axes_y = torch.randint(min(width, height) // 16, min(width, height) // 5, (1,)).item()

            # Create elliptical mask using PIL (faster on tensors)
            ellipse_mask = Image.new("L", (width, height), 0)
            ellipse_draw = ImageDraw.Draw(ellipse_mask)
            ellipse_draw.ellipse([c_x - axes_x, c_y - axes_y, c_x + axes_x, c_y + axes_y], fill=255)
            ellipse_mask = self.to_tensor(ellipse_mask)
            mask = mask * (1 - ellipse_mask)
        image[:, mask[0] == 0] = 0  # Set masked pixels to black (0)
        return image, mask

    def pad_to_dim(self, image: torch.Tensor) -> torch.Tensor:
        [_, height, width] = image.shape
        pad_h = max(0, self.max_img_dim - height)
        pad_w = max(0, self.max_img_dim - width)
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]  # [left, top, right, bottom]
        return TF.pad(image, padding, fill=1)  # Fill with white (1 for normalized images)

    def __len__(self):
        return len(self.data_paths) if self.max_length is None else min(len(self.data_paths), self.max_length)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load the image
        img_path = self.data_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Resize the image if it exceeds the max dimensions
        if max(image.size) > self.max_img_dim:
            image.thumbnail((self.max_img_dim, self.max_img_dim))

        image_tensor = self.to_tensor(image)

        # Apply augmentation (before padding)
        if self.augmentation_fn:
            image_tensor = self.augmentation_fn(image_tensor)

        blob_img, mask = self.apply_random_mask(image_tensor.clone())

        blob_img = self.pad_to_dim(blob_img)
        mask = self.pad_to_dim(mask)
        image_tensor = self.pad_to_dim(image_tensor)

        # Apply mask
        return blob_img, image_tensor, mask


if __name__ == "__main__":
    # Example usage
    dataset = SightSweepDataset(data_folder=Path(r"data\train"), augmentation_fn=None)
    print(f"Number of images in dataset: {len(dataset)}")
    blob_img, label, mask = dataset[0]
    print(f"Blob image shape: {blob_img.shape}")
    print(f"Label image shape: {label.shape}")
    print(f"Mask shape: {mask.shape}")

    # Ensure all tensors are floating point and values are between 0 and 1
    assert blob_img.dtype == torch.float32 and blob_img.min() >= 0 and blob_img.max() <= 1, "blob_img is not valid"
    assert label.dtype == torch.float32 and label.min() >= 0 and label.max() <= 1, "label is not valid"
    assert mask.dtype == torch.float32 and mask.min() >= 0 and mask.max() <= 1, "mask is not valid"

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
