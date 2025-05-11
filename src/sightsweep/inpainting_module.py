import os
import torch
from sightsweep.models.conv_autoencoder import ConvAutoencoder
from sightsweep.models.vae import ConvVAE
from sightsweep.models.mat import MATInpaintingLitModule
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


class Inpainting:
    def __init__(self, config, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        model_name = config.get("inpaint_model")
        checkpoint_path = config.get("inpaint_checkpoint")
        if not model_name or not checkpoint_path:
            raise ValueError("Inpainting model or checkpoint not found in config.")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Inpainting checkpoint not found at {checkpoint_path}")

        self.model = self.load_model(model_name, checkpoint_path)
        self.model.to(self.device)
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.img_dim = config.get("img_dim", 800)

    def load_model(self, model_name, checkpoint_path):
        if model_name == "conv_autoencoder":
            model = ConvAutoencoder.load_from_checkpoint(checkpoint_path)
            model.eval()
            return model

        if model_name == "vae":
            model = ConvVAE.load_from_checkpoint(checkpoint_path)
            model.eval()
            return model

        if model_name == "mat":
            model = MATInpaintingLitModule.load_from_checkpoint(checkpoint_path)
            model.eval()
            return model

        raise ValueError(f"Model {model_name} not supported.")

    def inpaint(self, image: Image, mask: Image) -> Image:
        image = image.convert("RGB")
        mask = mask.convert("L")
        mask = mask.point(lambda p: 0 if p > 0 else 255, mode="1")  # Convert to binary mask
        if max(image.size) > self.img_dim:
            image.thumbnail((self.img_dim, self.img_dim))
            mask.thumbnail((self.img_dim, self.img_dim))

        image_tensor = self.to_tensor(image).to(self.device)
        mask_tensor = self.to_tensor(mask).to(self.device)

        image_tensor[:, mask_tensor[0] == 0] = 0  # Set masked pixels to black (0)
        [_, height, width] = image_tensor.shape
        pad_h = max(0, self.img_dim - height)
        pad_w = max(0, self.img_dim - width)
        padding = [
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        ]  # [left, top, right, bottom]
        image_tensor = TF.pad(image_tensor, padding, fill=1)  # Fill with white (1 for normalized images)
        mask_tensor = TF.pad(mask_tensor, padding, fill=1)

        # Perform inpainting
        with torch.no_grad():
            inpainted_image = self.model(image_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))  # Add batch dimension
            inpainted_image = inpainted_image[0]  # Remove batch dimension

        # Apply inpainting to the original image
        image_tensor[:, mask_tensor[0] == 0] = inpainted_image[:, mask_tensor[0] == 0]
        # image_tensor = inpainted_image
        # Remove padding
        image_tensor = image_tensor[:, pad_h // 2 : height + pad_h // 2, pad_w // 2 : width + pad_w // 2]
        image_tensor = self.to_pil(image_tensor.squeeze(0).cpu())  # Convert back to PIL image
        # image_tensor.resize(original_size)
        return image_tensor
