import os
import torch
from sightsweep.conv_autoencoder import ConvAutoencoder


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

        model_name = config.get("inpainting_model")
        checkpoint_path = config.get("inpainting_checkpoint")
        if not model_name or not checkpoint_path:
            raise ValueError("Inpainting model or checkpoint not found in config.")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Inpainting checkpoint not found at {checkpoint_path}")

        self.model = self.load_model(model_name, checkpoint_path)
        self.model.to(self.device)

    def load_model(self, model_name, checkpoint_path):
        if model_name == "conv_autoencoder":
            model = ConvAutoencoder.load_from_checkpoint(checkpoint_path)
            model.eval()
            return model

        raise ValueError(f"Model {model_name} not supported.")

    def inpaint(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Move the image and mask to the device
        image = image.to(self.device)
        mask = mask.to(self.device)

        # Perform inpainting
        with torch.no_grad():
            inpainted_image = self.model(image, mask)

        return inpainted_image.cpu()
