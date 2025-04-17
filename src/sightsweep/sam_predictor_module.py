import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image


class SAM2Predictor:
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

        self.sam_checkpoint = config.get("sam_checkpoint")
        self.model_cfg = config.get("model_cfg")

        if not self.sam_checkpoint or not self.model_cfg:
            raise ValueError("SAM checkpoint or model config not found in config.")

        self.model = build_sam2(self.model_cfg, self.sam_checkpoint, self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.image_set = False

    def set_image(self, image_np):
        self.predictor.set_image(image_np)
        self.image_set = True

    def predict(self, positive_points, negative_points):
        if not self.image_set:
            raise RuntimeError("Image not set. Call set_image() first.")

        if not positive_points and not negative_points:
            return None, None, None

        input_points = np.array(positive_points + negative_points)
        input_labels = np.array(
            [1] * len(positive_points) + [0] * len(negative_points),
            dtype=np.int32,
        )

        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,  # Get single best mask
        )
        return masks, scores, logits

    def create_mask_image(self, mask_np, color):
        """Creates a colored PIL Image from a boolean mask."""
        if mask_np is None:
            return None
        mask_color_image = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        mask_color_image[mask_np] = color
        return Image.fromarray(mask_color_image, "RGBA")
