import os

import customtkinter as ctk
import yaml

from sightsweep.inpainting_module import Inpainting
from sightsweep.sam_predictor_module import SAM2Predictor
from sightsweep.ui import ImageClickerApp


# --- Load Configuration ---
def load_config(filepath: str):
    """Loads configuration from a YAML file."""
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {filepath}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return {}


if __name__ == "__main__":
    config = load_config("src/sightsweep/app_config.yaml")

    sam_checkpoint = config.get("sam_checkpoint")
    if not os.path.exists(sam_checkpoint):
        print(f"ERROR: SAM2 Checkpoint not found at: {sam_checkpoint}")
        print("Please download a checkpoint (e.g., sam2_hiera_tiny.pt) from the SAM2 repository:")
        print("https://github.com/facebookresearch/sam2")
        print("and update the sam_checkpoint key in config.yaml.")

    inpainting_checkpoint = config.get("inpainting_checkpoint")
    inpainting_model = config.get("inpainting_model")

    sam_predictor = SAM2Predictor(config)
    inpainting = Inpainting(config)
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = ImageClickerApp(root, sam_predictor, inpainting, config)
    root.mainloop()
