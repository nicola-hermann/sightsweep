import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sightsweep.conv_autoencoder import ConvAutoencoder
from sightsweep.dataset import SightSweepDataset
from pathlib import Path
import lightning as L
import wandb
import os


def train(config=None):
    wandb.finish()  # Finish any previous runs to avoid conflicts
    remove_old_checkpoints()
    device = get_device()
    torch.set_float32_matmul_precision("high")  # Set float32 matmul precision to high for better performance
    L.seed_everything(42)  # Set random seed for reproducibility

    with wandb.init(config=config, entity="cvai-sightsweep", project="sightsweep", job_type="train"):
        config = wandb.config
        wandb.run.name = f"{config["model_name"]}_{config["lr"]}_{config["weight_decay"]}"

        # --- Dataset and DataLoader ---
        train_loader, val_loader = create_data_loaders(config["batch_size"])

        # --- Model ---
        model = create_model(config).to(device=device)

        # --- Logging ---
        logger = WandbLogger(project="sightsweep")

        # --- Callbacks ---
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="conv_autoencoder-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
            save_last=True,
            every_n_epochs=1,
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")

        # --- Trainer ---
        trainer = Trainer(
            max_epochs=100,
            accelerator="gpu",
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

        # --- Training ---
        trainer.fit(model, train_loader, val_loader)

        # --- Artifact ---
        artifact = wandb.Artifact(name=wandb.run.name, type="model")
        artifact.add_file(checkpoint_callback.best_model_path)
        wandb.log_artifact(artifact)
        wandb.finish()


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    return device


def remove_old_checkpoints():
    if os.path.exists("checkpoints/"):
        for file in os.listdir("checkpoints/"):
            os.remove(os.path.join("checkpoints/", file))


def print_batch_size_mem_usage(config, batch_sizes: list):
    """Print the memory usage of the model for different batch sizes."""
    device = get_device()
    model = create_model(config).to(device)
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, 1024, 1024).to(device)  # (B, C, H, W)
        mask = torch.ones((batch_size, 1, x.shape[-2], x.shape[-1])).to(device)  # (B, C, H, W)
        mask[:, 0, 512:, 512:] = 0  # Set the bottom right corner to zero
        x_hat = model(x, mask)
        mem = torch.cuda.memory_allocated(device) / (1024**2)
        mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        mem_percent = mem / mem_total * 100
        print(f"Batch size {batch_size}: {mem:.2f} MB / {mem_total:.2f} MB ({mem_percent:.2f}%)")
        del x, mask, x_hat  # Delete tensors to free memory
        torch.cuda.empty_cache()  # Clear cache to avoid memory fragmentation


def create_data_loaders(batch_size):
    train_dataset = SightSweepDataset(data_folder=Path(r"data/train"), augmentation_fn=None)
    val_dataset = SightSweepDataset(data_folder=Path(r"data/validation"), augmentation_fn=None)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=6, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=6, persistent_workers=True)

    return train_loader, val_loader


def create_model(config):
    if config["model_name"] == "conv_autoencoder":
        return ConvAutoencoder(config["lr"], config["weight_decay"])
    else:
        raise ValueError(f"Unknown model name: {config["model_name"]}")


if __name__ == "__main__":
    wandb.login()  # Login to Weights & Biases
    config = {
        "model_name": "conv_autoencoder",
        "batch_size": 8,
        "lr": 0.001,
        "weight_decay": 0.0001,
    }
    # print_batch_size_mem_usage(config, [16, 32, 64])  # Print memory usage for different batch sizes
    train(config)
