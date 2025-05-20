import gc
import os
from pathlib import Path

import lightning as L
import torch
import torch.optim as optim
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchinfo import summary

import wandb
from sightsweep.dataset import SightSweepDataset
from sightsweep.models import ConvAutoencoder, ConvVAE, MATInpaintingLitModule


def train(config=None):
    wandb.finish()  # Finish any previous runs to avoid conflicts
    remove_old_checkpoints()
    device = get_device()
    torch.set_float32_matmul_precision(
        "high"
    )  # Set float32 matmul precision to high for better performance
    L.seed_everything(42)  # Set random seed for reproducibility

    with wandb.init(
        config=config, entity="cvai-sightsweep", project="sightsweep", job_type="train"
    ):
        config = wandb.config
        run_name = f"{config['model_name']}_lr_{config['lr']}_weight_decay_{config['weight_decay']}"
        if config.model_name == "mat_inpainting":
            run_name += f"_patch_{config.patch_size}_dim_{config.embed_dim}_heads_{config.num_heads}_layers_{config.num_layers}"
        wandb.run.name = run_name

        # --- Dataset and DataLoader ---
        train_loader, val_loader = create_data_loaders(config["batch_size"], config["img_dim"])

        # --- Model ---
        model = create_model(config).to(device=device)

        # --- Logging ---
        logger = WandbLogger(project="sightsweep")

        # --- Callbacks ---
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename=f"{config.model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
            save_top_k=3,
            mode="min",
            save_last=True,
            every_n_epochs=1,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=3, verbose=True, mode="min"
        )

        # --- Trainer ---
        trainer = Trainer(
            max_epochs=100,
            accelerator="auto",
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            accumulate_grad_batches=config.get("accum_steps", 1),  # Default to 1 if not specified
            # profiler="simple",
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
        for file_name in os.listdir("checkpoints/"):
            file_path = os.path.join("checkpoints/", file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def print_batch_size_mem_usage(config, batch_sizes: list, img_size=512):
    """Print the memory usage of the model for different batch sizes."""
    device = get_device()
    model = create_model(config).to(device)
    summary(
        model,
        input_size=[
            (batch_sizes[-1], 3, img_size, img_size),
            (batch_sizes[-1], 1, img_size, img_size),
        ],
        device=device,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    model.train()  # Set the model to training mode
    train_dataset = SightSweepDataset(
        data_folder=Path(r"data/train"),
        augmentation_fn=None,
        max_img_dim=img_size,
    )
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() // 2),
            pin_memory=True,
            persistent_workers=True,
        )
        batch = next(iter(train_dataloader))  # Get a batch of data
        print("Data loaders created.")
        (x, label, mask) = batch
        x = x.to(device)
        label = label.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        if config["model_name"] == "mat_inpainting":
            loss = model._common_step((x, label, mask), 0, stage="train")
        else:
            x_hat = model(x, mask)
            loss = model.masked_loss(label, x_hat, mask)
        loss.backward()
        optimizer.step()

        # Print memory usage
        torch.cuda.synchronize(device)
        mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        mem_percent = mem / mem_total * 100
        print(f"Batch size {batch_size}: {mem:.2f} MB / {mem_total:.2f} MB ({mem_percent:.2f}%)")
        del x, label, batch, train_dataloader
        gc.collect()  # Garbage collection to free up memory
        torch.cuda.empty_cache()  # Clear cache to avoid memory fragmentation
        if mem_percent > 75:
            print(f"Warning: Memory usage exceeds 75% for batch size {batch_size}.")
            break  # Stop if memory usage is too high
        continue


def create_data_loaders(batch_size, img_size=1024):
    train_dataset = SightSweepDataset(
        data_folder=Path(r"data/train"),
        augmentation_fn=None,
        max_img_dim=img_size,
    )
    val_dataset = SightSweepDataset(
        data_folder=Path(r"data/validation"),
        augmentation_fn=None,
        max_img_dim=img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count() // 2),
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() // 2),
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader


def create_model(config: dict):
    if config["model_name"] == "conv_autoencoder":
        return ConvAutoencoder(config["lr"], config["weight_decay"])

    elif config["model_name"] == "vae":
        return ConvVAE(
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            img_size=config["img_dim"],
        )
    elif config["model_name"] == "mat_inpainting":
        return MATInpaintingLitModule(
            image_size=config["img_dim"],
            patch_size=config["patch_size"],
            num_channels=3,  # Assuming 3 channels for RGB
            num_layers=config["num_layers"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            ffn_dim=config["embed_dim"] * 4,  # Common practice, or add to config
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            dropout=config.get("dropout", 0.1),  # Optional dropout from config
        )
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")


def upload_model_to_wandb():
    wandb.init(entity="cvai-sightsweep", project="sightsweep")
    art = wandb.Artifact("conv_autoencoder_0.001_0.0001", type="model")
    art.add_file(
        "checkpoints/conv_autoencoder-epoch=07-val_loss=0.08.ckpt",
        name="conv_autoencoder-epoch=07-val_loss=0.0807.ckpt",
    )
    art.add_file(
        "checkpoints/conv_autoencoder-epoch=08-val_loss=0.08.ckpt",
        name="conv_autoencoder-epoch=08-val_loss=0.0809.ckpt",
    )
    art.add_file(
        "checkpoints/conv_autoencoder-epoch=09-val_loss=0.08.ckpt",
        name="conv_autoencoder-epoch=09-val_loss=0.0811.ckpt",
    )
    wandb.log_artifact(art)


if __name__ == "__main__":
    wandb.login()  # Login to Weights & Biases
    # Base config
    base_config = {
        "batch_size": 64,  # Start small for MAT
        "weight_decay": 1e-5,  # MAT might prefer smaller weight decay
        "img_dim": 512,  # Start with smaller images for MAT
        "max_epochs": 15,
    }

    # Model specific configs
    vae_config = {
        **base_config,
        "model_name": "vae",
        "lr": 1e-4,
    }

    mat_config = {
        **base_config,
        "model_name": "mat_inpainting",
        "patch_size": 32,  # img_dim (512) % patch_size (32) == 0
        "num_layers": 8,  # Transformer layers
        "embed_dim": 512,  # Embedding dimension
        "num_heads": 8,  # embed_dim (512) % num_heads (8) == 0
        "dropout": 0.1,
        "lr": 5e-5,  # Often transformers benefit from smaller LR with warmup
        # "accum_steps": 8,
    }
    # Ensure embed_dim is divisible by num_heads for MAT
    if mat_config["embed_dim"] % mat_config["num_heads"] != 0:
        print(
            f"Adjusting MAT embed_dim or num_heads: {mat_config['embed_dim']}%{mat_config['num_heads']} != 0"
        )
        # Simple adjustment: make embed_dim divisible, could also adjust num_heads
        mat_config["embed_dim"] = (mat_config["embed_dim"] // mat_config["num_heads"]) * mat_config[
            "num_heads"
        ]
        print(f"New embed_dim: {mat_config['embed_dim']}")

    # --- SELECT CONFIG TO RUN ---
    # current_config = vae_config
    current_config = mat_config
    # ----------------------------

    # To run memory benchmark:
    # print_batch_size_mem_usage(
    #     current_config,
    #     batch_sizes=[1, 2, 4, 8, 16, 32, 64],  # Test these batch sizes
    #     img_size=current_config["img_dim"],
    # )

    train(current_config)
