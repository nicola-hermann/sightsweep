from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from sightsweep.dataset import SightSweepDataset


class ConvAutoencoder(pl.LightningModule):
    def __init__(self, latent_dim=1024, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        # Maybe add skip connections later

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # (B, 32, 240, 320)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # (B, 64, 120, 160)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # (B, 128, 60, 80)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # (B, 256, 30, 40)
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # (B, 512, 15, 20)
            nn.ReLU(),
        )

        self.flatten = nn.Flatten(start_dim=1)  # Flatten the output to (B, 512 * 15 * 20)
        self.fc_enc = nn.Linear(512 * 15 * 20, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 512 * 15 * 20)  # (B, 512 * 15 * 20)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # (B, 256, 30, 40)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (B, 128, 60, 80)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (B, 64, 120, 160)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # (B, 32, 240, 320)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # (B, 3, 480, 640)
            nn.Sigmoid(),  # Scale to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (h, w) = x.shape[2:4]  # (H, W)
        x = x.float() / 255.0  # Scale to [0, 1]
        x = F.interpolate(x, size=(480, 640), mode="bilinear", align_corners=False)  # Resize to (B, 3, 480, 640)
        z = self.encoder(x)
        z = self.fc_enc(self.flatten(z))
        x_hat = self.fc_dec(z).view(-1, 512, 15, 20)  # Reshape to (B, 512, 15, 20)
        x_hat = self.decoder(x_hat)
        x_hat = x_hat * 255  # Scale back to [0, 255]
        x_hat = F.interpolate(x_hat, size=(h, w), mode="bilinear", align_corners=False)  # Resize back to original shape
        return x_hat

    def training_step(self, batch, batch_idx):
        x_masked, label, mask = batch
        x_hat = self(x_masked)
        loss = self.masked_loss(label, x_hat, mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_masked, label, mask = batch
        x_hat = self(x_masked)
        loss = self.masked_loss(label, x_hat, mask)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.trainer is None:
            raise ValueError("Trainer must be set before calling configure_optimizers")
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="linear",
            div_factor=25,
            final_div_factor=1e-4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def masked_loss(self, x, x_hat, mask):
        # maybe add perceptual loss e.g. VGG16
        loss = F.l1_loss(x_hat[mask == 0], x[mask == 0])
        return loss


if __name__ == "__main__":
    # crate a random tensor with the same shape as the input images
    x = torch.randn(1, 3, 1920, 1080).to(device="cuda")  # (B, C, H, W)
    model = ConvAutoencoder().to(device="cuda")
    x_hat = model(x)
    assert x_hat.shape == x.shape, f"Output shape {x_hat.shape} does not match input shape {x.shape}"

    dataset = SightSweepDataset(data_folder=Path(r"D:\sightsweep\train"), augmentation_fn=None)
    masked_image, y, mask = dataset[0]
    print(f"Masked image shape: {masked_image.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Mask shape: {mask.shape}")
    masked_image = masked_image.unsqueeze(0).to(device="cuda")  # (B, C, H, W)
    x_hat = model(masked_image)
    print(f"Output shape: {x_hat.shape}")
    assert (
        x_hat.shape == masked_image.shape
    ), f"Output shape {x_hat.shape} does not match input shape {masked_image.shape}"
