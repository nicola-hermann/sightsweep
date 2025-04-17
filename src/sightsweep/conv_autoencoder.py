from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from sightsweep.dataset import SightSweepDataset
from torchinfo import summary


class ConvAutoencoder(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(4, 32, 4, stride=2, padding=1), nn.ReLU())  # -> (B, 32, 512, 512)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU())  # -> (B, 64, 256, 256)
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU())  # -> (B, 128, 128, 128)
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU())  # -> (B, 256, 64, 64)
        self.enc5 = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.ReLU())  # -> (B, 512, 32, 32)

        # Decoder
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU()  # -> (B, 256, 64, 64)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1), nn.ReLU()  # -> (B, 128, 128, 128)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.ReLU()  # -> (B, 64, 256, 256)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1), nn.ReLU()  # -> (B, 32, 512, 512)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1), nn.Sigmoid()  # -> (B, 3, 1024, 1024)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 3
        assert mask.shape == (x.shape[0], 1, *x.shape[2:])
        x = torch.cat([x, mask], dim=1)  # (B, 4, H, W)

        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottleneck = self.enc5(e4)

        # Decoder with skip connections
        d4 = self.dec5(bottleneck)
        d3 = self.dec4(torch.cat([d4, e4], dim=1))
        d2 = self.dec3(torch.cat([d3, e3], dim=1))
        d1 = self.dec2(torch.cat([d2, e2], dim=1))
        out = self.dec1(torch.cat([d1, e1], dim=1))
        return out

    def training_step(self, batch, batch_idx):
        x_masked, label, mask = batch
        x_hat = self(x_masked, mask)
        loss = self.masked_loss(label, x_hat, mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_masked, label, mask = batch
        x_hat = self(x_masked, mask)
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
        mask = mask.expand_as(x)  # (B, 1, H, W) -> (B, 3, H, W)
        return 0.9 * F.l1_loss(x_hat[mask == 0], x[mask == 0]) + 0.1 * F.mse_loss(x_hat[mask == 0], x[mask == 0])


if __name__ == "__main__":
    model = ConvAutoencoder().to(device="cuda")
    summary(model, input_size=[(64, 3, 1024, 1024), (64, 1, 1024, 1024)], device="cuda")

    # Example usage
    x = torch.randn(1, 3, 1024, 1024).to(device="cuda")  # (B, C, H, W)
    mask = torch.ones((1, 1, x.shape[-2], x.shape[-1])).to(device="cuda")  # (B, C, H, W)
    mask[:, 0, 512:, 512:] = 0  # Set the bottom right corner to zero
    x_hat = model(x, mask)
    assert x_hat.shape == x.shape, f"Output shape {x_hat.shape} does not match input shape {x.shape}"

    # Test non-square input
    batch_size = 16
    x = torch.randn(batch_size, 3, 512, 1024).to(device="cuda")  # (B, C, H, W)
    mask = torch.ones((batch_size, 1, x.shape[-2], x.shape[-1])).to(device="cuda")  # (B, C, H, W)
    mask[:, 0, 512:, 512:] = 0  # Set the bottom right corner to zero
    x_hat = model(x, mask)
    assert x_hat.shape == x.shape, f"Output shape {x_hat.shape} does not match input shape {x.shape}"

    dataset = SightSweepDataset(data_folder=Path(r"data\train"), augmentation_fn=None)
    masked_image, y, mask = dataset[0]
    mask = mask.unsqueeze(0).to(device="cuda")  # (B, 1, H, W)
    masked_image = masked_image.unsqueeze(0).to(device="cuda")  # (B, C, H, W)
    x_hat = model(masked_image, mask)
    assert (
        x_hat.shape == masked_image.shape
    ), f"Output shape {x_hat.shape} does not match input shape {masked_image.shape}"
