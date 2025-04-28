import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from typing import Tuple, Dict, Optional

# --- Optional: Add imports if logging images ---
try:
    import torchvision
except ImportError:
    print("torchvision not found, cannot log images during validation.")
    torchvision = None  # type: ignore


class ConvBlock(nn.Module):
    """Conv -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )  # Bias often false if using BN/GN, but keep for simplicity unless issues arise
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvTransposeBlock(nn.Module):
    """ConvTranspose -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv_transpose(x))


class ConvVAE(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 4,  # 3 (image) + 1 (mask)
        output_channels: int = 3,  # Reconstructed image
        latent_dim: int = 8,
        kl_weight: float = 1e-6,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        img_size: int = 512,
    ):  # Add img_size for potential future use/assertions
        super().__init__()
        assert latent_dim > 0
        assert img_size % 32 == 0, (
            "Image size must be divisible by 32 due to 5 down/up samples"
        )
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.img_size = img_size

        # --- Define Channel Sizes ---
        encoder_channels = [input_channels, 32, 64, 128, 256, 512]
        decoder_channels = [512, 256, 128, 64, 32]

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoder_blocks.append(
                ConvBlock(encoder_channels[i], encoder_channels[i + 1])
                # Example comment for first block: In: 4, Out: 32, Spatial: img/2
                # Example comment for last block: In: 256, Out: 512, Spatial: img/32
            )

        # --- Latent Space Projection ---
        bottleneck_channels = encoder_channels[-1]  # 512
        self.fc_mu = nn.Conv2d(
            bottleneck_channels, self.latent_dim, kernel_size=3, padding=1
        )
        self.fc_logvar = nn.Conv2d(
            bottleneck_channels, self.latent_dim, kernel_size=3, padding=1
        )
        # Output shape: (B, latent_dim, latent_spatial_dim, latent_spatial_dim)

        # --- Decoder Input Projection ---
        self.decoder_input = nn.Conv2d(
            self.latent_dim, bottleneck_channels, kernel_size=3, padding=1
        )
        # Output shape: (B, 512, latent_spatial_dim, latent_spatial_dim)

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # First block (dec5 equivalent): Takes projected z (512 channels)
        self.decoder_blocks.append(
            ConvTransposeBlock(bottleneck_channels, decoder_channels[1])  # 512 -> 256
            # Spatial: latent_spatial_dim * 2 = img/16
        )
        # Blocks 2 to 4 (dec4 to dec2 equivalent): Takes concat(prev_dec, skip_enc)
        # Skip channels are the outputs of encoder blocks 0 to 3 (reversed index 3 to 0)
        # encoder_channels = [4, 32, 64, 128, 256, 512] -> skips from index 4, 3, 2, 1
        encoder_skip_channels = encoder_channels[1:-1][::-1]  # [256, 128, 64, 32]
        for i in range(1, len(decoder_channels) - 1):
            # e.g., i=1 (dec4): prev_dec_ch=256 (from dec5), skip_ch=256 (from enc4), out_ch=128
            in_ch = decoder_channels[i] + encoder_skip_channels[i - 1]
            out_ch = decoder_channels[i + 1]
            self.decoder_blocks.append(
                ConvTransposeBlock(in_ch, out_ch)
                # Spatial: img/(2^(4-i)) e.g. i=1 -> img/8, i=3 -> img/2
            )

        # Final block (dec1 equivalent)
        # Input: concat(dec2_out, e1_out) = decoder_channels[-1] + encoder_channels[1] = 32 + 32 = 64
        final_in_ch = decoder_channels[-1] + encoder_channels[1]
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(final_in_ch, output_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # Output image in [0, 1] range
            # Spatial: img_size
        )

    def encode(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encodes the input image and mask into latent distribution parameters and skip connections.
        """
        assert x.shape[1] == 3, f"Input image should have 3 channels, got {x.shape[1]}"
        assert x.shape[2] == self.img_size and x.shape[3] == self.img_size, (
            f"Input image dimensions should be {self.img_size}x{self.img_size}, got {x.shape[2:]}"
        )
        assert mask.shape[1] == 1 and mask.shape[2:] == x.shape[2:], (
            f"Mask shape mismatch: expected (B, 1, {self.img_size}, {self.img_size}), got {mask.shape}"
        )

        inp = torch.cat([x, mask], dim=1)  # (B, 4, H, W)

        # Encoder pass & store skip connections
        skip_connections = {}
        current_feature = inp
        for i, block in enumerate(self.encoder_blocks):
            current_feature = block(current_feature)
            if (
                i < len(self.encoder_blocks) - 1
            ):  # Don't store the final bottleneck before latent projection
                skip_connections[f"e{i + 1}"] = current_feature  # Store e1, e2, e3, e4

        bottleneck = current_feature  # Output of the last encoder block

        # Calculate mu and logvar
        mu = self.fc_mu(bottleneck)
        logvar = self.fc_logvar(bottleneck)

        return mu, logvar, skip_connections

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if not self.training:
            return mu  # Use mean during evaluation/inference
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self, z: torch.Tensor, skip_connections: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Decodes the latent variable z back into an image using skip connections.
        """
        current_feature = self.decoder_input(
            z
        )  # Project latent z back to bottleneck channels

        # Apply decoder blocks with skip connections
        current_feature = self.decoder_blocks[0](current_feature)  # First block (dec5)

        # Loop through remaining decoder blocks (dec4 to dec2)
        num_decoder_blocks = len(self.decoder_blocks)
        for i in range(1, num_decoder_blocks):
            # Skip connection index: use e4 for dec4 (i=1), e3 for dec3 (i=2), e2 for dec2 (i=3)
            skip_idx = len(skip_connections) - (i - 1)  # e.g., i=1 -> 4 - 0 = 4 -> 'e4'
            skip_feature = skip_connections[f"e{skip_idx}"]
            # print(f"Decode block {i}: current={current_feature.shape}, skip=e{skip_idx} ({skip_feature.shape})")
            concat_feature = torch.cat([current_feature, skip_feature], dim=1)
            current_feature = self.decoder_blocks[i](concat_feature)

        # Final convolution layer (dec1 equivalent)
        # Skip connection 'e1'
        skip_feature_e1 = skip_connections["e1"]
        # print(f"Final conv: current={current_feature.shape}, skip=e1 ({skip_feature_e1.shape})")
        concat_feature = torch.cat([current_feature, skip_feature_e1], dim=1)
        out = self.final_conv(concat_feature)

        return out

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, logvar, skips = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, skips)
        return x_hat, mu, logvar

    def masked_loss(
        self, target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates reconstruction loss (L1 + MSE) only on the masked area.
        Mask=0 indicates the region to reconstruct/calculate loss on.
        """
        mask_expanded = mask.expand_as(target)  # (B, 1, H, W) -> (B, 3, H, W)

        # Select pixels where mask is 0 (the area to be inpainted)
        pixels_pred = prediction[mask_expanded == 0]
        pixels_target = target[mask_expanded == 0]

        if (
            pixels_pred.numel() == 0
        ):  # Handle case where mask is all 1s (no inpainting needed)
            # Return a zero loss tensor that requires gradients
            return torch.tensor(0.0, device=target.device, requires_grad=True)

        # Combination loss (weighted L1 + MSE)
        loss_l1 = F.l1_loss(pixels_pred, pixels_target, reduction="mean")
        loss_mse = F.mse_loss(pixels_pred, pixels_target, reduction="mean")

        # Adjust weights as needed (0.9 L1, 0.1 MSE was used before)
        # Using only L1 is also common for VAEs to reduce blurriness
        # return loss_l1
        return 0.9 * loss_l1 + 0.1 * loss_mse

    def kl_divergence_loss(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the KL divergence loss."""
        # KL divergence between N(mu, var) and N(0, 1)
        # Formula: 0.5 * sum(1 + log(var) - mu^2 - var) per element
        # We have logvar, so var = exp(logvar)
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]
        )  # Sum over C, H', W'
        return kl_loss.mean()  # Average over batch dimension B

    def _common_step(self, batch, batch_idx, stage: str) -> torch.Tensor:
        x_masked, label, mask = batch  # label is the original unmasked image

        x_hat, mu, logvar = self(x_masked, mask)  # VAE forward pass

        recon_loss = self.masked_loss(label, x_hat, mask)
        kl_loss = self.kl_divergence_loss(mu, logvar)

        total_loss = recon_loss + self.hparams.kl_weight * kl_loss

        # Logging
        self.log(
            f"{stage}_loss",
            total_loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_recon_loss",
            recon_loss,
            prog_bar=False,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_kl_loss",
            kl_loss,
            prog_bar=False,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )
        if stage == "train":
            self.log(
                "hyperparams_kl_weight",
                self.hparams.kl_weight,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
            # Calculate and log average std deviation in the latent space only during training steps for efficiency
            with torch.no_grad():
                avg_std = torch.exp(0.5 * logvar).mean()
            self.log(
                "metrics_latent_avg_std",
                avg_std,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

        # Optional: Log images during validation
        if (
            stage == "val"
            and batch_idx == 0
            and self.logger
            and hasattr(self.logger, "experiment")
            and torchvision
        ):
            num_log_images = min(4, label.shape[0])
            grid_orig = torchvision.utils.make_grid(label[:num_log_images])
            grid_masked = torchvision.utils.make_grid(x_masked[:num_log_images])
            grid_recon = torchvision.utils.make_grid(x_hat[:num_log_images].clamp(0, 1))
            self.logger.log_image(
                f"{stage}_original_images", [grid_orig], self.global_step
            )
            self.logger.log_image(
                f"{stage}_masked_images", [grid_masked], self.global_step
            )
            self.logger.log_image(
                f"{stage}_reconstructed_images", [grid_recon], self.global_step
            )

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Optional: Use a learning rate scheduler
        scheduler_config: Optional[Dict] = None
        if (
            self.trainer
            and hasattr(self.trainer, "estimated_stepping_batches")
            and self.trainer.estimated_stepping_batches
        ):
            try:
                total_steps = int(self.trainer.estimated_stepping_batches)
                print(f"INFO: Configuring OneCycleLR with total_steps={total_steps}")
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.hparams.lr,
                    total_steps=total_steps,
                    pct_start=0.1,
                    anneal_strategy="linear",
                    div_factor=25,
                    final_div_factor=1e4,
                )
                scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            except Exception as e:
                print(f"WARN: Could not configure OneCycleLR: {e}. Falling back.")
                scheduler_config = None  # Ensure fallback if error

        if scheduler_config is None:
            # Fallback scheduler if OneCycleLR setup fails or trainer steps unknown
            print("WARN: Using ReduceLROnPlateau scheduler as fallback.")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=5, factor=0.5
            )
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitor validation loss (ensure key matches logged key)
                "interval": "epoch",
                "frequency": 1,
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
