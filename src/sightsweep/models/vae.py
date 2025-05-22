from typing import Dict, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Optional: Add imports if logging images ---
try:
    import torchvision
except ImportError:
    print("torchvision not found, cannot log images during validation.")
    torchvision = None  # type: ignore


def _get_actual_num_groups(num_channels: int, requested_num_groups: int) -> int:
    """
    Helper to determine a valid number of groups for GroupNorm.
    Ensures num_channels is divisible by the returned number of groups.
    """
    if num_channels == 0:  # Should not happen in typical model architectures
        return 1
    if requested_num_groups <= 0:
        requested_num_groups = 1  # Ensure positive

    # If num_channels is divisible by requested_num_groups, use it
    if num_channels % requested_num_groups == 0:
        return requested_num_groups

    # Otherwise, find the largest factor of num_channels that is <= requested_num_groups
    # This prioritizes using a number of groups close to the requested one.
    for ng in range(min(requested_num_groups, num_channels), 0, -1):
        if num_channels % ng == 0:
            return ng
    return 1  # Fallback, should always find 1 as a divisor


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU"""

    def __init__(
        self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, num_groups=32
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        actual_num_groups = _get_actual_num_groups(out_channels, num_groups)
        self.norm = nn.GroupNorm(actual_num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class UpsampleConvBlock(nn.Module):
    """Upsample -> Conv -> GroupNorm -> ReLU"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        num_groups=32,
        upsample_mode="bilinear",
    ):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode=upsample_mode,
            align_corners=(
                True if upsample_mode in ["linear", "bilinear", "bicubic", "trilinear"] else None
            ),
        )
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
        )
        actual_num_groups = _get_actual_num_groups(out_channels, num_groups)
        self.norm = nn.GroupNorm(actual_num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvVAE(L.LightningModule):
    def __init__(
        self,
        input_channels: int = 4,  # 3 (image) + 1 (mask)
        output_channels: int = 3,  # Reconstructed image
        latent_dim: int = 64,  # Increased default latent_dim
        kl_weight: float = 1e-6,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        img_size: int = 512,
        num_groups: int = 32,  # For GroupNorm
        upsample_mode: str = "bilinear",  # For UpsampleConvBlock
        kl_anneal_epochs: int = 0,  # Epochs for KL weight to reach max_kl_weight * kl_anneal_factor
        kl_anneal_start_epoch: int = 0,  # Epoch to start KL annealing
        kl_anneal_factor: float = 1.0,  # Max KL weight multiplier during annealing
    ):
        super().__init__()
        assert latent_dim > 0
        assert img_size % 32 == 0, "Image size must be divisible by 32 due to 5 down/up samples"
        self.save_hyperparameters()  # Saves all __init__ args

        # --- Define Channel Sizes ---
        # Ensure these are compatible with num_groups or _get_actual_num_groups will handle it
        encoder_channels = [self.hparams.input_channels, 32, 64, 128, 256, 512]
        decoder_channels = [
            512,
            256,
            128,
            64,
            32,
        ]  # Last one is for the layer before final projection

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoder_blocks.append(
                ConvBlock(
                    encoder_channels[i], encoder_channels[i + 1], num_groups=self.hparams.num_groups
                )
            )

        # --- Latent Space Projection ---
        bottleneck_channels = encoder_channels[-1]  # 512

        # GroupNorm for mu and logvar layers
        gn_mu_logvar_groups = _get_actual_num_groups(
            self.hparams.latent_dim, self.hparams.num_groups
        )
        self.fc_mu = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels, self.hparams.latent_dim, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(gn_mu_logvar_groups, self.hparams.latent_dim),
            # No ReLU for mu/logvar
        )
        self.fc_logvar = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels, self.hparams.latent_dim, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(gn_mu_logvar_groups, self.hparams.latent_dim),
            # No ReLU for mu/logvar
        )

        # --- Decoder Input Projection ---
        gn_decoder_input_groups = _get_actual_num_groups(
            bottleneck_channels, self.hparams.num_groups
        )
        self.decoder_input_projection = nn.Sequential(
            nn.Conv2d(
                self.hparams.latent_dim, bottleneck_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(gn_decoder_input_groups, bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # First block (dec5 equivalent): Takes projected z (bottleneck_channels)
        self.decoder_blocks.append(
            UpsampleConvBlock(
                bottleneck_channels,
                decoder_channels[1],  # Output channels for this block (e.g., 256)
                num_groups=self.hparams.num_groups,
                upsample_mode=self.hparams.upsample_mode,
            )
        )

        encoder_skip_channels = encoder_channels[1:-1][::-1]  # [256, 128, 64, 32]
        for i in range(1, len(decoder_channels) - 1):
            # e.g., i=1 (dec4): prev_dec_ch=256 (from dec5), skip_ch=256 (from enc4), out_ch=128
            in_ch = decoder_channels[i] + encoder_skip_channels[i - 1]
            out_ch = decoder_channels[i + 1]
            self.decoder_blocks.append(
                UpsampleConvBlock(
                    in_ch,
                    out_ch,
                    num_groups=self.hparams.num_groups,
                    upsample_mode=self.hparams.upsample_mode,
                )
            )

        # Final block (dec1 equivalent)
        # Input: concat(dec2_out, e1_out) = decoder_channels[-1] + encoder_channels[1]
        final_block_in_ch = decoder_channels[-1] + encoder_channels[1]  # e.g. 32 + 32 = 64

        # Using Upsample then Conv for the final layer as well
        self.final_upsample = nn.Upsample(
            scale_factor=2,
            mode=self.hparams.upsample_mode,
            align_corners=(
                True
                if self.hparams.upsample_mode in ["linear", "bilinear", "bicubic", "trilinear"]
                else None
            ),
        )
        # Final convolution to get to output_channels, typically without norm/relu before sigmoid/tanh
        self.final_conv_layer = nn.Conv2d(
            final_block_in_ch,
            self.hparams.output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,  # Bias can be true here as no subsequent norm
        )
        self.final_activation = nn.Sigmoid()  # Output image in [0, 1] range

    def encode(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        assert x.shape[1] == 3, f"Input image should have 3 channels, got {x.shape[1]}"
        assert x.shape[2] == self.hparams.img_size and x.shape[3] == self.hparams.img_size, (
            f"Input image dimensions should be {self.hparams.img_size}x{self.hparams.img_size}, got {x.shape[2:]}"
        )
        assert mask.shape[1] == 1 and mask.shape[2:] == x.shape[2:], (
            f"Mask shape mismatch: expected (B, 1, {self.hparams.img_size}, {self.hparams.img_size}), got {mask.shape}"
        )

        inp = torch.cat([x, mask], dim=1)

        skip_connections = {}
        current_feature = inp
        for i, block in enumerate(self.encoder_blocks):
            current_feature = block(current_feature)
            if i < len(self.encoder_blocks) - 1:
                skip_connections[f"e{i + 1}"] = current_feature

        bottleneck = current_feature
        mu = self.fc_mu(bottleneck)
        logvar = self.fc_logvar(bottleneck)
        return mu, logvar, skip_connections

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, skip_connections: Dict[str, torch.Tensor]) -> torch.Tensor:
        current_feature = self.decoder_input_projection(z)

        current_feature = self.decoder_blocks[0](current_feature)

        num_main_decoder_blocks = len(self.decoder_blocks)  # This is len(decoder_channels) - 1
        for i in range(1, num_main_decoder_blocks):
            skip_idx = len(skip_connections) - (i - 1)
            skip_feature = skip_connections[f"e{skip_idx}"]
            concat_feature = torch.cat([current_feature, skip_feature], dim=1)
            current_feature = self.decoder_blocks[i](concat_feature)

        skip_feature_e1 = skip_connections["e1"]
        concat_final_feature = torch.cat([current_feature, skip_feature_e1], dim=1)

        # Final upsample, conv, and activation
        upsampled_feature = self.final_upsample(concat_final_feature)
        out_logits = self.final_conv_layer(upsampled_feature)
        out = self.final_activation(out_logits)

        return out

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, skips = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, skips)
        return x_hat, mu, logvar

    def masked_loss(
        self, target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask_expanded = mask.expand_as(target)
        pixels_pred = prediction[mask_expanded == 0]
        pixels_target = target[mask_expanded == 0]

        if pixels_pred.numel() == 0:
            return torch.tensor(0.0, device=target.device, requires_grad=True)

        loss_l1 = F.l1_loss(pixels_pred, pixels_target, reduction="mean")
        loss_mse = F.mse_loss(pixels_pred, pixels_target, reduction="mean")
        return 0.9 * loss_l1 + 0.1 * loss_mse  # Or just loss_l1

    def kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        return kl_loss.mean()

    def _common_step(self, batch, batch_idx, stage: str) -> torch.Tensor:
        x_masked, label, mask = batch
        x_hat, mu, logvar = self(x_masked, mask)

        recon_loss = self.masked_loss(label, x_hat, mask)

        # KL Annealing
        current_kl_weight = self.hparams.kl_weight
        if (
            self.hparams.kl_anneal_epochs > 0
            and self.trainer.current_epoch >= self.hparams.kl_anneal_start_epoch
        ):
            # Make sure not to divide by zero if kl_anneal_epochs is 1 and start_epoch is current_epoch
            anneal_duration = float(self.hparams.kl_anneal_epochs)
            anneal_progress = min(
                1.0,
                (self.trainer.current_epoch - self.hparams.kl_anneal_start_epoch) / anneal_duration,
            )
            current_kl_weight = (
                self.hparams.kl_weight * self.hparams.kl_anneal_factor * anneal_progress
            )

        kl_loss_val = self.kl_divergence_loss(mu, logvar)
        total_loss = recon_loss + current_kl_weight * kl_loss_val

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
            f"{stage}_kl_loss_raw",
            kl_loss_val,
            prog_bar=False,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )  # Raw KL
        self.log(
            f"{stage}_kl_loss_weighted",
            current_kl_weight * kl_loss_val,
            prog_bar=False,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )  # Weighted KL

        if stage == "train":
            self.log(
                "hyperparams_target_kl_weight",
                self.hparams.kl_weight,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "hyperparams_current_kl_weight",
                current_kl_weight,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
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
            self.logger.log_image(f"{stage}_original_images", [grid_orig], self.global_step)
            self.logger.log_image(f"{stage}_masked_images", [grid_masked], self.global_step)
            self.logger.log_image(f"{stage}_reconstructed_images", [grid_recon], self.global_step)

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
        scheduler_config: Optional[Dict] = None
        # ... (rest of your configure_optimizers, likely no changes needed here) ...
        if (
            self.trainer
            and hasattr(self.trainer, "estimated_stepping_batches")
            and self.trainer.estimated_stepping_batches
            and self.trainer.estimated_stepping_batches > 0  # Ensure positive
        ):
            try:
                total_steps = int(self.trainer.estimated_stepping_batches)
                print(f"INFO: Configuring OneCycleLR with total_steps={total_steps}")
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.hparams.lr,
                    total_steps=total_steps,
                    pct_start=0.1,  # Standard
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
                scheduler_config = None

        if scheduler_config is None:
            print("INFO: Using ReduceLROnPlateau scheduler as fallback or default.")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=5, factor=0.5, verbose=True
            )
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
