from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
from torch.utils.data import DataLoader
from sightsweep.dataset import SightSweepDataset
from torchvision.utils import make_grid  # For logging images
import matplotlib.pyplot as plt  # For displaying patches

# This code is inspired by the Mask-Aware Transformer (MAT) for image inpainting.
# https://github.com/fenglinglwb/MAT/tree/main


# --- MAT Transformer Components ---
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, input_patch_mask=None):
        # query, key, value: (batch_size, seq_len, embed_dim)
        # input_patch_mask: (batch_size, seq_len) where 1 is unmasked, 0 is masked.
        batch_size, seq_len, _ = query.shape
        q = self.q_proj(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if input_patch_mask is not None:
            # input_patch_mask: (B, S)
            mask_q = input_patch_mask.unsqueeze(1)  # (B, 1, S_key) - which keys can be attended to by any query
            mask_k = input_patch_mask.unsqueeze(2)  # (B, S_query, 1) - which queries are valid to attend from

            # attn_mask_value[b, i, j] is 1 if query_i AND key_j are unmasked
            attn_mask_value = mask_k * mask_q  # (B, S_query, S_key)
            attn_mask_value = attn_mask_value.unsqueeze(1).expand_as(scores)  # (B, num_heads, S_query, S_key)

            scores = scores.masked_fill(attn_mask_value == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        # Handle NaN from softmax if all scores in a row are -inf (e.g. a fully masked query attending to fully masked keys)
        # This happens if a query patch is masked and all key patches are also masked.
        # In such a case, attn_weights would be NaN. Set them to 0.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(self.activation(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x


class MaskAwareTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MaskedMultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout_attn = nn.Dropout(dropout)  # Often dropout is applied *inside* MHA and FFN
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, input_patch_mask=None):
        # Pre-norm
        normed_x = self.norm1(x)
        attn_output = self.attn(normed_x, normed_x, normed_x, input_patch_mask=input_patch_mask)
        x = x + self.dropout_attn(attn_output)  # Residual

        normed_x = self.norm2(x)
        ffn_output = self.ffn(normed_x)
        x = x + self.dropout_ffn(ffn_output)  # Residual
        return x


class SimplePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=2048):  # Increased max_len for larger images/smaller patches
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        if embed_dim % 2 == 0:
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:  # Odd embed_dim
            pe[0, :, 0::2] = torch.sin(position * div_term)
            if embed_dim > 1:
                pe[0, :, 1::2] = torch.cos(position * div_term)[:, :-1]  # Adjust for odd
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        return x + self.pe[:, : x.size(1)]


class MaskAwareImageTransformer(nn.Module):  # This is the core nn.Module
    def __init__(self, image_size, patch_size, num_channels, num_layers, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = num_channels * patch_size * patch_size

        self.patch_embedding = nn.Linear(self.patch_dim, embed_dim)
        self.pos_encoding = SimplePositionalEncoding(embed_dim, max_len=self.num_patches + 1)
        self.dropout_emb = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [MaskAwareTransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, self.patch_dim)

    def image_to_patches(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, self.num_patches, -1)
        return x

    def patches_to_image(self, x):
        B = x.shape[0]
        num_patches = x.shape[1]
        images = torch.zeros(B, self.num_channels, self.image_size, self.image_size, device=x.device)
        for b in range(B):
            for i in range(num_patches):
                row, col = divmod(i, self.num_patches_w)
                patch = x[b, i].view(self.num_channels, self.patch_size, self.patch_size)
                images[b, :, row * self.patch_size : (row + 1) * self.patch_size, col * self.patch_size : (col + 1) * self.patch_size] = patch
        return images

    def forward(self, patched_images, patch_level_mask_ones_unmasked):
        # patched_images: (B, NumPatches, PatchDim) - input patches (masked ones might be zeroed or special)
        # patch_level_mask_ones_unmasked: (B, NumPatches) - 1 for unmasked patch, 0 for masked patch
        x = self.patch_embedding(patched_images)
        x = self.pos_encoding(x)
        x = self.dropout_emb(x)

        for layer in self.layers:
            x = layer(x, input_patch_mask=patch_level_mask_ones_unmasked)

        x = self.norm(x)
        predicted_patches = self.output_projection(x)
        return predicted_patches


class MATInpaintingLitModule(L.LightningModule):
    def __init__(
        self,
        image_size,
        patch_size,
        num_channels,
        num_layers,
        embed_dim,
        num_heads,
        ffn_dim,
        lr,
        weight_decay,
        dropout=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()  # Saves all __init__ args to self.hparams

        self.model = MaskAwareImageTransformer(
            image_size=self.hparams.image_size,
            patch_size=self.hparams.patch_size,
            num_channels=self.hparams.num_channels,
            num_layers=self.hparams.num_layers,
            embed_dim=self.hparams.embed_dim,
            num_heads=self.hparams.num_heads,
            ffn_dim=self.hparams.ffn_dim,
            dropout=self.hparams.dropout,
        )
        self.l1_loss_fn = nn.L1Loss(reduction="none")
        self.mse_loss_fn = nn.MSELoss(reduction="none")

    def masked_loss(self, ground_truth_patches, predicted_patches, patch_level_loss_mask_bool):
        """
        Calculates a weighted L1 and MSE loss only on the masked patches.

        Args:
            ground_truth_patches (torch.Tensor): The original ground truth patches (B, NumPatches, PatchDim).
            predicted_patches (torch.Tensor): The patches predicted by the model (B, NumPatches, PatchDim).
            patch_level_loss_mask_bool (torch.Tensor): Boolean mask (B, NumPatches),
                                                      True where patches were originally masked and loss should be computed.
        Returns:
            torch.Tensor: The total calculated loss.
        """
        loss_l1 = torch.tensor(0.0, device=predicted_patches.device)
        loss_l2 = torch.tensor(0.0, device=predicted_patches.device)
        num_masked_elements = 0

        if patch_level_loss_mask_bool.any():
            # Select the components of predicted and ground truth patches that were masked
            pred_masked_components = predicted_patches[patch_level_loss_mask_bool]
            gt_masked_components = ground_truth_patches[patch_level_loss_mask_bool]

            if pred_masked_components.numel() > 0:
                # Your proposed logic:
                # 0.9 * F.l1_loss(x_hat[mask == 0], x[mask == 0]) + 0.1 * F.mse_loss(x_hat[mask == 0], x[mask == 0])
                # Here, pred_masked_components is x_hat[mask==0] and gt_masked_components is x[mask==0]
                # F.l1_loss and F.mse_loss with default reduction='mean' will average over all elements.

                loss_l1_val = F.l1_loss(pred_masked_components, gt_masked_components, reduction="mean")
                loss_l2_val = F.mse_loss(pred_masked_components, gt_masked_components, reduction="mean")

                loss_l1 = 0.8 * loss_l1_val
                loss_l2 = 0.2 * loss_l2_val
                num_masked_elements = pred_masked_components.numel()

        total_loss = loss_l1 + loss_l2  # Note: these are already weighted if using hparams

        # For logging the unweighted individual components if desired
        # self.log("unweighted_l1_on_masked", loss_l1_val if num_masked_elements > 0 else 0)
        # self.log("unweighted_l2_on_masked", loss_l2_val if num_masked_elements > 0 else 0)

        return (
            total_loss,
            loss_l1_val if num_masked_elements > 0 else torch.tensor(0.0),
            loss_l2_val if num_masked_elements > 0 else torch.tensor(0.0),
            num_masked_elements,
        )

    def image_pixel_mask_to_patch_mask(self, pixel_mask_ones_unmasked):
        # pixel_mask_ones_unmasked: (B, 1, H, W), 1 for unmasked pixel, 0 for masked
        # Output: (B, NumPatches), 1 for unmasked patch, 0 for masked patch
        P = self.hparams.patch_size
        # A patch is unmasked if ALL pixels within it are unmasked.
        # Or, a patch is masked if ANY pixel within it is masked.
        # Let's use: a patch is unmasked if its central pixel is unmasked (simplification)
        # Or more robustly: if mean value of mask in patch > threshold (e.g. 0.5)
        # Or: min value in patch == 1 (all pixels unmasked)

        # Using "min pooling" on the mask: if any pixel in the patch is 0 (masked),
        # the patch_mask value becomes 0.
        unfolded_mask = pixel_mask_ones_unmasked.unfold(2, P, P).unfold(3, P, P)  # (B, 1, NpH, NpW, P, P)
        patch_mask = unfolded_mask.amin(dim=(-1, -2))  # Min over (P,P) -> (B, 1, NpH, NpW)
        patch_mask = patch_mask.view(pixel_mask_ones_unmasked.shape[0], -1)  # (B, NumPatches)
        return patch_mask

    def display_patches(self, patches, title="Patches"):
        """
        Display patches in a popup window.

        Args:
            patches (torch.Tensor): Tensor of shape (B, NumPatches, PatchDim).
            title (str): Title of the popup window.
        """
        B, NumPatches, PatchDim = patches.shape
        patch_size = int(math.sqrt(PatchDim // self.hparams.num_channels))
        num_patches_per_row = int(math.sqrt(NumPatches))

        for b in range(B):
            fig, axes = plt.subplots(num_patches_per_row, num_patches_per_row, figsize=(10, 10))
            fig.suptitle(f"{title} - Batch {b + 1}", fontsize=16)
            for i in range(NumPatches):
                row, col = divmod(i, num_patches_per_row)
                patch = patches[b, i].view(self.hparams.num_channels, patch_size, patch_size).permute(1, 2, 0).cpu().numpy()
                axes[row, col].imshow(patch)
                axes[row, col].axis("off")
            plt.show()

    def forward(self, input_masked_imgs_pixels, pixel_mask_ones_unmasked, original_gt_imgs_pixels=None):
        # input_masked_imgs_pixels: (B, C, H, W) - image with masked areas (e.g., zeroed out)
        # pixel_mask_ones_unmasked: (B, 1, H, W) - 1 where content is original/valid, 0 where it's masked.
        # original_gt_imgs_pixels (Optional): (B,C,H,W) If provided, unmasked parts of output will be from this.

        input_patches = self.model.image_to_patches(input_masked_imgs_pixels)

        # Display the input patches
        # self.display_patches(input_patches, title="Input Patches")
        patch_level_mask_ones_unmasked = self.image_pixel_mask_to_patch_mask(pixel_mask_ones_unmasked)
        print(f"Patch level mask shape: {patch_level_mask_ones_unmasked}")

        # Transformer predicts all patches based on the input_patches and attention mask
        predicted_patches_all = self.model(input_patches, patch_level_mask_ones_unmasked)

        if original_gt_imgs_pixels is not None:
            original_gt_patches = self.model.image_to_patches(original_gt_imgs_pixels)
            # We need patch_level_mask_zeros_unmasked_bool (True for masked patches)
            patch_level_mask_zeros_unmasked_bool = patch_level_mask_ones_unmasked == 0

            final_patches = torch.where(
                patch_level_mask_zeros_unmasked_bool.unsqueeze(-1).expand_as(original_gt_patches),
                predicted_patches_all,  # Use prediction for MASKED patches
                original_gt_patches,  # Use original for UNMASKED patches
            )
        else:
            final_patches = predicted_patches_all  # Output is model's prediction for all patches

        final_patches.clamp_(0, 1)  # Ensure values are in [0, 1] range
        # self.display_patches(predicted_patches_all, title="Predicted Patches")
        inpainted_image_reconstructed = self.model.patches_to_image(final_patches)
        return inpainted_image_reconstructed

    def _common_step(self, batch, batch_idx, stage):
        # Batch from SightSweepDataset: (blob_img, image_tensor, pixel_mask_from_dataset)
        # blob_img: (B,C,H,W) - image with masked regions (e.g. zeroed out by dataset)
        # image_tensor: (B,C,H,W) - original ground truth image
        # pixel_mask_from_dataset: (B,1,H,W) - 0 for masked, 1 for unmasked (as per dataset annotes)
        input_masked_imgs, original_gt_imgs, pixel_mask_ones_unmasked = batch

        # 1. Convert input images (blob_img) to patches
        input_patches = self.model.image_to_patches(input_masked_imgs)

        # 2. Convert ground truth images to patches (for loss calculation)
        original_gt_patches = self.model.image_to_patches(original_gt_imgs)

        # 3. Convert pixel-level mask to patch-level mask for attention
        # pixel_mask_ones_unmasked already has 1 for unmasked, 0 for masked.
        patch_level_attention_mask = self.image_pixel_mask_to_patch_mask(pixel_mask_ones_unmasked)

        # 4. Forward pass through the transformer
        # The model receives input_patches (where some are from masked regions)
        # and patch_level_attention_mask to guide attention.
        predicted_patches = self.model(input_patches, patch_level_attention_mask)

        # 5. Calculate loss ONLY on the masked patches
        # We need a mask that is True for patches that were originally masked.
        patch_level_loss_mask_bool = patch_level_attention_mask == 0  # True for masked patches

       # Call the new masked_loss method
        total_loss, unweighted_l1, unweighted_l2, num_masked_elements = self.masked_loss(
            original_gt_patches,
            predicted_patches,
            patch_level_loss_mask_bool
        )

        self.log(
            f"{stage}_loss",
            total_loss,
            prog_bar=True,
            batch_size=original_gt_imgs.shape[0],
            sync_dist=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )

        # Log images for the first batch of validation
        if (
            stage == "val"
            and batch_idx == 0
            and self.logger
            and hasattr(self.logger.experiment, "log_image")
            and patch_level_loss_mask_bool.any()
        ):
            with torch.no_grad():
                # Use the model's forward for consistent visualization if original_gt_imgs is available
                inpainted_images_for_log = self.forward(
                    input_masked_imgs[:4], pixel_mask_ones_unmasked[:4], original_gt_imgs_pixels=original_gt_imgs[:4]
                )

                grid_original = make_grid(original_gt_imgs[:4].cpu(), nrow=2, normalize=True, value_range=(0, 1))
                grid_masked_input = make_grid(input_masked_imgs[:4].cpu(), nrow=2, normalize=True, value_range=(0, 1))
                grid_inpainted = make_grid(inpainted_images_for_log.cpu(), nrow=2, normalize=True, value_range=(0, 1))

                self.logger.experiment.log_image(f"{stage}_original_images", grid_original, self.global_step)
                self.logger.experiment.log_image(f"{stage}_masked_input_images", grid_masked_input, self.global_step)
                self.logger.experiment.log_image(f"{stage}_inpainted_images", grid_inpainted, self.global_step)

        return total_loss

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.trainer is None:
            raise ValueError("Trainer must be set before calling configure_optimizers")
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
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


if __name__ == "__main__":
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    # 1. Configuration for 512x512 image
    test_config = {
        "image_size": 512,  # Test with 512x512 image
        "patch_size": 32,  # (512/32)^2 = 16^2 = 256 patches
        "num_channels": 3,
        "num_layers": 2,  # Keep shallow for a quick test
        "embed_dim": 192,  # Keep relatively small for a test (192 % 6 == 0)
        "num_heads": 6,
        "ffn_dim": 192 * 4,  # Standard FFN dimension
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
    }

    print(f"--- Starting MATInpaintingLitModule Test (Image Size: {test_config['image_size']}) ---")

    # 2. Instantiate Model
    model = MATInpaintingLitModule(**test_config)
    print(f"Model instantiated: {model.__class__.__name__}")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # Corrected MPS check
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.train()  # Set to training mode

    # 3. Create Dummy Data (one batch)
    B = 1  # Use a small batch size for 512x512 test to manage memory
    train_dataset = SightSweepDataset(
        data_folder=Path(r"data/train"), augmentation_fn=None, max_img_dim=test_config["image_size"], max_length=3
    )
    val_dataset = SightSweepDataset(
        data_folder=Path(r"data/train"), augmentation_fn=None, max_img_dim=test_config["image_size"], max_length=3
    )
    input_masked_imgs, original_gt_imgs, pixel_mask_ones_unmasked = train_dataset[0]

    # 4. Test forward pass of the LightningModule
    print("\n--- Testing LightningModule forward() ---")
    try:
        with torch.no_grad():  # Ensure no gradients are computed during forward test
            reconstructed_image = model(
                input_masked_imgs, pixel_mask_ones_unmasked, original_gt_imgs_pixels=original_gt_imgs
            )
            print(f"LightningModule forward output shape: {reconstructed_image.shape}")
            assert reconstructed_image.shape == original_gt_imgs.shape, "Output shape mismatch"
            print("LightningModule forward() test passed.")
    except Exception as e:
        print(f"LightningModule forward() test FAILED: {e}")
        import traceback

        traceback.print_exc()

    # 5. Test _common_step (simulates a training/validation step)
    print("\n--- Testing _common_step() (simulates training_step) ---")

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="auto",  # Automatically choose the best accelerator (GPU/CPU)
        devices=1,  # Use one device
        logger=False,  # Disable logging for this test
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(train_dataset),
        val_dataloaders=DataLoader(val_dataset),
    )

    print(f"\n--- MATInpaintingLitModule Test (Image Size: {test_config['image_size']}) Finished ---")
