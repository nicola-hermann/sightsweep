import gc
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import FluxControlNetModel, FluxControlNetPipeline
from PIL import Image
from torchvision import transforms

from sightsweep.models.conv_autoencoder import ConvAutoencoder
from sightsweep.models.mat import MATInpaintingLitModule
from sightsweep.models.vae import ConvVAE


class Inpainting:
    def __init__(self, config, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.config = config
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.img_dim = self.config.get("img_dim", 512)

        self.model_name = self.config.get("inpaint_model")
        self.checkpoint_path = self.config.get("inpaint_checkpoint")

    def _load_inpainting_model(self):
        """Loads the main inpainting model to self.device in float32."""
        model_name = self.config.get("inpaint_model")
        checkpoint_path = self.config.get("inpaint_checkpoint")
        if not model_name or not checkpoint_path:
            raise ValueError(
                "Inpainting model or checkpoint not found in config for _load_inpainting_model."
            )
        if not os.path.exists(checkpoint_path):
            raise ValueError(
                f"Inpainting checkpoint not found at {checkpoint_path} for _load_inpainting_model."
            )

        print(f"Loading inpainting model '{model_name}' to {self.device} (float32)...")
        if model_name == "conv_autoencoder":
            model = ConvAutoencoder.load_from_checkpoint(checkpoint_path, map_location=self.device)
        elif model_name == "vae":
            model = ConvVAE.load_from_checkpoint(checkpoint_path, map_location=self.device)
        elif model_name == "mat":
            model = MATInpaintingLitModule.load_from_checkpoint(
                checkpoint_path, map_location=self.device
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")

        model.eval()
        model = model.float()
        model.to(self.device)
        print(f"Inpainting model '{model_name}' loaded in float32.")
        return model

    def _load_upscale_model_pipeline(self):
        """Loads the Flux upscaling pipeline if enabled."""
        enable_upscaling = self.config.get("enable_upscaling", False)
        if not enable_upscaling:
            return None

        try:
            controlnet_model_id = self.config.get(
                "flux_controlnet_model_id", "jasperai/Flux.1-dev-Controlnet-Upscaler"
            )
            pipeline_model_id = self.config.get(
                "flux_pipeline_model_id", "black-forest-labs/FLUX.1-dev"
            )

            torch_dtype_str = self.config.get("flux_torch_dtype", "bfloat16")
            if torch_dtype_str == "bfloat16":
                resolved_torch_dtype = torch.bfloat16
            elif torch_dtype_str == "float16":
                resolved_torch_dtype = torch.float16
            else:
                resolved_torch_dtype = torch.float32

            if self.device.type != "cuda" and resolved_torch_dtype == torch.bfloat16:
                print(
                    f"Warning: bfloat16 was configured but device is {self.device.type}. Using float32 for Flux upscaler."
                )
                resolved_torch_dtype = torch.float32
            elif self.device.type == "mps" and resolved_torch_dtype == torch.bfloat16:
                print(
                    "Warning: bfloat16 was configured for MPS. Using float16 instead for Flux upscaler for better compatibility."
                )
                resolved_torch_dtype = torch.float16

            print(
                f"Loading Flux ControlNet Model: {controlnet_model_id} with dtype: {resolved_torch_dtype}"
            )
            controlnet = FluxControlNetModel.from_pretrained(
                controlnet_model_id, torch_dtype=resolved_torch_dtype
            )

            print(
                f"Loading Flux ControlNet Pipeline: {pipeline_model_id} with dtype: {resolved_torch_dtype}"
            )
            upscaler_pipe = FluxControlNetPipeline.from_pretrained(
                pipeline_model_id, controlnet=controlnet, torch_dtype=resolved_torch_dtype
            )

            upscaler_pipe.to(self.device)

            if self.device.type == "cuda":
                print("Enabling model CPU offload for Flux upscaler.")
                upscaler_pipe.enable_model_cpu_offload()
            print(f"Flux Upscaler pipeline configured and moved to {self.device}.")
            return upscaler_pipe
        except Exception as e:
            print(f"Failed to load Flux upscaler pipeline: {e}")
            import traceback

            traceback.print_exc()
            return None

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        inpainting_model = None
        upscaler_pipeline = None

        try:
            inpainting_model = self._load_inpainting_model()
            model_name = self.config.get("inpaint_model")

            original_image_pil = image.convert("RGB")
            mask_pil = mask.convert("L")
            binary_mask_pil = mask_pil.point(lambda p: 0 if p > 0 else 255, mode="1")

            proc_image_pil = original_image_pil.copy()
            proc_mask_pil = binary_mask_pil.copy()

            if max(proc_image_pil.size) > self.img_dim:
                proc_image_pil.thumbnail((self.img_dim, self.img_dim), Image.Resampling.LANCZOS)
                proc_mask_pil.thumbnail((self.img_dim, self.img_dim), Image.Resampling.NEAREST)

            kernel_size = self.config.get("mask_dilation_kernel_size", 5)
            iterations = self.config.get("mask_dilation_iterations", 2)
            if kernel_size > 0 and iterations > 0:
                mask_np_uint8 = np.array(proc_mask_pil.convert("L"), dtype=np.uint8)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask_to_dilate = 255 - mask_np_uint8
                dilated_mask_inverted_np = cv2.dilate(mask_to_dilate, kernel, iterations=iterations)
                final_dilated_mask_np = 255 - dilated_mask_inverted_np
                proc_mask_pil = Image.fromarray(final_dilated_mask_np, mode="L").convert("1")
            else:
                proc_mask_pil = proc_mask_pil.convert("1")

            image_tensor = self.to_tensor(proc_image_pil).to(self.device)
            mask_tensor = self.to_tensor(proc_mask_pil).to(self.device)

            _, current_h, current_w = image_tensor.shape
            pad_h = max(0, self.img_dim - current_h)
            pad_w = max(0, self.img_dim - current_w)
            padding_dims = [pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2)]
            padded_image_tensor = TF.pad(image_tensor, padding_dims, fill=0)
            padded_mask_tensor = TF.pad(mask_tensor, padding_dims, fill=1)
            model_input_masked_tensor = padded_image_tensor.clone()
            model_input_masked_tensor[:, padded_mask_tensor[0] == 0] = 0

            with torch.no_grad():
                batch_model_input_masked = model_input_masked_tensor.unsqueeze(0)
                batch_padded_mask = padded_mask_tensor.unsqueeze(0)

                if model_name == "vae":
                    inpainted_output_tensor_batch, _, _ = inpainting_model(
                        batch_model_input_masked, batch_padded_mask
                    )
                elif model_name == "mat":
                    inpainted_output_tensor_batch = inpainting_model(
                        batch_model_input_masked, batch_padded_mask
                    )
                elif model_name == "conv_autoencoder":
                    inpainted_output_tensor_batch = inpainting_model(
                        padded_image_tensor.unsqueeze(0), batch_padded_mask
                    )
                else:
                    raise ValueError(f"Unsupported model type for inpainting: {model_name}")

                inpainted_padded_tensor = inpainted_output_tensor_batch.squeeze(0)

            final_padded_tensor = padded_image_tensor.clone()

            if final_padded_tensor.dtype != inpainted_padded_tensor.dtype:
                print(
                    f"Converting inpainted_padded_tensor from {inpainted_padded_tensor.dtype} to {final_padded_tensor.dtype} for assignment."
                )
                inpainted_padded_tensor_for_assignment = inpainted_padded_tensor.to(
                    final_padded_tensor.dtype
                )
            else:
                inpainted_padded_tensor_for_assignment = inpainted_padded_tensor

            final_padded_tensor[:, padded_mask_tensor[0] == 0] = (
                inpainted_padded_tensor_for_assignment[:, padded_mask_tensor[0] == 0]
            )
            if pad_h > 0 or pad_w > 0:
                unpadded_result_tensor = final_padded_tensor[
                    :,
                    padding_dims[1] : self.img_dim - padding_dims[3],
                    padding_dims[0] : self.img_dim - padding_dims[2],
                ]
                unpadded_result_tensor = unpadded_result_tensor[:, :current_h, :current_w]
            else:
                unpadded_result_tensor = final_padded_tensor

            unpadded_result_tensor = torch.clamp(unpadded_result_tensor, 0, 1)
            current_inpainted_pil = self.to_pil(unpadded_result_tensor.cpu())

            del (
                image_tensor,
                mask_tensor,
                padded_image_tensor,
                padded_mask_tensor,
                model_input_masked_tensor,
            )
            del (
                batch_model_input_masked,
                batch_padded_mask,
                inpainted_output_tensor_batch,
                inpainted_padded_tensor,
            )

            del final_padded_tensor, unpadded_result_tensor, inpainted_padded_tensor_for_assignment

            if self.config.get("enable_upscaling", False):
                upscaler_pipeline = self._load_upscale_model_pipeline()
                if upscaler_pipeline is not None:
                    print("Starting Flux upscaling process...")
                    image_to_upscale = current_inpainted_pil.copy()

                    pre_upscale_max_dim = self.config.get("flux_pre_upscale_max_dim", 0)
                    if pre_upscale_max_dim > 0 and max(image_to_upscale.size) > pre_upscale_max_dim:
                        print(
                            f"Pre-upscaling: Resizing image from {image_to_upscale.size} to fit within {pre_upscale_max_dim} max dimension."
                        )
                        image_to_upscale.thumbnail(
                            (pre_upscale_max_dim, pre_upscale_max_dim),
                            Image.Resampling.LANCZOS,
                        )

                    w, h = image_to_upscale.size
                    upscale_factor = self.config.get("flux_upscale_factor", 4)
                    target_w, target_h = int(w * upscale_factor), int(h * upscale_factor)

                    resampling_method_str = self.config.get("flux_pil_resampling_method", "LANCZOS")
                    resampling_methods = {
                        "NEAREST": Image.Resampling.NEAREST,
                        "BOX": Image.Resampling.BOX,
                        "BILINEAR": Image.Resampling.BILINEAR,
                        "HAMMING": Image.Resampling.HAMMING,
                        "BICUBIC": Image.Resampling.BICUBIC,
                        "LANCZOS": Image.Resampling.LANCZOS,
                    }
                    pil_resampling_method = resampling_methods.get(
                        resampling_method_str.upper(), Image.Resampling.LANCZOS
                    )

                    print(
                        f"Flux: Pre-resizing control image to target dimensions: ({target_w}, {target_h}) using {resampling_method_str}"
                    )
                    control_image_pil = image_to_upscale.resize(
                        (target_w, target_h), pil_resampling_method
                    )

                    prompt = self.config.get("flux_prompt", "")
                    controlnet_conditioning_scale = self.config.get(
                        "flux_controlnet_conditioning_scale", 0.6
                    )
                    num_inference_steps = self.config.get("flux_num_inference_steps", 28)
                    guidance_scale = self.config.get("flux_guidance_scale", 3.5)

                    print(
                        f"Flux: Running pipeline with prompt='{prompt}', steps={num_inference_steps}, guidance={guidance_scale}, cond_scale={controlnet_conditioning_scale}"
                    )
                    upscaled_image_result = upscaler_pipeline(
                        prompt=prompt,
                        control_image=control_image_pil,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=target_h,
                        width=target_w,
                    ).images[0]
                    print("Flux Upscaling complete.")
                    current_inpainted_pil = upscaled_image_result
                    del image_to_upscale, control_image_pil
                else:
                    print(
                        "Upscaling enabled but Flux upscaler pipeline failed to load. Skipping upscaling."
                    )
            return current_inpainted_pil

        finally:
            if inpainting_model is not None:
                del inpainting_model
                print("Inpainting model released.")
            if upscaler_pipeline is not None:
                del upscaler_pipeline
                print("Upscaler pipeline released.")

            if self.device.type != "cpu":
                torch.cuda.empty_cache()
            gc.collect()
            print("Inpaint function finished, resources cleaned up.")
