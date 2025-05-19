import os
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk
from torchvision import transforms

from sightsweep.inpainting_module import Inpainting
from sightsweep.sam_predictor_module import SAM2Predictor


class ImageClickerApp:
    def __init__(
        self, root: ctk.CTk, sam_predictor: SAM2Predictor, inpainting: Inpainting, config: dict
    ):
        self.root = root
        self.root.title("SightSweep - SAM2 Segmentation")
        self.root.geometry(
            f"{config.get('max_display_width', 1920)}x{config.get('max_display_height', 1080)}"
        )
        self.root.minsize(600, 500)

        self.sam_predictor = sam_predictor
        self.inpainting = inpainting
        self.config = config
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

        # --- Variables ---
        self.filepath = None
        self.pil_image_original = None
        self.pil_image_display = None
        self.pil_inpainting_original = None
        self.pil_inpainting_display = None
        self.tk_image_display = None
        self.tk_image_overlay = None
        self.tk_inpainting_display = None
        self.display_width = 0
        self.display_height = 0
        self.scale_ratio = 1.0

        self.positive_points = []
        self.negative_points = []
        self.current_mask_display = None

        # --- UI Elements ---
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # --- Controls Frame ---
        self.controls_frame = ctk.CTkFrame(root, corner_radius=0)
        self.controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        self.controls_frame.grid_columnconfigure(2, weight=1)

        self.btn_select = ctk.CTkButton(
            self.controls_frame,
            text="Select Image",
            command=self.select_image,
            width=130,
            height=35,
            corner_radius=15,
        )
        self.btn_select.grid(row=0, column=0, padx=(10, 5), pady=10)

        self.btn_clear_clicks = ctk.CTkButton(
            self.controls_frame,
            text="Clear Clicks",
            command=self.clear_clicks,
            width=130,
            height=35,
            corner_radius=15,
            state="disabled",
        )
        self.btn_clear_clicks.grid(row=0, column=1, padx=5, pady=10)

        self.lbl_filepath = ctk.CTkLabel(
            self.controls_frame, text="No image selected", anchor="w", wraplength=600
        )
        self.lbl_filepath.grid(row=0, column=2, sticky="ew", padx=5, pady=10)

        # --- Display Frame Original ---
        self.display_frame = ctk.CTkFrame(root)
        self.display_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.display_frame.grid_rowconfigure(0, weight=1)
        self.display_frame.grid_columnconfigure(0, weight=1)
        self.display_frame.grid_columnconfigure(1, weight=1)

        frame_bg_color = root.cget("fg_color")
        canvas_bg = (
            frame_bg_color[1] if isinstance(frame_bg_color, (list, tuple)) else frame_bg_color
        )
        self.canvas = ctk.CTkCanvas(
            self.display_frame,
            bg=canvas_bg,
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_image_click)  # Left Click (Positive)
        self.canvas.bind("<Button-3>", self.on_image_click)  # Right Click (Negative)
        self.root.bind("<Configure>", self.on_window_resize)

        self.lbl_coords = ctk.CTkLabel(
            self.display_frame,
            text="Left-click: Add Point | Right-click: Remove Area | Select Image",
            anchor="center",
        )
        self.lbl_coords.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))

        # --- Display Frame Inpainting --- #
        self.canvas_inpainting = ctk.CTkCanvas(
            self.display_frame,
            bg=canvas_bg,
            cursor="arrow",
            highlightthickness=0,
        )
        self.canvas_inpainting.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    def select_image(self):
        """Opens dialog, loads image, prepares for display and SAM2."""
        if not self.sam_predictor.predictor:
            messagebox.showwarning(
                "SAM2 Not Ready", "SAM2 model is not loaded. Cannot process image."
            )
            return

        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff"),
            ("All files", "*.*"),
        )
        filepath = filedialog.askopenfilename(title="Select an Image", filetypes=filetypes)
        if not filepath:
            return

        self._reset_image_data()
        self.filepath = filepath
        self.lbl_filepath.configure(text=os.path.basename(filepath))
        self.lbl_coords.configure(text="Left-click: Add Point | Right-click: Remove Area")

        try:
            # --- Load Original Image ---
            self.pil_image_original = Image.open(self.filepath).convert("RGB")
            image_np = np.array(self.pil_image_original)

            # --- handle rotation ---
            self.pil_image_original = ImageOps.exif_transpose(self.pil_image_original)
            image_np = np.array(self.pil_image_original)

            # --- SAM2 ---
            self.sam_predictor.set_image(image_np)
            print("Image features computed by SAM2.")

            # --- Calculate Display Size and Resize ---
            self._calculate_display_size()
            self._update_display_image()

            # --- Initialize Inpainting Display ---
            self._update_display_inpainting()

            self.btn_clear_clicks.configure(state="normal")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or process image with SAM2:\n{e}")
            self._reset_image_data()

    def _calculate_display_size(self):
        """Calculates the display size based on original size and canvas size."""
        if self.pil_image_original is None:
            return

        original_width, original_height = self.pil_image_original.size

        rotated = False
        # Check if the image is rotated (e.g., width > height after rotation)
        if original_width > original_height:
            rotated = True
            original_width, original_height = original_height, original_width

        canvas_width = 0.5 * self.display_frame.winfo_width() - 10
        canvas_height = self.display_frame.winfo_height() - 10
        canvas_width = max(canvas_width, 100)
        canvas_height = max(canvas_height, 100)

        width_ratio = canvas_width / original_width
        height_ratio = canvas_height / original_height
        self.scale_ratio = min(width_ratio, height_ratio, 1.0)

        self.display_width = int(original_width * self.scale_ratio)
        self.display_height = int(original_height * self.scale_ratio)

        if rotated:
            self.display_width, self.display_height = self.display_height, self.display_width

    def _update_display_image(self):
        """Resizes original image, creates Tkinter PhotoImage for display."""
        if self.pil_image_original is None or self.display_width <= 0 or self.display_height <= 0:
            self.canvas.delete("all")
            self.pil_image_display = None
            self.tk_image_display = None
            self.tk_image_overlay = None
            return

        try:
            self.pil_image_display = self.pil_image_original.resize(
                (self.display_width, self.display_height), Image.Resampling.LANCZOS
            )
            self.tk_image_display = ImageTk.PhotoImage(self.pil_image_display)
            self.canvas.config(width=self.display_width, height=self.display_height)
            self.canvas.delete("all")
            self._draw_canvas_content()
        except Exception as e:
            print(f"Error updating display image: {e}")
            messagebox.showerror("Display Error", f"Could not resize or display image:\n{e}")
            self._reset_image_data()

    def _update_display_inpainting(self):
        """Updates the inpainting display canvas."""
        if (
            self.pil_inpainting_original is None
            or self.display_width <= 0
            or self.display_height <= 0
        ):
            self.canvas_inpainting.delete("all")
            self.pil_inpainting_display = None
            self.tk_inpainting_display = None
            return

        try:
            # self.pil_inpainting_display = self.pil_inpainting_original.resize(
            #     (self.display_width, self.display_height), Image.Resampling.LANCZOS
            # )
            self.tk_inpainting_display = ImageTk.PhotoImage(self.pil_inpainting_original)
            self.canvas_inpainting.config(width=self.display_width, height=self.display_height)
            self.canvas_inpainting.delete("all")
            self.canvas_inpainting.create_image(0, 0, anchor="nw", image=self.tk_inpainting_display)
            self.canvas_inpainting.image = self.tk_inpainting_display  # Keep reference
        except Exception as e:
            print(f"Error updating inpainting display: {e}")
            messagebox.showerror(
                "Inpainting Display Error", f"Could not update inpainting display:\n{e}"
            )
            self._reset_image_data()

    def on_window_resize(self, event=None):
        """Handle window resize event to recalculate display size and update."""
        if (
            self.pil_image_original
            and self.display_frame.winfo_width() > 1
            and self.display_frame.winfo_height() > 1
        ):
            old_w, old_h = self.display_width, self.display_height
            self._calculate_display_size()
            if old_w != self.display_width or old_h != self.display_height:
                self._update_display_image()
                self._draw_canvas_content()

    def on_image_click(self, event):
        """Handles clicks, adds points, triggers SAM2 prediction and updates display."""
        if (
            self.pil_image_original is None
            or self.pil_image_display is None
            or not self.sam_predictor.predictor  # Check if predictor is initialized
        ):
            self.lbl_coords.configure(text="Please select an image first.")
            return

        display_x, display_y = event.x, event.y

        if 0 <= display_x < self.display_width and 0 <= display_y < self.display_height:
            original_x = int(display_x / self.scale_ratio)
            original_y = int(display_y / self.scale_ratio)
            original_width, original_height = self.pil_image_original.size
            original_x = min(max(original_x, 0), original_width - 1)
            original_y = min(max(original_y, 0), original_height - 1)

            is_positive = event.num == 1
            point_type = "Positive" if is_positive else "Negative"

            if is_positive:
                self.positive_points.append((original_x, original_y))
            else:
                self.negative_points.append((original_x, original_y))

            print(
                f"{point_type} click: Display=({display_x}, {display_y}), Original=({original_x}, {original_y})"
            )
            self.lbl_coords.configure(text=f"Added {point_type} point. Predicting...")

            # --- Run SAM2 Prediction --- #
            self.run_sam2_prediction()
            self._draw_canvas_content()

            # --- Run Inprint Model --- #
            self.run_inpainting()
            self._update_display_inpainting()

        else:
            self.lbl_coords.configure(text="Clicked outside displayed image bounds")

    def run_inpainting(self):
        """Runs inpainting based on the current mask."""
        if not self.pil_image_original or not self.current_mask_display:
            return

        try:
            self.pil_inpainting_original = self.inpainting.inpaint(
                self.pil_image_original.copy(), self.current_mask_display.copy()
            )
            print("Inpainting done.")

        except Exception as e:
            messagebox.showerror("Inpainting Error", f"Error during inpainting:\n{e}")
            print(f"Error during inpainting: {e}")

    def run_sam2_prediction(self):
        """Runs SAM2 prediction based on current points."""
        if not self.sam_predictor.predictor or (
            not self.positive_points and not self.negative_points
        ):
            self.current_mask_display = None
            return

        try:
            masks, scores, logits = self.sam_predictor.predict(
                self.positive_points, self.negative_points
            )

            if masks is not None and masks.shape[0] > 0:
                mask_original_np = masks[0].astype(bool)
                mask_color = tuple(self.config.get("mask_color", [0, 0, 255, 128]))
                self.current_mask_display = self.sam_predictor.create_mask_image(
                    mask_original_np, mask_color
                )
                best_score = scores[0] if len(scores) > 0 else -1
                self.lbl_coords.configure(text=f"Prediction done. Score: {best_score:.3f}")
                print(f"Prediction score: {best_score:.3f}")
            else:
                print("SAM2 did not return any masks.")
                self.current_mask_display = None
                self.lbl_coords.configure(text="Prediction done. No mask found.")

        except RuntimeError as e:
            messagebox.showerror("SAM2 Error", f"Error during SAM2 prediction:\n{e}")
            print(f"RuntimeError during SAM2 prediction: {e}")
            self.current_mask_display = None
        except Exception as e:
            messagebox.showerror("SAM2 Prediction Error", f"Failed to run SAM2 prediction:\n{e}")
            print(f"Error during SAM2 prediction: {e}")
            self.current_mask_display = None

    def _draw_canvas_content(self):
        """Draws the base image, mask overlay, and points onto the canvas."""
        if not self.tk_image_display or self.pil_image_display is None:
            self.canvas.delete("all")
            return

        # Base image for display (already resized)
        combined_image = self.pil_image_display.copy().convert("RGBA")

        if self.current_mask_display:
            try:
                # Convert the original-sized mask to RGBA
                mask_original_rgba = self.current_mask_display.convert("RGBA")

                mask_display_rgba = mask_original_rgba.resize(
                    (self.display_width, self.display_height), Image.Resampling.NEAREST
                )

                if mask_display_rgba.size == combined_image.size:
                    combined_image = Image.alpha_composite(combined_image, mask_display_rgba)
                else:
                    print(
                        f"Warning: Resized mask size mismatch ({mask_display_rgba.size} vs {combined_image.size}), skipping overlay."
                    )
            except Exception as e:
                print(f"Error resizing or blending mask: {e}")

        draw = ImageDraw.Draw(combined_image)
        point_radius = self.config.get("point_radius", 5)
        # Draw positive points
        for ox, oy in self.positive_points:
            dx, dy = int(ox * self.scale_ratio), int(oy * self.scale_ratio)
            bbox = (
                dx - point_radius,
                dy - point_radius,
                dx + point_radius,
                dy + point_radius,
            )
            draw.ellipse(bbox, fill="green", outline="white")
        # Draw negative points
        for ox, oy in self.negative_points:
            dx, dy = int(ox * self.scale_ratio), int(oy * self.scale_ratio)
            bbox = (
                dx - point_radius,
                dy - point_radius,
                dx + point_radius,
                dy + point_radius,
            )
            draw.ellipse(bbox, fill="red", outline="white")
        try:
            self.tk_image_overlay = ImageTk.PhotoImage(combined_image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image_overlay)
            self.canvas.image = self.tk_image_overlay  # Keep reference
        except Exception as e:
            print(f"Error creating/updating canvas image: {e}")

    def clear_clicks(self):
        """Clears positive/negative points and the mask."""
        if not self.pil_image_original:
            return

        print("Clearing clicks and mask.")
        self.positive_points = []
        self.negative_points = []
        self.current_mask_display = None
        self.lbl_coords.configure(text="Clicks cleared. Add new points.")
        self._draw_canvas_content()

    def _reset_image_data(self):
        """Helper method to clear all image-related variables."""
        print("Resetting image data.")
        self.clear_clicks()
        self.filepath = None
        self.pil_image_original = None
        self.pil_image_display = None
        self.pil_inpainting_display = None
        self.tk_image_display = None
        self.tk_image_overlay = None
        self.tk_inpainting_display = None
        self.display_width = 0
        self.display_height = 0
        self.scale_ratio = 1.0
        self.lbl_filepath.configure(text="No image selected")
        self.lbl_coords.configure(text="Select Image")
        self.canvas.delete("all")
        self.canvas.config(width=100, height=100)
        self.canvas_inpainting.delete("all")  # Clear the inpainting canvas
        self.btn_clear_clicks.configure(state="disabled")
