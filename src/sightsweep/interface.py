import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageChops
import os
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Constants ---
MAX_DISPLAY_WIDTH = 1920  # Adjust as needed for your screen
MAX_DISPLAY_HEIGHT = 1080  # Adjust as needed for your screen

# --- SAM2 Configuration --- # <--- SAM2 Change
SAM_CHECKPOINT = r"sam2.1_hiera_base_plus.pt"
MODEL_CFG = r"C:\Projects\HSLU\sightsweep\cfg\sam2.1_hiera_base_plus.yaml"  # Example: Corresponding model type key in the registry


MASK_COLOR = (0, 0, 255, 128)  # Blue with 50% alpha (R, G, B, Alpha)
POINT_RADIUS = 5  # Radius for drawing click markers

# --- Set Appearance Mode and Color Theme ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ImageClickerApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("SightSweep - SAM2 Segmentation")  # <--- SAM2 Change (Title)
        self.root.geometry("1920x1080")
        self.root.minsize(600, 500)

        # --- Variables ---
        self.filepath = None
        self.pil_image_original = None
        self.pil_image_display = None
        self.tk_image_display = None
        self.tk_image_overlay = None
        self.display_width = 0
        self.display_height = 0
        self.scale_ratio = 1.0

        # --- SAM2 Variables --- # <--- SAM2 Change
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.sam_model = build_sam2(MODEL_CFG, SAM_CHECKPOINT, self.device)
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)

        # SAM2 predictor likely handles embeddings internally after set_image
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

        # --- Display Frame ---
        self.display_frame = ctk.CTkFrame(root)
        self.display_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.display_frame.grid_columnconfigure(0, weight=1)
        self.display_frame.grid_rowconfigure(0, weight=1)

        frame_bg_color = root.cget("fg_color")
        canvas_bg = (
            frame_bg_color[1]
            if isinstance(frame_bg_color, (list, tuple))
            else frame_bg_color
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

    def select_image(self):
        """Opens dialog, loads image, prepares for display and SAM2."""
        if not self.sam_predictor:
            messagebox.showwarning(
                "SAM2 Not Ready", "SAM2 model is not loaded. Cannot process image."
            )  # <--- SAM2 Change (Message)
            return

        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff"),
            ("All files", "*.*"),
        )
        filepath = filedialog.askopenfilename(
            title="Select an Image", filetypes=filetypes
        )
        if not filepath:
            return

        self._reset_image_data()
        self.filepath = filepath
        self.lbl_filepath.configure(text=os.path.basename(filepath))
        self.lbl_coords.configure(
            text="Left-click: Add Point | Right-click: Remove Area"
        )

        try:
            # --- Load Original Image ---
            self.pil_image_original = Image.open(self.filepath).convert("RGB")

            # --- Prepare Image for SAM2 --- # <--- SAM2 Change
            print("Setting image in SAM2 predictor...")
            image_np = np.array(self.pil_image_original)
            # Assuming Sam2Predictor has the same set_image method
            self.sam_predictor.set_image(image_np)
            print("Image features computed by SAM2.")

            # --- Calculate Display Size and Resize ---
            self._calculate_display_size()
            self._update_display_image()

            self.btn_clear_clicks.configure(state="normal")

        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to load or process image with SAM2:\n{e}"
            )  # <--- SAM2 Change (Message)
            self._reset_image_data()

    def _calculate_display_size(self):
        """Calculates the display size based on original size and canvas size."""
        if self.pil_image_original is None:
            return

        original_width, original_height = self.pil_image_original.size
        canvas_width = self.display_frame.winfo_width() - 10
        canvas_height = self.display_frame.winfo_height() - 10
        canvas_width = max(canvas_width, 100)
        canvas_height = max(canvas_height, 100)

        width_ratio = canvas_width / original_width
        height_ratio = canvas_height / original_height
        self.scale_ratio = min(width_ratio, height_ratio, 1.0)

        self.display_width = int(original_width * self.scale_ratio)
        self.display_height = int(original_height * self.scale_ratio)

        # print(f"Original: {original_width}x{original_height}, Canvas: {canvas_width}x{canvas_height}, Display: {self.display_width}x{self.display_height}, Ratio: {self.scale_ratio:.4f}")

    def _update_display_image(self):
        """Resizes original image, creates Tkinter PhotoImage for display."""
        if (
            self.pil_image_original is None
            or self.display_width <= 0
            or self.display_height <= 0
        ):
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
            messagebox.showerror(
                "Display Error", f"Could not resize or display image:\n{e}"
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
                # print("Window resized, updating image display...")
                self._update_display_image()
                self._draw_canvas_content()

    def on_image_click(self, event):
        """Handles clicks, adds points, triggers SAM2 prediction and updates display."""
        if (
            self.pil_image_original is None
            or self.pil_image_display is None
            or not self.sam_predictor
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

            # --- Run SAM2 Prediction --- # <--- SAM2 Change
            self.run_sam2_prediction()
            self._draw_canvas_content()

        else:
            self.lbl_coords.configure(text="Clicked outside displayed image bounds")

    def run_sam2_prediction(self):  # <--- SAM2 Change (Method name and content)
        """Runs SAM2 prediction based on current points."""
        if not self.sam_predictor or (
            not self.positive_points and not self.negative_points
        ):
            self.current_mask_display = None
            return

        input_points = np.array(self.positive_points + self.negative_points)
        # Ensure labels are integers (required by SAM/SAM2)
        input_labels = np.array(
            [1] * len(self.positive_points) + [0] * len(self.negative_points),
            dtype=np.int32,
        )

        print(
            f"Running SAM2 with {len(self.positive_points)} positive, {len(self.negative_points)} negative points."
        )

        try:
            # --- Predict with SAM2 ---
            # Assuming Sam2Predictor has a compatible predict method signature.
            # Verify arguments (point_coords, point_labels, multimask_output)
            # and return format (masks, scores, logits) from SAM2 documentation if issues arise.
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,  # Get single best mask
            )

            # masks should be (num_masks, H, W) boolean numpy array.
            # Since multimask_output=False, we expect (1, H, W)
            if masks.shape[0] == 0:
                print("SAM2 did not return any masks.")
                self.current_mask_display = None
                self.lbl_coords.configure(text="Prediction done. No mask found.")
                return

            mask_original_np = masks[0].astype(bool)  # Ensure it's boolean

            # --- Create visual mask for display ---
            h, w = mask_original_np.shape
            mask_color_image = np.zeros((h, w, 4), dtype=np.uint8)
            mask_color_image[mask_original_np] = MASK_COLOR

            mask_pil_original = Image.fromarray(mask_color_image, "RGBA")

            self.current_mask_display = mask_pil_original.resize(
                (self.display_width, self.display_height), Image.Resampling.NEAREST
            )

            best_score = (
                scores[0] if len(scores) > 0 else -1
            )  # Handle cases where scores might be empty
            self.lbl_coords.configure(text=f"Prediction done. Score: {best_score:.3f}")
            print(f"Prediction score: {best_score:.3f}")

        except Exception as e:
            messagebox.showerror(
                "SAM2 Prediction Error", f"Failed to run SAM2 prediction:\n{e}"
            )  # <--- SAM2 Change (Message)
            print(
                f"Error during SAM2 prediction: {e}"
            )  # Also print to console for debugging
            self.current_mask_display = None

    def _draw_canvas_content(self):
        """Draws the base image, mask overlay, and points onto the canvas."""
        if not self.tk_image_display or self.pil_image_display is None:
            self.canvas.delete("all")
            return

        combined_image = self.pil_image_display.copy().convert("RGBA")

        if self.current_mask_display:
            try:
                mask_rgba = self.current_mask_display.convert("RGBA")
                if mask_rgba.size == combined_image.size:
                    combined_image = Image.alpha_composite(combined_image, mask_rgba)
                else:
                    print(
                        f"Warning: Mask size mismatch ({mask_rgba.size} vs {combined_image.size}), skipping overlay."
                    )
            except Exception as e:
                print(f"Error blending mask: {e}")

        draw = ImageDraw.Draw(combined_image)
        for ox, oy in self.positive_points:
            dx, dy = int(ox * self.scale_ratio), int(oy * self.scale_ratio)
            draw.ellipse(
                (
                    dx - POINT_RADIUS,
                    dy - POINT_RADIUS,
                    dx + POINT_RADIUS,
                    dy + POINT_RADIUS,
                ),
                fill="green",
                outline="white",
            )
        for ox, oy in self.negative_points:
            dx, dy = int(ox * self.scale_ratio), int(oy * self.scale_ratio)
            draw.ellipse(
                (
                    dx - POINT_RADIUS,
                    dy - POINT_RADIUS,
                    dx + POINT_RADIUS,
                    dy + POINT_RADIUS,
                ),
                fill="red",
                outline="white",
            )

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
        self.filepath = None
        self.pil_image_original = None
        self.pil_image_display = None
        self.tk_image_display = None
        self.tk_image_overlay = None
        self.display_width = 0
        self.display_height = 0
        self.scale_ratio = 1.0
        # Clear SAM2 specific data for the image
        # Predictor is kept, but internal image state is cleared on next set_image
        self.clear_clicks()
        self.lbl_filepath.configure(text="No image selected")
        self.lbl_coords.configure(text="Select Image")
        self.canvas.delete("all")
        self.canvas.config(width=100, height=100)
        self.btn_clear_clicks.configure(state="disabled")


# --- Main Execution ---
if __name__ == "__main__":
    # Check for SAM2 checkpoint existence at startup
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"ERROR: SAM2 Checkpoint not found at: {SAM_CHECKPOINT}")
        print(
            "Please download a checkpoint (e.g., sam2_hiera_tiny.pt) from the SAM2 repository:"
        )
        print("https://github.com/facebookresearch/sam2")
        print("and update the SAM_CHECKPOINT variable in the script.")
        # Optionally add a file dialog to ask for the checkpoint here if needed
        # exit() # Exit if you want to force the user to fix the path

    root = ctk.CTk()
    app = ImageClickerApp(root)
    root.mainloop()
