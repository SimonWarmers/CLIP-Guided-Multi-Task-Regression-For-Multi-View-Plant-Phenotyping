
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
import os
import shutil
from skimage.segmentation import slic, mark_boundaries
from sklearn.linear_model import Ridge

from PIL import Image

from torchvision import transforms
from groundingdino.util.inference import load_image
from skimage.segmentation import mark_boundaries


class CustomLIME:
    """Custom LIME-like implementation for multiple images (24 images) with global superpixel indexing"""
    
    def __init__(self, output_folder="lime_perturbed_images"):
        self.output_folder = output_folder
        self.segments_list = []  # List of segment arrays, one per image
        self.num_segments_per_image = []  # Number of segments per image
        self.total_num_segments = 0  # Total segments across all images
        self.segment_offsets = []  # Starting index for each image's segments
        self.images = []  # Store original images
        self.predictions = []
        self.perturbations = []
        self.plot_title = ""
        self.preprocess_manual_no_normalization = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        print("CustomLIME initialized.")

        
    def generate_superpixels(self, images, n_segments=50, compactness=10):
        """Generate superpixels for each image with global indexing across all images
        
        Args:
            images: List of numpy arrays (24 images)
            n_segments: Number of segments per image
            compactness: SLIC compactness parameter
        """
        self.images = images
        self.segments_list = []
        self.num_segments_per_image = []
        self.segment_offsets = [0]
        
        current_offset = 0
        for img_idx, image in enumerate(images):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            segments = slic(image_np, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=2)
            num_segs = len(np.unique(segments))
            
            segments_global = segments + current_offset
            
            self.segments_list.append(segments_global)
            self.num_segments_per_image.append(num_segs)
            
            current_offset += num_segs
            self.segment_offsets.append(current_offset)
        
        self.total_num_segments = current_offset
        print(f"Generated {self.total_num_segments} total superpixels across {len(images)} images")
        print(f"Segments per image: {self.num_segments_per_image}")
        
        return self.segments_list
    
    def create_perturbed_images(self, num_samples=100, hide_color=0):
        """Create perturbed images by randomly masking superpixels across all 24 images
        
        For each sample, creates a folder containing all 24 perturbed images.
        """
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)
        
        self.perturbations = []
        
        original_mask = np.ones(self.total_num_segments, dtype=int)
        self.perturbations.append(original_mask.copy())
        
        sample_folder = os.path.join(self.output_folder, "sample_0000_original")
        os.makedirs(sample_folder)
        for img_idx, image in enumerate(self.images):
            image = image.permute(1, 2, 0)
            img_path = os.path.join(sample_folder, f"image_{img_idx:02d}.png")
            Image.fromarray((image.numpy() * 255).astype(np.uint8)).save(img_path)
        
        for sample_idx in range(1, num_samples):
            global_mask = np.random.randint(0, 2, self.total_num_segments)
            self.perturbations.append(global_mask.copy())
            
            sample_folder = os.path.join(self.output_folder, f"sample_{sample_idx:04d}")
            os.makedirs(sample_folder)
            
            for img_idx, (image, segments) in enumerate(zip(self.images, self.segments_list)):
                perturbed_image = (image.numpy() * 255).copy().transpose(1, 2, 0)
                
                start_idx = self.segment_offsets[img_idx]
                end_idx = self.segment_offsets[img_idx + 1]
                
                for global_seg_idx in range(start_idx, end_idx):
                    if global_mask[global_seg_idx] == 0:
                        perturbed_image[segments == global_seg_idx] = hide_color
                
                img_path = os.path.join(sample_folder, f"image_{img_idx:02d}.png")
                Image.fromarray(perturbed_image.astype(np.uint8)).save(img_path)
        
        print(f"Saved {num_samples} sample folders (each with {len(self.images)} images) to '{self.output_folder}'")
    
    def run_predictions(self, prediction_fn):
        """Load saved images and run predictions using the provided function
        
        For each sample, loads all 24 images and passes them to prediction_fn.
        """
        self.predictions = []
        
        sample_folders = sorted([f for f in os.listdir(self.output_folder) if f.startswith('sample_')])
        
        for sample_folder in tqdm(sample_folders, desc="Running predictions"):
            sample_path = os.path.join(self.output_folder, sample_folder)
            
            image_files = sorted([f for f in os.listdir(sample_path) if f.endswith('.png')])
            image_sources = []
            
            for img_file in image_files:
                img_path = os.path.join(sample_path, img_file)
                image_source, image_tensor = load_image(img_path)
                image_source = image_source.transpose(2, 0, 1)
                image_sources.append(image_source)
            
            pred = prediction_fn(image_sources, None)
            self.predictions.append(pred)
        
        print(f"Completed {len(self.predictions)} predictions")
        return np.array(self.predictions)
    
    def compute_importance(self):
        """Compute feature importance using linear regression (like LIME)"""
        X = np.array(self.perturbations)  # (num_samples, total_num_segments)
        y = np.array(self.predictions)     # (num_samples,)
        
        distances = np.sqrt(np.sum((X - X[0]) ** 2, axis=1))
        
        kernel_width = 0.25 * self.total_num_segments
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        
        print("Fitting linear model...")
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X, y, sample_weight=weights)
        print("Linear model fitted.")
        
        importance = ridge_model.coef_
        
        return importance
    
    def get_top_importance_mask(self, importance, num_features=5):
        """Create masks highlighting top impactful superpixels by absolute value
        
        Returns masks for each image.
        """
        top_indices = np.argsort(np.abs(importance))[-num_features:]
        
        masks = []
        for img_idx, segments in enumerate(self.segments_list):
            mask = np.zeros(segments.shape, dtype=bool)
            for global_idx in top_indices:
                mask[segments == global_idx] = True
            masks.append(mask)
        
        return masks, top_indices, importance[top_indices]
    
    def visualize(self, num_features=-1, images_to_show=None):
        """Visualize the explanation with top impactful superpixels (by absolute value)
        
        Args:
            num_features: Number of top features to highlight
            images_to_show: Number of images to display in visualization (None = show all)
        """
        print("\nVisualizing explanations...")
        print("Computing importance...")
        importance = self.compute_importance()
        print("compute_importance Done")

        if num_features == -1:
            abs_importance = np.abs(importance)
            max_abs_importance = np.max(abs_importance)
            std_abs_importance = np.std(abs_importance)
            threshold = max_abs_importance - std_abs_importance
            top_indices = np.where(abs_importance >= threshold)[0]
            num_features = len(top_indices)
            print(f"Number of top impactful segments selected (within 1 std of max): {num_features}")
            
        masks, top_indices, top_importance = self.get_top_importance_mask(importance, num_features)
        
        if images_to_show is None:
            num_to_show = len(self.images)
        else:
            num_to_show = min(images_to_show, len(self.images))
        
        cols = min(8, num_to_show)
        rows = (num_to_show + cols - 1) // cols  # Ceiling division
        
        fig2, axes2 = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes2 = [axes2]
        elif rows == 1:
            axes2 = axes2.reshape(1, -1)
        elif cols == 1:
            axes2 = axes2.reshape(-1, 1)
        
        for i in range(num_to_show):
            row = i // cols
            col = i % cols
            
            image = self.images[i]
            mask = masks[i]
            segments = self.segments_list[i]
            
            img_superpixels = image.permute(1, 2, 0).numpy()
            
            superpixel_boundaries = mark_boundaries(np.zeros_like(img_superpixels), segments, 
                                                   color=(180/255, 180/255, 180/255), mode='thick')
            yellow_mask = (superpixel_boundaries[:, :, 0] > 0) & (superpixel_boundaries[:, :, 1] > 0)
            img_superpixels[yellow_mask] = 0.9 * superpixel_boundaries[yellow_mask] + 0.1 * img_superpixels[yellow_mask]
            
            img_with_explanation = img_superpixels.transpose(0, 1, 2)
            green_overlay = np.zeros_like(img_with_explanation)
            green_overlay[mask] = [0, 1, 0]
            img_with_explanation[mask] = 0.6 * img_with_explanation[mask] + 0.4 * green_overlay[mask]
            axes2[row, col].imshow(img_with_explanation)
            axes2[row, col].axis('off')
        
        total_plots = rows * cols
        for i in range(num_to_show, total_plots):
            row = i // cols
            col = i % cols
            axes2[row, col].axis('off')
        
        plt.figure(fig2.number)
        plt.tight_layout(pad=0.25)
        fig2.suptitle(self.plot_title, fontsize=16, y=1.02)
        plt.show()
        
        plt.figure(figsize=(15, 5))
        
        colors = []
        color_map = plt.cm.tab20(np.linspace(0, 1, len(self.images)))
        for img_idx in range(len(self.images)):
            start = self.segment_offsets[img_idx]
            end = self.segment_offsets[img_idx + 1]
            for _ in range(end - start):
                colors.append(color_map[img_idx])
        
        plt.bar(range(len(importance)), importance, color=colors, alpha=0.7)
        plt.xlabel("Global Superpixel Index")
        plt.ylabel("Importance Score")
        plt.title(f"Superpixel Importance Scores ({len(self.images)} images)")
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        for offset in self.segment_offsets[1:-1]:
            plt.axvline(x=offset-0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.show()
        
        return importance, masks

    def run_custom_lime(self, image_dir_path, prediction_fn, num_samples=50, n_segments=30, num_features=-1, compactness=10, plot_title=""):
        """Run custom LIME explanation on multiple images (24 images)
        
        Args:
            image_dir_path: Directory path containing images
            prediction_fn: Prediction function that takes list of 24 images
            num_samples: Number of perturbed samples to generate
            n_segments: Number of superpixels per image
            num_features: Number of top impactful superpixels to highlight (default: -1 and replaced with max stad )
        """
        self.plot_title = plot_title
        image_paths = [os.path.join(image_dir_path, f) for f in os.listdir(image_dir_path) if f.endswith('.png') or f.endswith('.jpg')]
        print(f"Loading images ", end='')
        image_sources = []
        for image_path in image_paths:
            print(f".", end='')
            image_source, _ = load_image(image_path)
            tmp_imag = self.preprocess_manual_no_normalization(Image.fromarray(image_source).convert("RGB"))
            image_sources.append(tmp_imag)  # Convert to HWC format
        
        print(f"Loaded {len(image_sources)} images")
        
        original_pred = prediction_fn(image_sources, None)
        print(f"Original prediction: {original_pred}")
        
        lime_explainer = CustomLIME(output_folder="../../lime_perturbed_images")
        
        lime_explainer.generate_superpixels(image_sources, n_segments=n_segments, compactness=compactness)
        
        lime_explainer.create_perturbed_images(num_samples=num_samples, hide_color=0)
        
        lime_explainer.run_predictions(prediction_fn)
        
        importance, masks = lime_explainer.visualize(num_features=num_features)
        
        return lime_explainer, importance, masks
        