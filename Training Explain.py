# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:27:28 2025

@author: admin
"""

# === Standard Python Libraries ===

import logging           # For logging messages during execution (e.g., training progress, errors)
import os                # To handle file paths and directory operations
import sys               # To interact with the system (used here to redirect logging to stdout)

# === PyTorch Core Libraries ===

import torch             # Main PyTorch module for tensor operations and model handling
import torch.nn as nn    # Contains neural network layers, activation functions, loss functions, etc.

# === Image and Visualization Libraries ===

from glob import glob            # To retrieve files matching a specified pattern (e.g., *.tif)
from PIL import Image            # Python Imaging Library to open and process image files
import matplotlib.pyplot as plt  # For visualizing input images and segmentation results

from skimage.measure import find_contours  # To extract contours from binary masks (useful for metrics or visualization)

# === Utilities ===

import numpy as np         # Fundamental library for numerical operations and array manipulation
from pathlib import Path    # Object-oriented file path manipulation (more robust than using raw strings)

# === Training Utilities ===

from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler that decays the LR by a factor every N steps

# === MONAI: Medical Imaging Deep Learning Framework ===

import monai  # MONAI is built on PyTorch and optimized for medical image segmentation tasks

# === MONAI: Core Data and Metrics ===

from monai.transforms import ToTensord                       # Converts input data to PyTorch tensors
from monai.metrics import DiceMetric                         # Computes Dice score (common segmentation metric)
from monai.data import list_data_collate, decollate_batch    # Utilities for batching and unbatching data
from monai.data import Dataset, DataLoader                   # Data loading and handling
from monai.losses import DiceFocalLoss                       # Loss combining Dice and Focal Loss (handles imbalance)
from monai.inferers import sliding_window_inference          # Inference on large images via patching
from monai.networks.nets import UNet, AttentionUnet, SegResNet  # Predefined segmentation model architectures

# === MONAI: Data Preprocessing and Augmentation Transforms ===

from monai.transforms import (
    Activations,               # Applies activation functions to model output (e.g., Sigmoid)
    AsDiscrete,                # Converts probabilities/logits to discrete labels
    Compose,                   # Combines multiple transforms into one pipeline
    ScaleIntensityRangeD,      # Normalizes image intensities to a desired range
    EnsureTyped,               # Ensures input data is converted to torch.Tensor or monai.data.MetaTensor
    RandAffineD,               # Applies random affine transformations (rotation, translation, scaling)
    ResizeD,                   # Resizes images to a consistent shape
    RandFlipD,                 # Applies random flipping along specified axes
)

# === Logging Setup ===

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# Configures logging to output messages to the console with level INFO or higher

# === Define Paths to Image and Mask Directories ===

image_dir = "/media/biomech/FLORA/SEGMENTATION/Training Set/Original"         # Training images
seg_dir = "/media/biomech/FLORA/SEGMENTATION/Training Set/Mask"              # Corresponding training masks
output_dir = "/media/biomech/FLORA/TEST COMPARISON/Output"                   # Output directory for saving results
test_image_dir = "/media/biomech/FLORA/TEST COMPARISON/Original"             # Images used for testing/prediction

val_image_dir = "/media/biomech/FLORA/SEGMENTATION/Validation Set/Original"  # Validation images
val_seg_dir = "/media/biomech/FLORA/SEGMENTATION/Validation Set/Mask"        # Corresponding validation masks

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# === Load File Paths for Images and Masks ===

images = sorted(glob(os.path.join(image_dir, "*.tif")))         # Sorted list of training images
segs = sorted(glob(os.path.join(seg_dir, "*.tif")))             # Sorted list of corresponding training masks
test_images = sorted(glob(os.path.join(test_image_dir, "*.tif")))  # Sorted list of images to test/predict

val_images = sorted(glob(os.path.join(val_image_dir, "*.tif")))  # Sorted list of validation images
val_segs = sorted(glob(os.path.join(val_seg_dir, "*.tif")))      # Sorted list of corresponding validation masks

# Ensure that training data is available
if len(images) == 0 or len(segs) == 0:
    raise ValueError("No images or segmentations found. Check your directories.")

# === Helper Function to Load and Convert Image ===

def load_and_convert_image(path):
    """
    Loads an image file using PIL and converts it into a NumPy array with a new axis
    to represent the channel (as MONAI expects shape [C, H, W]).
    """
    with Image.open(path) as img:
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)  # Add channel dimension
    return img

# Optional intensity thresholding (currently not applied)
def threshold_and_scale(img):
    # Example to limit intensity range manually if needed:
    # img[img < 20000] = 0
    # img[img > 40000] = 65536
    return img

# === Create MONAI-style dictionaries for dataset loading ===

# For training and validation (training images used for validation metrics during training)
val_files = [
    {"img": threshold_and_scale(load_and_convert_image(img)),
     "seg": load_and_convert_image(seg)}
    for img, seg in zip(images, segs)
]

# For testing (images only, no ground truth masks)
test_files = [
    {"img": threshold_and_scale(load_and_convert_image(img))}
    for img in test_images
]

# For final evaluation on a separate validation set
val_files_eval = [
    {"img": threshold_and_scale(load_and_convert_image(img)),
     "seg": load_and_convert_image(seg)}
    for img, seg in zip(val_images, val_segs)
]

# === Preprocessing and Data Augmentation Settings ===

optimal_patch_size = (512, 512)  # Target size for input images
min_val = 0                      # Min intensity for rescaling
max_val = 65536                  # Max intensity (16-bit images)

# === Data Transforms for Training ===

train_transforms = Compose([
    ToTensord(keys=["img", "seg"]),                  # Convert to torch.Tensor
    EnsureTyped(keys=["img", "seg"]),                # Ensure MONAI compatibility
    ScaleIntensityRangeD(keys=["img"], a_min=min_val, a_max=max_val, b_min=0.0, b_max=1.0),  # Normalize image
    ScaleIntensityRangeD(keys=["seg"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),            # Normalize mask
    ResizeD(keys=["img", "seg"], spatial_size=optimal_patch_size),                          # Resize both
    RandAffineD(  # Apply random affine transformation for data augmentation
        keys=["img", "seg"],
        prob=0.5,
        shear_range=(0.2, 0.2),
        scale_range=(0, 0.1),
        rotate_range=(-0.05, 0.05)
    ),
])

# === Data Transforms for Validation ===

val_transforms = Compose([
    ToTensord(keys=["img", "seg"]),
    EnsureTyped(keys=["img", "seg"]),
    ScaleIntensityRangeD(keys=["img"], a_min=min_val, a_max=max_val, b_min=0.0, b_max=1.0),
    ScaleIntensityRangeD(keys=["seg"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
    ResizeD(keys=["img", "seg"], spatial_size=optimal_patch_size),
    RandAffineD(
        keys=["img", "seg"],
        prob=0.5,
        shear_range=(0.2, 0.2),
        scale_range=(0, 0.1),
        rotate_range=(-0.05, 0.05)
    ),
])

# === Data Transforms for Test Set (only input images) ===

test_transforms = Compose([
    ToTensord(keys=["img"]),
    EnsureTyped(keys=["img"]),
    ScaleIntensityRangeD(keys=["img"], a_min=min_val, a_max=max_val, b_min=0.0, b_max=1.0),
    ResizeD(keys=["img"], spatial_size=optimal_patch_size),
])

# === Create MONAI Datasets and DataLoaders ===

# Validation loader (from training set, for monitoring performance during training)
val_ds = Dataset(data=val_files, transform=train_transforms)
val_loader = DataLoader(val_ds, batch_size=8, num_workers=4, collate_fn=list_data_collate)

# Test loader (inference only, no ground truth)
test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=8, num_workers=4, collate_fn=list_data_collate, shuffle=False)

# Evaluation loader (from a separate validation set, for final performance evaluation)
val_ds_eval = Dataset(data=val_files_eval, transform=val_transforms)
val_loader_eval = DataLoader(val_ds_eval, batch_size=8, num_workers=4, collate_fn=list_data_collate)

# === Define Device (GPU or CPU) ===

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define Segmentation Models ===

# You can switch between several architectures below by commenting/uncommenting.

# 1. U-Net model
# model1 = UNet(
#     spatial_dims=2,
#     in_channels=1,
#     out_channels=1,
#     channels=(64, 128, 256, 512),
#     strides=(2, 2, 2, 2),
#     num_res_units=4,
# ).to(device)

# 2. Attention U-Net model
model2 = AttentionUnet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(64, 128, 256, 512),  # Depth/width of network
    strides=(2, 2, 2, 2),
    dropout=0.1
).to(device)

# 3. SegResNet model (used as default in this script)
model = SegResNet(
    spatial_dims=2,
    init_filters=16,  # when you want to use a model already saved, you need the parameters here to match the saved model, here it's 8 for the last model used 
    in_channels=1,
    out_channels=1,
    upsample_mode="deconv"  # Use transposed convolution for upsampling
).to(device)

# === Define Loss Function, Optimizer, and Learning Rate Schedulers ===

# Combination of DiceFocalLoss and Binary Cross Entropy (BCE):
# DiceFocal helps handle class imbalance; BCE ensures pixel-wise separation
loss_function = lambda pred, target: DiceFocalLoss(alpha=0.5, gamma=3)(pred, target) + nn.BCEWithLogitsLoss()(pred, target)

# Adam optimizer with low learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Two options for learning rate scheduling:
scheduler2 = StepLR(optimizer, step_size=10, gamma=0.1)                      # Reduces LR every 10 epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # Cosine annealing (preferred)

# === Training Configuration ===

# Pretrained or previously saved model path (optional)
model_path = "/media/biomech/FLORA/GUI/Modèle/0505_segresnet.0.6.3.pth"

# Lists to store training history
loss_history = []
dice_scores = []

# MONAI Dice metric for evaluation (mean over batch, include background)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

# Post-processing transform: apply sigmoid and threshold at 0.6 to get binary masks
post_trans = Compose([
    Activations(sigmoid=True),
    AsDiscrete(threshold=0.6)
])  # Try threshold = 0.6 for better separation of thrombi

# === Function to Visualize Predictions during Training ===

def visualize_prediction(inputs, predictions, targets):
    """
    Displays a side-by-side comparison of input image, ground truth mask, and predicted mask
    for two samples in the batch.
    """
    for i in range(2):  # Show first two images of the batch
        img = inputs[i].detach().cpu().numpy().squeeze()
        pred = predictions[i].detach().cpu().numpy().squeeze()
        target = targets[i].detach().cpu().numpy().squeeze()

        # Binarize prediction with a fixed threshold
        pred = (pred > 0.7).astype(np.uint8)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Original image")

        plt.subplot(1, 3, 2)
        plt.imshow(target, cmap="gray")
        plt.title("Manual mask")

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Prediction")

        plt.show()

# === Load or Train the Model ===

if os.path.exists(model_path):
    print("A model already trained has been found. Charging of the model for testing..")
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

else:
    print("No model found. Training of the model...")

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()  # Training mode
        epoch_loss = 0

        # === Training Loop ===
        for batch in val_loader:
            inputs, labels = batch["img"].to(device), batch["seg"].to(device)
            optimizer.zero_grad()

            outputs = torch.sigmoid(model(inputs))  # Apply sigmoid for binary segmentation
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(val_loader)
        loss_history.append(avg_loss)

        # Update learning rate scheduler
        scheduler.step()

        print(f"Epoch {epoch+1}, Learning rate: {optimizer.param_groups[0]['lr']}")

        # === Evaluation Loop ===
        model.eval()
        dice_score_epoch = []

        with torch.no_grad():
            for val_batch in val_loader_eval:
                val_inputs, val_labels = val_batch["img"].to(device), val_batch["seg"].to(device)

                # Inference using sliding window to handle full-size images
                val_outputs = sliding_window_inference(val_inputs, (512, 512), 8, model)

                # Visualize predictions
                visualize_prediction(val_inputs, val_outputs, val_labels)

                # Post-processing
                val_labels = decollate_batch(val_labels)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                # Compute Dice Score
                dice_metric(y_pred=val_outputs, y=val_labels)

            dice_score = dice_metric.aggregate().item()
            dice_metric.reset()
            dice_score_epoch.append(dice_score)

        avg_dice = np.mean(dice_score_epoch)
        dice_scores.append(avg_dice)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Dice Score: {dice_score:.4f}")

    # === Plot Training History ===
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), loss_history, marker='o', linestyle='-', label="Training Loss")
    plt.plot(range(1, num_epochs+1), dice_scores, marker='s', linestyle='--', color='r', label="Validation Dice Score")
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title('Evolution of Dice Score and Loss during the training and evaluation')
    plt.legend()
    plt.grid()
    plt.show()

    # === Save the Trained Model ===
    torch.save(model.state_dict(), model_path)
    print("Model saved !")

# === Load Trained Model ===

model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

sw_batch_size = 4
threshold = 0.6  # Threshold to binarize the model's sigmoid outputs

# === Inference on Test Set ===

# test_image_filenames will be used to name the output files
test_image_filenames = sorted(glob(os.path.join(test_image_dir, "*.tif")))

with torch.no_grad():  # Disable gradient tracking for inference
    total_index = 0

    for i, test_data in enumerate(test_loader):
        test_images = test_data["img"].to(device)
        test_outputs = model(test_images)

        for j in range(test_images.size(0)):
            # Apply sigmoid activation and convert to NumPy array
            output = torch.sigmoid(test_outputs[j]).cpu().numpy().squeeze()

            # Binarize the prediction using the defined threshold
            binary_output = (output > threshold).astype(np.uint8) * 255

            # Retrieve original file name for saving
            original_name = Path(test_image_filenames[total_index]).stem
            new_filename = f"{original_name}.mask.png"

            # Save binary prediction mask as PNG
            Image.fromarray(binary_output).save(os.path.join(output_dir, new_filename))
            print(f"Image {total_index} treated and saved.")

            # === Optional Display of Results ===
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.title("Original image")
            plt.imshow(test_images[j].cpu().numpy().squeeze(), cmap="gray")

            plt.subplot(1, 2, 2)
            plt.title("Predicted image")
            plt.imshow(binary_output, cmap="gray")

            plt.show()

            total_index += 1
