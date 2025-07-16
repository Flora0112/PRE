# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:46:02 2025

@author: admin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:59:55 2025

@author: biomech

This script loads a grayscale microscopy image, preprocesses it,
loads a trained AttentionUnet model, runs segmentation inference,
and displays the predicted binary mask.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from monai.networks.nets import AttentionUnet

def load_and_preprocess_image(path, size=512):
    """
    Load an image from disk, resize it, normalize pixel values,
    and convert it into a PyTorch tensor of shape [1,1,H,W].
    
    Args:
        path (str): Path to the image file.
        size (int): Target width and height to resize the image to (default 512).
        
    Returns:
        torch.Tensor: Image tensor normalized and shaped for model input.
    """
    # Open the image using PIL
    with Image.open(path) as img:
        # Resize image to (size x size) pixels
        img = img.resize((size, size))
        
        # Convert the image to a numpy array of type float32
        img = np.array(img, dtype=np.float32)
        
        # Normalize the image to have pixel values between 0 and 1
        img = img / img.max()
        
        # Add two extra dimensions to convert shape from (H, W) to (1, 1, H, W)
        # This is required because PyTorch models expect batches and channel dims
        img = np.expand_dims(img, axis=(0, 1))
        
        # Convert numpy array to PyTorch tensor of type float32
        return torch.tensor(img, dtype=torch.float32)

def predict_segmentation(image_path, model_path, size=512):
    """
    Load a pretrained segmentation model and predict a binary mask on the input image.
    
    Args:
        image_path (str): Path to the input grayscale image (.tif).
        model_path (str): Path to the pretrained model weights (.pth).
        size (int): Image size for resizing (default 512).
        
    Returns:
        np.ndarray: Predicted binary mask as a 2D numpy array (size x size).
    """
    # Use CPU device (change to "cuda" if GPU is available)
    device = torch.device("cpu")

    # Initialize the AttentionUnet model
    model = AttentionUnet(
        spatial_dims=2,          # 2D image input
        in_channels=1,           # single channel grayscale input
        out_channels=1,          # single output channel for binary mask
        channels=(64, 128, 256, 512),  # number of filters at each depth level
        strides=(2, 2, 2, 2),    # downsampling strides
        kernel_size=3,           # convolution kernel size
        up_kernel_size=3,        # upsampling kernel size
        dropout=1,               # dropout rate (1 means dropout always on; adjust as needed)
    ).to(device)
    
    # When you load a model, you MUST match the model parameters above with the model you are loading.

    # Load pretrained weights into the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm updates)

    # Load and preprocess the image into a tensor
    input_tensor = load_and_preprocess_image(image_path, size=size).to(device)

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Run the model forward pass to get raw output logits
        output = model(input_tensor)

        # Apply sigmoid to get probabilities in [0,1]
        output = torch.sigmoid(output)

        # Threshold probabilities at 0.6 to obtain binary mask (1 = predicted object, 0 = background)
        mask = (output > 0.6).float().squeeze().cpu().numpy()

        # Display the predicted binary mask
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Predicted segmentation mask")
        plt.imshow(mask, cmap="gray")
        plt.axis('off')
        plt.show()

    return mask

