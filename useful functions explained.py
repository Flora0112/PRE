# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:07:16 2025

@author: admin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:40:05 2025

@author: biomech
"""

# Importing necessary libraries
import os  # For file and path manipulations
import numpy as np  # For numerical operations on arrays
import cv2  # OpenCV library for image processing
import re  # Regular expressions for string pattern matching
from skimage.measure import label  # For connected component labeling
from scipy import ndimage as ndi  # For advanced image processing (not used here but imported)
from skimage.feature import peak_local_max  # To detect local maxima in images (not used here)
from skimage.segmentation import watershed  # Watershed segmentation (not used here)
from skimage.measure import label  # Imported again (redundant, but no effect)

# Path to the trained model file (commented old path, active new path)
# MODEL_PATH = "/media/biomech/FLORA/GUI/0505_segresnet.20.40.0.7.pth"
MODEL_PATH = "E:/FLORA/GUI/0505_segresnet.20.40.0.7.pth"

# Function to crop an image vertically based on top and bottom percentages
def crop_image(img, top_pct_str, bottom_pct_str):
    h = img.shape[0]  # Get image height (number of rows)
    try:
        top_pct = float(top_pct_str)  # Convert top crop percentage to float
        bottom_pct = float(bottom_pct_str)  # Convert bottom crop percentage to float
    except ValueError:
        print("Error: Crop percentages must be numbers.")  # Handle non-numeric inputs
        return img  # Return original image if error
    
    # Calculate pixel indices corresponding to crop percentages
    top_pixels_to_remove = int(h * (top_pct / 100.0))
    bottom_crop_line = int(h * (bottom_pct / 100.0))

    # Clamp the crop pixel values within valid image range
    top_pixels_to_remove = max(0, min(top_pixels_to_remove, h - 1))
    bottom_crop_line = max(top_pixels_to_remove + 1, min(bottom_crop_line, h))

    # If top crop goes beyond or equals bottom crop, return a minimal slice and print error
    if top_pixels_to_remove >= bottom_crop_line :
        print("Crop error: Top crop percentage is greater than or equal to bottom crop percentage.")
        return img[0:1,:]  # Return just first row slice as fallback

    # Return the cropped image slice between top and bottom crop lines
    return img[top_pixels_to_remove:bottom_crop_line, :]


# Function to convert pixel area to real area based on magnification
def pixel_conversion(pixel_area,magnification):
    # Dictionary of effective conversion factors per magnification level
    effective_magnification ={10:0.5193,20:0.1298,40:0.03247}

    # Return converted area by multiplying pixel area by the factor
    return pixel_area*effective_magnification[magnification]

# Function to calculate various image metrics from original and segmented images
def calculate_metrics(original_img, segmented_img,magnification):
    """
    Calculate various metrics from a fluorescence image and its segmented mask.

    Parameters:
    - original_img (np.ndarray): Original fluorescence image (grayscale or RGB).
    - segmented_img (np.ndarray): Binary mask or soft segmentation prediction.

    Returns:
    - fluorescence_intensity (float): Mean intensity of the original image.
    - surface_coverage (float): Fraction of the image covered by thrombi.
    - mean_thrombi_fluorescence (float): Mean intensity within thrombus regions.
    - actual_num_thrombi (int): Number of detected thrombi.
    - mean_thrombus_area (float): Mean area of individual thrombi.
    - mean_roundness (float): Mean circularity of thrombi.
    """

    # Resize segmented image to fixed size (2960x2960) using nearest neighbor interpolation
    segmented_img=cv2.resize(segmented_img,(2960,2960),interpolation=cv2.INTER_NEAREST)

    # Binarize the segmentation: pixels > 0.5 become 1, else 0
    segmented_img = (segmented_img > 0.5).astype(np.uint8)

    # Calculate mean intensity of the original image (all pixels)
    fluorescence_intensity = float(np.mean(original_img))

    # Calculate surface coverage: ratio of segmented (non-zero) pixels to total pixels
    surface_coverage = np.sum(segmented_img) / segmented_img.size if segmented_img.size > 0 else 0

    # Check if original image and segmented image sizes match
    if original_img.shape[:2] != segmented_img.shape[:2]:
        # If original image is color (3D), convert to grayscale float32
        if original_img.ndim == 3:
            original_img_gray = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            original_img_gray = original_img.astype(np.float32)

        # Resize grayscale original image to segmentation size
        original_img_resized = cv2.resize(original_img_gray, (segmented_img.shape[1], segmented_img.shape[0]))
    else:
        # If sizes match, convert to grayscale float32 if necessary
        if original_img.ndim == 3:
            original_img_resized = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            original_img_resized = original_img.astype(np.float32)

    # Compute mean intensity within thrombus areas (where segmentation mask > 0)
    mean_thrombi_fluorescence = float(np.mean(original_img_resized[segmented_img > 0])) if np.any(segmented_img > 0) else 0

    # Label connected components in segmentation mask to count thrombi
    labeled_mask=label(segmented_img)
    actual_num_thrombi= labeled_mask.max()  # Number of connected thrombi

    # Find contours of thrombi for area and roundness calculation
    contours, _ = cv2.findContours(segmented_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    thrombi_areas = []  # List to hold individual thrombus areas
    roundness_values = []  # List to hold roundness values

    for contour in contours:
        area = cv2.contourArea(contour)  # Calculate contour area
        thrombi_areas.append(area)
        perimeter = cv2.arcLength(contour, True)  # Calculate contour perimeter
        if perimeter > 0:
            # Compute roundness (circularity) using formula 4*pi*area / perimeter^2
            roundness = 4 * np.pi * area / (perimeter ** 2)
            roundness_values.append(roundness)

    # Convert mean thrombi area from pixels to real units using magnification
    mean_thrombus_area = pixel_conversion(np.mean(thrombi_areas),magnification)

    # Compute mean roundness, or 0 if no values
    mean_roundness = np.mean(roundness_values) if roundness_values else 0

    # Return all computed metrics as a tuple
    return (
        fluorescence_intensity,
        surface_coverage,
        mean_thrombi_fluorescence,
        actual_num_thrombi,
        mean_thrombus_area,
        mean_roundness
    )


# Function to extract experimental condition and image info from filepath
def get_condition_and_image_info(filepath):
    """
    Extracts condition key and image number from a given filepath.
    Also returns a suggested prediction filename based on the image's original name.
    """
    # Split filepath into parts using OS separator, normalize path first
    parts = os.path.normpath(filepath).split(os.sep)

    # Default suggested filename is the filename without extension
    suggested_filename = os.path.splitext(os.path.basename(filepath))[0]

    try:
        # Extract folder with date info identified by presence of ")"
        date_raw_folder = [p for p in parts if ")" in p][0]
        date = date_raw_folder.split(")")[-1].strip()

        # Extract experimental condition folder matching specific keywords
        condition_folder_str = [p for p in parts if "Coll" in p and "Shear" in p and "chip" in p and "_P_" in p][0]

        # Parse condition string parts for collagen, shear, chip and P value
        coll = condition_folder_str.split("_")[0].replace("Coll", "")
        shear = condition_folder_str.split("_")[1].replace("Shear", "")
        chip = condition_folder_str.split("_")[2].replace("chip", "")
        p_val = condition_folder_str.split("_")[4]

        # Build condition key string from extracted values
        condition_key = f"{date}.{coll}.{shear}.{chip}.{p_val}"

        # Extract image number from filename's last underscore-separated part (remove leading zeros)
        filename = os.path.basename(filepath)
        img_num_str_raw = filename.split("_")[-1].split(".")[0]
        img_num_str = img_num_str_raw.lstrip("0") or "0"

        int(img_num_str)  # Check if image number is convertible to int (raises if not)

        # Compose suggested filename using condition key and image number
        suggested_filename = f"{condition_key}_Img{img_num_str}"

        return condition_key, img_num_str, suggested_filename

    except Exception as e:
        # If extraction fails, print warning and use fallback method
        print(f"[WARNING] Unable to extract image information for {filepath} using standard method: {e}. Attempting fallback.")

        # Use parent directory name as condition key fallback
        parent_dir_name = os.path.basename(os.path.dirname(filepath))

        # Extract numeric parts from filename stem and use last number as image number fallback
        fname_stem = os.path.splitext(os.path.basename(filepath))[0]
        num_search = re.findall(r'\d+', fname_stem)
        img_num_fallback = num_search[-1].lstrip("0") if num_search else "0"
        if not img_num_fallback:
            img_num_fallback = "0"

        print(f"Fallback for {filepath}: condition_key='{parent_dir_name}', img_num_str='{img_num_fallback}'")

        # Fallback suggested filename is original filename stem
        return parent_dir_name, img_num_fallback, fname_stem
