import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow duplicate OpenMP libraries to avoid crashes

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import predictions as pr  # Your custom module to predict segmentation masks
import cv2
import threading
import useful_functions as uf
import pandas as pd

# MODEL_PATH: path to the pretrained segmentation model weights file
MODEL_PATH = "D:/GUI/Modèle/2305_segresnet.0.6.pth"

# List to store any image display windows references (to manage later)
image_windows = []

# Global tkinter variables and other globals initialized as None or default values
prediction_file_name_var = None  # tkinter StringVar linked to user input for mask file name
magnification_choice = None      # stores magnification selection
crop_top_entry = None            # entry widget for top crop percentage
crop_bottom_entry = None         # entry widget for bottom crop percentage
last_magnification = "10"        # default magnification as string

# Dictionary to store segmentation results grouped by condition name
results_by_condition = {}

# Variables for batch processing control
current_index = 0
file_queue = []
processed_folder_path = ""

def save_mask(mask_image, original_image_path):
    """
    Save a predicted segmentation mask image to disk.
    
    Arguments:
    - mask_image: numpy array or PIL Image representing the mask
    - original_image_path: full path of the original image file, used to derive mask filename
    
    If the user entered a custom file name in the GUI input, the mask is saved
    with that name plus ".mask" suffix and the original extension.
    Otherwise, the mask filename is original file name with ".mask" suffix.
    """
    
    # Get directory of the original image file
    dir_path = os.path.dirname(original_image_path)
    
    # Get the base file name (with extension) of the original image
    base_name = os.path.basename(original_image_path)
    
    # Split into name and extension, e.g. "image.tif" => ("image", ".tif")
    name, ext = os.path.splitext(base_name)

    # Retrieve user input mask name if defined and non-empty, otherwise empty string
    custom_name = prediction_file_name_var.get().strip() if 'prediction_file_name_var' in globals() else ""

    # Build the mask filename based on user input or fallback to original filename
    if custom_name:
        mask_name = f"{custom_name}.mask{ext}"
    else:
        mask_name = f"{name}.mask{ext}"

    # Full path to save the mask image
    mask_path = os.path.join(dir_path, mask_name)

    # Convert numpy mask array to PIL Image before saving
    if isinstance(mask_image, np.ndarray):
        if mask_image.dtype == np.float32 or mask_image.dtype == np.float64:
            # If mask values are floats between 0 and 1, scale to 0-255 for saving
            if np.max(mask_image) <= 1.0 and np.min(mask_image) >= 0.0:
                pil_mask = Image.fromarray((mask_image * 255).astype(np.uint8))
            else:
                # If values are outside 0-1 range, just convert to uint8
                pil_mask = Image.fromarray(mask_image.astype(np.uint8))
        else:
            # Already integer type: convert directly to PIL Image
            pil_mask = Image.fromarray(mask_image.astype(np.uint8))
    else:
        # If already PIL Image (unlikely), use it as is
        pil_mask = mask_image

    # Save the PIL Image mask to disk
    pil_mask.save(mask_path)
    print(f"Mask saved to {mask_path}")
    

def export_to_excel_structured(results_data_sorted, save_base_path):
    """
    Export segmentation results to a structured Excel file with formatted headers, 
    vertical metric names, color-coded time points, and organized by condition.

    Parameters:
    - results_data_sorted (dict): 
        Keys are condition names (str).
        Values are lists of dicts with keys: 'img_num_int' (int) and 'metrics' (dict).
    - save_base_path (str): folder path to save the Excel file.

    The Excel file will be named "segmentation_results_summary.xlsx" inside save_base_path.
    """

    try:
        excel_filename = "segmentation_results_summary.xlsx"
        excel_full_path = os.path.join(save_base_path, excel_filename)
        # Use xlsxwriter engine for advanced formatting
        writer = pd.ExcelWriter(excel_full_path, engine='xlsxwriter')
    except Exception as e:
        print(f"Error creating ExcelWriter (check if 'xlsxwriter' is installed): {e}")
        messagebox.showerror("Excel Export Error", f"Unable to initialize ExcelWriter: {e}\nPlease install 'xlsxwriter'.")
        return

    workbook = writer.book
    # Create a worksheet named 'Segmentation Results'
    worksheet = workbook.add_worksheet('Segmentation Results')

    # Define time intervals and max display duration in seconds
    time_interval_s = 10
    max_seconds_display = 240

    # Define Excel cell formats for headers, rotated text, data cells, colored time cells, etc.
    format_condition_header = workbook.add_format({
        'bold': True, 'font_size': 14, 'align': 'center', 'bg_color': '#DCE6F1', 'border': 1
    })
    format_metric_name_vertical = workbook.add_format({
        'bold': True, 'font_size': 12, 'align': 'center', 'valign': 'vcenter',
        'rotation': 90, 'bg_color': '#C6EFCE', 'border': 1
    })
    format_time_header = workbook.add_format({
        'bold': True, 'font_size': 12, 'align': 'center', 'bg_color': '#DDEBF7', 'border': 1
    })
    format_value_header = workbook.add_format({
        'bold': True, 'font_size': 12, 'align': 'center', 'bg_color': '#DDEBF7', 'border': 1
    })

    format_time_cell = workbook.add_format({'align': 'center', 'border': 1, 'bg_color': '#FFFFFF'})
    format_minute_red = workbook.add_format({'align': 'center', 'border': 1, 'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    format_minute_orange = workbook.add_format({'align': 'center', 'border': 1, 'bg_color': '#FFEB9C', 'font_color': '#9C5700'})

    format_data_cell = workbook.add_format({'align': 'right', 'num_format': '0.00', 'border': 1})
    format_empty_cell = workbook.add_format({'border': 1})

    current_excel_row = 0

    # Define the metrics and their keys in the results dictionary
    metrics_config = [
        ("Fluorescence Intensity", "mean_intensity"),
        ("Surface Coverage (%)", "surface"),
        ("Thrombus Fluorescence Intensity", "mean_thrombi_fluorescence"),
        ("Number of Thrombi", "count"),
        ("Mean Thrombus Area [µm²]", "mean_area"),
        ("Roundness", "roundness")
    ]

    # Create a list of time points to display as strings like '10s', '20s', ..., '1m', '2m', etc.
    display_time_points = []
    for s in range(time_interval_s, max_seconds_display + 1, time_interval_s):
        if s % 60 == 0:
            # Convert multiples of 60 seconds to minutes display
            minute = s // 60
            display_time_points.append(f"{minute}m")
        else:
            display_time_points.append(f"{s}s")

    metric_name_height = len(display_time_points)

    # Iterate through each condition and its list of results
    for condition_name, images_results_list in results_data_sorted.items():
        if not images_results_list:
            continue

        # Sort images by their integer image number for time ordering
        images_results_list.sort(key=lambda x: x['img_num_int'])

        # Write the condition name as a merged header across 3 columns
        worksheet.merge_range(current_excel_row, 0, current_excel_row, 2, condition_name, format_condition_header)
        current_excel_row += 2  # Add empty row below header for spacing

        # Create a dict mapping image number to its metrics dict for quick lookup
        metrics_by_img_num = {res['img_num_int']: res['metrics'] for res in images_results_list}

        # Loop over each metric to write vertically per condition
        for metric_label, metric_key in metrics_config:
            # Check metric key exists in results (skip if missing)
            if metric_key not in images_results_list[0]['metrics']:
                print(f"Warning: Metric '{metric_key}' not found in results for {condition_name}")
                continue

            # Write column headers for time and value
            worksheet.write(current_excel_row, 1, "Time (s)", format_time_header)
            worksheet.write(current_excel_row, 2, "Value", format_value_header)

            # Merge cells vertically for metric label on leftmost column, rotated text
            worksheet.merge_range(
                current_excel_row, 0, current_excel_row + metric_name_height, 0,
                metric_label, format_metric_name_vertical
            )

            data_start_row_for_metric = current_excel_row + 1

            # Write time points and corresponding metric values
            for i, t_display in enumerate(display_time_points):
                current_time_row = data_start_row_for_metric + i

                # Color-code minute labels for readability (odd minutes red, even minutes orange)
                if "m" in t_display:
                    minute_num = int(t_display.replace("m", ""))
                    if minute_num % 2 == 1:
                        worksheet.write(current_time_row, 1, t_display, format_minute_red)
                    else:
                        worksheet.write(current_time_row, 1, t_display, format_minute_orange)
                else:
                    worksheet.write(current_time_row, 1, t_display, format_time_cell)

                # Determine which image number corresponds to this time point
                img_num_for_time = None
                if "s" in t_display:
                    seconds_val = int(t_display.replace("s", ""))
                    if seconds_val % time_interval_s == 0:
                        img_num_for_time = seconds_val // time_interval_s
                elif "m" in t_display:
                    seconds_val = int(t_display.replace("m", "")) * 60
                    if seconds_val % time_interval_s == 0:
                        img_num_for_time = seconds_val // time_interval_s

                # Write the metric value if available, else leave blank with border
                if img_num_for_time is not None and img_num_for_time * time_interval_s <= max_seconds_display:
                    if img_num_for_time in metrics_by_img_num:
                        metric_value = metrics_by_img_num[img_num_for_time].get(metric_key)
                        print(f"Writing '{metric_label}' ({metric_key}): {metric_value} at time {t_display}")
                        if metric_value is not None and isinstance(metric_value, (int, float)):
                            worksheet.write_number(current_time_row, 2, metric_value, format_data_cell)
                        else:
                            worksheet.write_blank(current_time_row, 2, None, format_empty_cell)
                    else:
                        worksheet.write_blank(current_time_row, 2, None, format_empty_cell)
                else:
                    worksheet.write_blank(current_time_row, 2, None, format_empty_cell)

            # Move the current row pointer downward for next metric plus some spacing
            current_excel_row += metric_name_height + 4

        # Extra spacing between conditions
        current_excel_row += 3

    # Adjust column widths for readability
    worksheet.set_column(0, 0, 5)   # Metric name column (narrow)
    worksheet.set_column(1, 1, 15)  # Time column
    worksheet.set_column(2, 2, 20)  # Values column

    try:
        # Save and close the Excel writer
        writer.close()
        print(f"Excel export successful: {excel_full_path}")
        messagebox.showinfo("Export Complete", f"Results saved successfully at:\n{excel_full_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        messagebox.showerror("Excel Export Error", f"Unable to save file: {e}")

def show_prediction_window(original, prediction_mask, title="Prediction", original_file_path_for_saving=None):
    # Create a new top-level Tkinter window
    win = tk.Toplevel()
    win.title(title)  # Set the window title

    original_disp = None
    if original.ndim == 2:
        # If original image is grayscale (2D array),
        # normalize to 0-255 and convert to BGR with green channel only for display
        norm_original = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        original_disp = cv2.merge([np.zeros_like(norm_original), norm_original, np.zeros_like(norm_original)])
    elif original.ndim == 3:
        if original.shape[2] == 3:
            # If original is a 3-channel BGR image, convert type to uint8 for display
            original_disp = original.astype(np.uint8)
        elif original.shape[2] == 4:
            # If original has 4 channels (e.g. BGRA), convert to BGR by removing alpha channel
            original_disp = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_BGRA2BGR)
        else:
            # Unsupported channel number: normalize grayscale and convert to BGR
            norm_original = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            original_disp = cv2.cvtColor(norm_original, cv2.COLOR_GRAY2BGR)
    else:
        # Unsupported image shape, create an error placeholder image with text
        print("Image shape not supported for display.")
        original_disp = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(original_disp, "Error Original", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    prediction_disp = None
    if prediction_mask.ndim == 2:
        # If prediction mask is grayscale:
        # - If float normalized between 0 and 1, scale to 0-255 uint8
        # - Otherwise, convert directly to uint8
        if prediction_mask.dtype == np.float32 or prediction_mask.dtype == np.float64:
            if np.max(prediction_mask) <= 1.0 and np.min(prediction_mask) >= 0.0:
                pred_mask_uint8 = (prediction_mask * 255).astype(np.uint8)
            else:
                pred_mask_uint8 = prediction_mask.astype(np.uint8)
        else:
            pred_mask_uint8 = prediction_mask.astype(np.uint8)
        # Convert grayscale mask to BGR for visualization
        prediction_disp = cv2.cvtColor(pred_mask_uint8, cv2.COLOR_GRAY2BGR)
    elif prediction_mask.ndim == 3 and prediction_mask.shape[2] == 3:
        # If prediction mask is already a 3-channel image, convert type to uint8
        prediction_disp = prediction_mask.astype(np.uint8)
    else:
        # Unsupported mask shape, create an error placeholder image with text
        print("Image shape not supported for display.")
        prediction_disp = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(prediction_disp, "Error Mask", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # Crop images if crop entry widgets exist
    if crop_top_entry and crop_bottom_entry:
        try:
            # Use user-defined function 'uf.crop_image' to crop images based on crop percentages from entries
            original_disp = uf.crop_image(original_disp, crop_top_entry.get(), crop_bottom_entry.get())
            prediction_disp = uf.crop_image(prediction_disp, crop_top_entry.get(), crop_bottom_entry.get())
        except Exception as e:
            print(f"Error during crop in show_prediction_window: {e}")
    else:
        print("Crop widgets not initialized, crop ignored in show_prediction_window.")

    # Determine the target height for displaying images side by side
    target_height = max(original_disp.shape[0], prediction_disp.shape[0])
    if target_height == 0:
        target_height = 100  # Fallback height if zero

    # If any image height is zero, replace with blank image of target height
    if original_disp.shape[0] == 0: original_disp = np.zeros((target_height, 100, 3), dtype=np.uint8)
    if prediction_disp.shape[0] == 0: prediction_disp = np.zeros((target_height, 100, 3), dtype=np.uint8)

    # Resize original image to target height maintaining aspect ratio
    if original_disp.shape[0] != target_height:
        ratio = target_height / original_disp.shape[0]
        original_disp = cv2.resize(original_disp, (int(original_disp.shape[1] * ratio), target_height))

    # Resize prediction mask image to target height maintaining aspect ratio
    if prediction_disp.shape[0] != target_height:
        ratio = target_height / prediction_disp.shape[0]
        prediction_disp = cv2.resize(prediction_disp, (int(prediction_disp.shape[1] * ratio), target_height))

    # Create a white vertical separator line between images
    white_line = 255 * np.ones((target_height, 10, 3), dtype=np.uint8)
    # Concatenate original, white line, and prediction images horizontally
    combined = np.hstack((original_disp, white_line, prediction_disp))

    # Convert BGR to RGB for PIL and Tkinter display
    img_rgb_display = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    img_pil_display = Image.fromarray(img_rgb_display)
    img_tk = ImageTk.PhotoImage(img_pil_display)

    # Create and pack a Tkinter Label to show the combined image
    label = tk.Label(win, image=img_tk)
    label.image = img_tk  # Keep a reference to avoid garbage collection
    label.pack()

    # If a file path is provided, save the comparison image
    if original_file_path_for_saving:
        try:
            base_dir = os.path.dirname(original_file_path_for_saving)
            results_subdir_name = "Results prediction"
            results_dir_path = os.path.join(base_dir, results_subdir_name)
            
            # Ensure the directory exists
            os.makedirs(results_dir_path, exist_ok=True)

            # Get custom filename from Tkinter variable prediction_file_name_var
            custom_name = prediction_file_name_var.get().strip()
            if custom_name:  # Use user-provided name if available
                comparison_image_filename = f"{custom_name}_comparison.png"
            else:
                # Otherwise, fallback to original file name stem
                original_filename_base = os.path.basename(original_file_path_for_saving)
                original_filename_stem, _ = os.path.splitext(original_filename_base)
                comparison_image_filename = f"{original_filename_stem}_comparison.png"
            
            full_save_path = os.path.join(results_dir_path, comparison_image_filename)

            # Save the combined PIL image to disk
            img_pil_display.save(full_save_path)
            print(f"Comparison saved: {full_save_path}")

        except Exception as e:
            print(f"Error when saving the comparison: {e}")

    # Calculate segmentation metrics using user function uf.calculate_metrics,
    # passing original image, prediction mask, and magnification factor from entry widget
    metrics_tuple = uf.calculate_metrics(original, prediction_mask, magnification_entry.get())
    # Format metrics into a multiline string for display
    metrics_text = (
        f"Fluorescence Intensity: {metrics_tuple[0]:.2f}\n"
        f"Surface Coverage: {metrics_tuple[1]*100:.2f}%\n"
        f"Mean Thrombi Fluorescence: {metrics_tuple[2]:.2f}\n"
        f"Number of Thrombi: {metrics_tuple[3]}\n"
        f"Mean Thrombus Area: {metrics_tuple[4]:.2f} pixels\n"
        f"Roundness of Thrombi: {metrics_tuple[5]:.2f}"
    )
    # Create and pack a Tkinter Label widget to show the metrics text below images
    metrics_label = tk.Label(win, text=metrics_text, font=("Aptos Narrow", 12), justify="left")
    metrics_label.pack(pady=10)

def on_close():
    # Function called when the prediction window is closed
    win.destroy()  # Close the window
    
    # If there is a file queue and the current index is less than the number of files
    if file_queue and current_index < len(file_queue):  # Changed <= to <
        process_next_image()  # Process the next image in the queue
    
    # If the last image was processed (current_index == length of queue)
    elif file_queue and current_index == len(file_queue):
        # This block handles exporting results to Excel after processing the last image
        if results_by_condition:
            # Sort results by image number for each condition
            for key in results_by_condition:
                results_by_condition[key].sort(key=lambda x: x['img_num_int'])
            # Export the sorted results to an Excel file
            export_to_excel_structured(results_by_condition, processed_folder_path)
        else:
            # If no results generated but files were processed, show info message
            if file_queue:
                messagebox.showinfo("Done", "Folder processing complete, but no results generated for export.")

# Bind the window close protocol to on_close function
win.protocol("WM_DELETE_WINDOW", on_close)


def start_processing_thread():
    # Start image processing in a separate daemon thread to keep UI responsive
    thread = threading.Thread(target=process_next_image)
    thread.daemon = True
    thread.start()


def choose_file():
    # Open a file dialog to select a single TIFF file
    file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
    if file_path:
        # Get suggested prediction file name based on the chosen file's metadata
        _, _, suggested_name = uf.get_condition_and_image_info(file_path)
        # If the prediction file name entry is empty, set it to the suggested name
        if not prediction_file_name_var.get().strip():
            prediction_file_name_var.set(suggested_name)

        try:
            # Load the image in unchanged mode (keeping original bit depth)
            img_original_cv = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img_original_cv is None:
                messagebox.showerror("Read Error", f"Unable to read image file: {file_path}")
                return

            # Convert image to float32 for processing
            img_original_float = img_original_cv.astype(np.float32)

            # Resize image for display purposes (512x512)
            img_for_display = cv2.resize(img_original_float, (512, 512))
            
            # Predict the segmentation mask using the model
            prediction = pr.predict_segmentation(file_path, MODEL_PATH)
            
            # Save the predicted mask to disk
            save_mask(prediction, file_path)
            
            # Show the prediction window with original image and prediction mask
            show_prediction_window(img_for_display, prediction,
                                   title=os.path.basename(file_path),
                                   original_file_path_for_saving=file_path)
            
        except Exception as e:
            # Show error messagebox and print error if processing fails
            messagebox.showerror("Processing Error", f"An error occurred during file processing:\n{e}")
            print(f"Error in choose_file: {e}")


def choose_folder():
    # Allow user to select a folder containing TIFF images to process in batch
    global file_queue, current_index, results_by_condition, processed_folder_path
    folder_path = filedialog.askdirectory()
    if folder_path:
        processed_folder_path = folder_path
        file_queue = []
        results_by_condition = {}
        current_index = 0
        
        # Collect all TIFF files from the folder into the processing queue
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(folder_path, filename)
                file_queue.append((file_path, filename))
        
        if file_queue:
            # Get suggested name from the first file to pre-fill prediction name entry
            first_file_path = file_queue[0][0]
            _, _, suggested_name_for_folder = uf.get_condition_and_image_info(first_file_path)
            # Use only the condition part for the suggested name
            prediction_file_name_var.set(suggested_name_for_folder.split("_Img")[0])
            # Start processing the batch in a background thread
            start_processing_thread()
        else:
            # Inform user if no TIFF files found in the selected folder
            messagebox.showinfo("No Files", "No .tif files found in the selected folder.")


def process_next_image():
    # Process images one by one from the file queue
    global current_index, results_by_condition

    if current_index < len(file_queue):
        # Get file path and filename of the current image to process
        file_path, filename = file_queue[current_index]
        # Extract condition key, image number string, and suggested name
        condition_key, img_num_str, suggested_name_for_prediction = uf.get_condition_and_image_info(file_path)
        
        # If prediction name entry is empty, fill it with suggested name
        if not prediction_file_name_var.get().strip():
            prediction_file_name_var.set(suggested_name_for_prediction)
        
        try:
            print(f"Processing: {file_path}")
            # Read image in unchanged mode
            img_cv_original = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img_cv_original is None:
                print(f"Error: Unable to read image {file_path}")
                current_index += 1
                process_next_image()  # Skip to next image if reading fails
                return

            # Convert image to float32
            img_float_original = img_cv_original.astype(np.float32)
            # Resize for display
            img_for_display = cv2.resize(img_float_original, (512, 512))
            
            # Get segmentation prediction
            prediction = pr.predict_segmentation(file_path, MODEL_PATH)
            # Calculate metrics based on original image and prediction mask
            metrics_values = uf.calculate_metrics(img_float_original, prediction)
            fluorescence_intensity, surface_coverage, mean_thrombi_fluorescence, actual_num_thrombi, mean_thrombus_area, mean_roundness = metrics_values

            # Get metadata again (redundant but safe)
            condition_key, img_num_str, suggested_name_for_prediction = uf.get_condition_and_image_info(file_path)
            
            # Update prediction file name with suggested name
            prediction_file_name_var.set(suggested_name_for_prediction)

            try:
                # Try to convert image number string to integer for sorting
                img_num_int = int(img_num_str)
            except ValueError:
                print(f"Error: img_num_str '{img_num_str}' is not a valid int for {file_path}. Using 0.")
                img_num_int = current_index  # Fallback to current index if conversion fails

            # Initialize the condition key list if not already present
            if condition_key not in results_by_condition:
                results_by_condition[condition_key] = []

            # Append metrics and metadata for this image to the results dictionary
            results_by_condition[condition_key].append({
                "img_num_int": img_num_int,
                "original_filename": filename,
                "metrics": {
                    "mean_intensity": fluorescence_intensity,
                    "surface": surface_coverage * 100,
                    "mean_thrombi_fluorescence": mean_thrombi_fluorescence,
                    "count": actual_num_thrombi,
                    "mean_area": mean_thrombus_area,
                    "roundness": mean_roundness
                }
            })
            
            # Save prediction mask to disk
            save_mask(prediction, file_path)

            # Show prediction window with original and mask
            show_prediction_window(img_for_display, prediction,
                                   title=filename,
                                   original_file_path_for_saving=file_path)
            # Move to next file index
            current_index += 1

        except Exception as e:
            # Print and show warning if processing fails, then continue with next file
            print(f"Error when processing {filename}: {e}")
            messagebox.showwarning("Processing Error", f"Error on {filename}:\n{e}\nNext file.")
            current_index += 1
            process_next_image()

    else:
        # If no files queued at all, inform user processing is done
        if not file_queue:
            messagebox.showinfo("Done", "No files were queued for processing.")


def update_entry_state(option_var, entry_widget, default_map=None, selected_value=None):
    # Enable or disable an entry widget depending on the option selected in a dropdown
    current_selection = option_var.get()

    if current_selection == "Personnalised":
        # If "Personnalised" option selected, enable entry and clear content
        entry_widget.config(state="normal")
        entry_widget.delete(0, tk.END)
    else:
        # Otherwise, set entry to current selection value (removing 'x'), then disable it
        entry_widget.config(state="normal")
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, current_selection.replace("x",""))
        entry_widget.config(state="disabled")

# --- Main Interface ---

root = tk.Tk()  # Create the main Tkinter window
root.title("Segmentation Viewer")  # Set the window title
root.geometry("700x350")  # Set the window size (width x height)
root.configure(bg="#f0f0f0")  # Set the window background color (light gray)

style = ttk.Style()  # Initialize a ttk style object to customize widgets
style.configure("TFrame", background="#f0f0f0")  # Set background color for ttk Frames
style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 12))  # Set bg color and font for ttk Labels
style.configure("TButton", font=("Helvetica", 12), padding=5)  # Set font and padding for ttk Buttons

settings_frame = tk.Frame(root, bg="#f0f0f0")  # Create a Frame widget for settings with a gray background
settings_frame.pack(pady=10, padx=10, fill="x")  # Pack frame with padding and make it fill horizontally

tk.Label(settings_frame, text="Name for saved prediction image:", bg="#f0f0f0").grid(row=0, column=0, sticky="w", padx=5, pady=2)  
# Create a label inside settings_frame with given text, background color, placed in row 0 column 0, aligned left with padding

prediction_file_name_var = tk.StringVar(value="")  # Create a Tkinter string variable initialized as empty string
tk.Entry(settings_frame, textvariable=prediction_file_name_var, width=30).grid(row=0, column=1, sticky="ew", padx=5, pady=2)  
# Create an Entry widget linked to the string variable, width 30, placed row 0 column 1, expands horizontally with padding

# Magnification
magnification_options = ["10", "20", "40", "60", "Personnalised"]  # Define options for magnification (zoom levels)
tk.Label(settings_frame, text="Magnification:", bg="#f0f0f0").grid(row=1, column=0, sticky="w", padx=5, pady=2)  
# Label for magnification option, placed row 1 column 0

magnification_choice = tk.StringVar(value=last_magnification)  # StringVar for the selected magnification, initialized to last_magnification
magnification_entry = tk.Entry(settings_frame, width=10)  # Entry widget for custom magnification value
magnification_entry.insert(0, last_magnification)  # Insert the default last_magnification value into the entry
magnification_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)  # Place the entry in row 1 column 1, aligned left with padding

update_entry_state(magnification_choice, magnification_entry)  
# Call function to update the state (enabled/disabled) of the entry based on the current choice

magnification_menu = tk.OptionMenu(settings_frame, magnification_choice, *magnification_options,
                                   command=lambda selected_val: update_entry_state(magnification_choice, magnification_entry))
# Create an OptionMenu widget for magnification choices, linked to magnification_choice
# On selection, update the entry state accordingly

magnification_menu.grid(row=1, column=2, sticky="w", padx=5, pady=2)  
# Place the OptionMenu in row 1 column 2, aligned left with padding

# Crop options
crop_percentage_options = ["0", "5", "10", "15", "20", "30", "50", "70", "85", "100", "Personnalised"]  
# List of percentage options for cropping top and bottom of images

# Crop Top
tk.Label(settings_frame, text="Crop top (% from top):", bg="#f0f0f0").grid(row=2, column=0, sticky="w", padx=5, pady=2)  
# Label for top crop percentage, placed row 2 column 0

crop_top_choice = tk.StringVar(value="0")  # StringVar to hold top crop choice, default "0%"
crop_top_entry = tk.Entry(settings_frame, width=10)  # Entry widget to allow custom crop top value
crop_top_entry.insert(0, "0")  # Insert default value "0" into entry
crop_top_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)  # Place entry widget row 2 column 1, aligned left with padding

update_entry_state(crop_top_choice, crop_top_entry)  
# Update the entry widget state based on crop_top_choice

crop_top_menu = tk.OptionMenu(settings_frame, crop_top_choice, *crop_percentage_options,
                              command=lambda selected_val: update_entry_state(crop_top_choice, crop_top_entry))  
# Create OptionMenu for crop top percentage choices linked to crop_top_choice
# On selection, update entry state

crop_top_menu.grid(row=2, column=2, sticky="w", padx=5, pady=2)  
# Place crop top OptionMenu in row 2 column 2, aligned left with padding

# Crop Bottom
tk.Label(settings_frame, text="Crop bottom (% from top):", bg="#f0f0f0").grid(row=3, column=0, sticky="w", padx=5, pady=2)  
# Label for bottom crop percentage, placed row 3 column 0

crop_bottom_choice = tk.StringVar(value="100")  # StringVar for crop bottom choice, default 100%
crop_bottom_entry = tk.Entry(settings_frame, width=10)  # Entry widget for custom crop bottom value
crop_bottom_entry.insert(0, "100")  # Insert default value "100" into entry
crop_bottom_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)  # Place entry row 3 column 1, aligned left with padding

update_entry_state(crop_bottom_choice, crop_bottom_entry)  
# Update entry widget state based on crop_bottom_choice

crop_bottom_menu = tk.OptionMenu(settings_frame, crop_bottom_choice, *crop_percentage_options,
                                 command=lambda selected_val: update_entry_state(crop_bottom_choice, crop_bottom_entry))  
# Create OptionMenu for crop bottom choices linked to crop_bottom_choice
# On selection, update entry state accordingly

crop_bottom_menu.grid(row=3, column=2, sticky="w", padx=5, pady=2)  
# Place crop bottom OptionMenu in row 3 column 2, aligned left with padding

settings_frame.columnconfigure(1, weight=1)  
# Make column 1 in settings_frame expandable (so entry widgets can stretch horizontally)

buttons_frame = tk.Frame(root, bg="#f0f0f0")  # Create a Frame for buttons with gray background
buttons_frame.pack(pady=20)  # Pack the buttons_frame with vertical padding

btn_style = {
    "font": ("Helvetica", 12),  # Font for buttons
    "bg": "#4CAF50",            # Background color (green)
    "fg": "white",              # Text color white
    "activebackground": "#45a049",  # Background color when button is active (hovered/clicked)
    "activeforeground": "white",     # Text color when active
    "width": 20,                # Button width
    "padx": 10,                 # Padding inside button horizontally
    "pady": 10,                 # Padding inside button vertically
    "relief": "flat"            # Flat style button (no border)
}

tk.Button(buttons_frame, text="Choose a file", command=choose_file, **btn_style).pack(pady=5)  
# Create a button to choose a single file, styled with btn_style, packed with vertical padding

tk.Button(buttons_frame, text="Choose a folder", command=choose_folder, **btn_style).pack(pady=5)  
# Create a button to choose a folder, styled with btn_style, packed with vertical padding

root.mainloop()  # Start the Tkinter event loop (keeps window open and responsive)

