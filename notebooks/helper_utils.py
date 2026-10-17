"""
Consolidated helper utilities for PyTorch deep learning projects.

Merged from the Course 1, 2 and 3 helper files with duplicate functions removed.
Every function starts with a one-line comment that says what it does.
Course-specific values (class names, folder layouts, normalization stats, label maps,
model call signatures) are arguments, not hardcoded, so the functions work on other projects.

Hard requirements: torch, torchvision, numpy, matplotlib, pillow.
Everything else (pandas, seaborn, sklearn, torchmetrics, transformers, diffusers,
IPython, ipywidgets, mlflow, cv2, ...) is imported only if installed. Functions
that need a missing package will fail when called, not when this file is imported.
"""
from __future__ import annotations
import ast
import copy
import gc
import glob
import inspect
import io
import itertools
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as tv_models
from PIL import ExifTags, Image, Image as PILImage, UnidentifiedImageError
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torchvision.utils import make_grid
try:
    from torch.amp import GradScaler, autocast
except ImportError:  # torch < 2.3 keeps GradScaler in torch.cuda.amp
    from torch.cuda.amp import GradScaler
    from torch.amp import autocast
try:
    from IPython.display import HTML, Markdown, clear_output, display
except ImportError:
    HTML = None
    Markdown = None
    clear_output = None
    display = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from directory_tree import DisplayTree
except ImportError:
    DisplayTree = None

try:
    from fastai.vision.all import show_image, show_titled_image
except ImportError:
    show_image = None
    show_titled_image = None

try:
    import imageio
except ImportError:
    imageio = None

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

try:
    import mlflow
except ImportError:
    mlflow = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import requests
except ImportError:
    requests = None

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    ConfusionMatrixDisplay = None
    classification_report = None
    compute_class_weight = None
    confusion_matrix = None
    f1_score = None

try:
    import torchmetrics
    from torchmetrics import Accuracy, F1Score
    from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix, Precision, Recall
except ImportError:
    Accuracy = None
    F1Score = None
    MulticlassAccuracy = None
    MulticlassConfusionMatrix = None
    Precision = None
    Recall = None
    torchmetrics = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BertModel, BertTokenizer, BlipForQuestionAnswering, BlipProcessor
except ImportError:
    AutoModel = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    BertModel = None
    BertTokenizer = None
    BlipForQuestionAnswering = None
    BlipProcessor = None
# Pause in seconds between live-plot refreshes
FRAME_PAUSE = 0.15
# Frames per second of the saved training GIF
GIF_FPS = 6
# Default image transforms used by get_dataloaders (ImageNet-style crop with dataset mean and std)
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.630, 0.554, 0.489],
                         std=[0.248, 0.271, 0.319]),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.630, 0.554, 0.489],
                         std=[0.248, 0.271, 0.319]),
])


# =============================================================================
# NOTEBOOK DISPLAY HELPERS
# =============================================================================


# Displays the source code of a function as a markdown block
def display_function(func):
    """
    Renders the source code of a given function as a formatted Markdown block.

    Args:
        func: The function object whose source code is to be displayed.
    """
    # Retrieve the source code of the function as a raw string.
    source_code_str = inspect.getsource(func)
    # Format the raw string into a Python code block for Markdown rendering.
    markdown_formatted_code = f"```python\n{source_code_str}\n```"
    # Display the formatted code block as Markdown output.
    display(Markdown(markdown_formatted_code))


# Shows a torchinfo summary as a styled HTML table
def display_torch_summary(summary_object, attr_names, display_names, depth):
    """
    Displays a torchinfo summary object as a styled HTML table.

    This function processes a summary object from the torchinfo library,
    formats it into a pandas DataFrame, and then renders it as a clean,
    readable HTML table within a Jupyter environment. It also displays
    key summary statistics like total parameters and memory usage below the table.

    Args:
        summary_object: The object returned by `torchinfo.summary()`.
        attr_names (list): A list of the layer attribute names to extract
                           (e.g., 'input_size', 'num_params').
        display_names (list): A list of the desired column headers for the
                              output table (e.g., 'Input Shape', 'Param #').
        depth (int, optional): The maximum depth of layers to display.
                               Defaults to infinity (showing all layers).
    """

    layer_data = []
    # Define the table column headers for the DataFrame with the new name.
    display_columns = ["Layer (type (var_name):depth-idx)"] + display_names

    for layer in summary_object.summary_list:
        # Only process layers that are within the specified depth.
        if layer.depth > depth:
            continue

        row = {}

        # Construct the hierarchical layer name with the var_name.
        indent = "&nbsp;"*4*layer.depth
        # NEW: Construct the layer name with the var_name included.
        layer_name = f"{layer.class_name} ({layer.var_name})"
        if layer.depth > 0:
            # Append depth and index for nested layers.
            layer_name = f"{layer_name}: {layer.depth}-{layer.depth_index}"

        row["Layer (type (var_name):depth-idx)"] = f"{indent}{layer_name}"

        # Iterate over both attribute and display names to populate row data.
        for attr, name in zip(attr_names, display_names):
            if attr == "num_params":
                # Mimic torchinfo's logic for displaying parameters.
                show_params = layer.is_leaf_layer or layer.depth == depth
                if show_params and layer.num_params > 0:
                    value = f"{layer.num_params:,}"
                else:
                    value = "--"
            else:
                # Fetch all other attributes directly.
                value = getattr(layer, attr, "N/A")

            row[name] = value
        layer_data.append(row)

    df = pd.DataFrame(layer_data, columns=display_columns)

    # Style the DataFrame for clean HTML presentation.
    styler = df.style.hide(axis="index")
    styler.set_table_styles([
        {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse")]},
        {"selector": "th", "props": [
            ("text-align", "left"), ("padding", "8px"),
            ("background-color", "#4f4f4f"), ("color", "white"),
            ("border-bottom", "1px solid #ddd")
        ]},
        {"selector": "td", "props": [
            ("text-align", "left"), ("padding", "8px"),
            ("border-bottom", "1px solid #ddd")
        ]},
    ]).set_properties(**{"white-space": "pre", "vertical-align": "top"})

    table_html = styler.to_html()

    # --- Summary Statistics ---
    total_params = f"{summary_object.total_params:,}"
    trainable_params = f"{summary_object.trainable_params:,}"
    non_trainable_params = f"{summary_object.total_params - summary_object.trainable_params:,}"
    total_mult_adds = f"{summary_object.total_mult_adds/1e9:.2f} GB"

    params_html = f"""
    <div style="margin-top: 20px; font-family: monospace; line-height: 1.6;">
        <hr><p><b>Total params:</b> {total_params}</p>
        <p><b>Trainable params:</b> {trainable_params}</p>
        <p><b>Non-trainable params:</b> {non_trainable_params}</p>
        <p><b>Total mult-adds:</b> {total_mult_adds}</p><hr>
    </div>"""

    input_size_mb = summary_object.total_input/(1024**2)
    fwd_bwd_pass_size_mb = summary_object.total_output_bytes/(1024**2)
    params_size_mb = summary_object.total_param_bytes/(1024**2)
    total_size_mb = (
        summary_object.total_input +
        summary_object.total_output_bytes +
        summary_object.total_param_bytes
    )/(1024**2)

    size_html = f"""
    <div style="font-family: monospace; line-height: 1.6;">
        <p><b>Input size (MB):</b> {input_size_mb:.2f}</p>
        <p><b>Forward/backward pass size (MB):</b> {fwd_bwd_pass_size_mb:.2f}</p>
        <p><b>Params size (MB):</b> {params_size_mb:.2f}</p>
        <p><b>Estimated Total Size (MB):</b> {total_size_mb:.2f}</p><hr>
    </div>"""

    # Combine all HTML parts and display.
    final_html = table_html + params_html + size_html
    display(HTML(final_html))


# Displays a markdown table comparing the size and inference time of all models
def display_full_comparison(stats_dict):
    """
    Displays a markdown table comparing the size and inference time of all models.

    Args:
        stats_dict (dict): A dictionary where keys are model names (str) and 
                           values are tuples containing the model size (float) 
                           and inference time (float).

    Returns:
        None: This function renders a Markdown table directly to the display output.
    """
    
    # Start building the markdown table string with headers
    table = "| Model Type | Size (MB) | Inference Time (ms) |\n"
    
    # Add the separator row for the markdown table
    table += "|--------------------------|-----------|-----------------------|\n"
    
    # Iterate through the dictionary to add a row for each model entry
    for model_name, (size, time) in stats_dict.items():
        # Format the model name, size, and time into a table row string
        table += f"| {model_name:<24} | {size:<9.2f} | {time:<21.2f} |\n"

    # Render and display the final markdown table string
    display(Markdown(table))


# Displays a model comparison table in a Jupyter-like environment
def comparison_table(baseline_model_size, baseline_model_time, quantized_model_size, quantized_model_time, quantization_type):
    """
    Displays a model comparison table in a Jupyter-like environment.

    Args:
        baseline_model_size (float): The size of the baseline model in megabytes.
        baseline_model_time (float): The average inference latency of the baseline model in milliseconds.
        quantized_model_size (float): The size of the quantized model in megabytes.
        quantized_model_time (float): The average inference latency of the quantized model in milliseconds.
        quantization_type (str): The method of quantization used (e.g., 'Static', 'Dynamic', 'QAT').

    Returns:
        None: This function directly renders a Markdown table to the display output.
    """
    # Calculate the reduction in model size
    size_diff = baseline_model_size - quantized_model_size
    
    # Calculate the reduction in inference latency
    time_diff = baseline_model_time - quantized_model_time

    # Dynamically create the header for the quantized model column based on the technique used
    quantized_header = f"Quantized Model ({quantization_type})"

    # Construct a formatted Markdown table string to compare baseline and quantized metrics
    markdown_table_string = f"""
| | Baseline Model | {quantized_header} | Change |
|:---|:---|:---|:---|
| **Model Size (MB)** | {baseline_model_size:.2f} | {quantized_model_size:.2f} | {size_diff:.2f} |
| **Inference Latency (ms)** | {baseline_model_time:.2f} | {quantized_model_time:.2f} | {time_diff:.2f} |
"""

    # Directly render and display the Markdown table in the output cell
    display(Markdown(markdown_table_string))


# Shows an HTML table comparing a baseline and a quantized BLIP model
def blip_comparison_table(
    question,
    baseline_answer,
    quantized_answer,
    baseline_size,
    quantized_size,
    baseline_time_s,
    quantized_time_s
):
    """
    Displays a formatted HTML table comparing the performance of the baseline
    and quantized BLIP model.

    Args:
        question (str): The text of the question asked to the model.
        baseline_answer (str): The answer generated by the original model.
        quantized_answer (str): The answer generated by the quantized model.
        baseline_size (float): The storage size of the baseline model in megabytes.
        quantized_size (float): The storage size of the quantized model in megabytes.
        baseline_time_s (float): The inference duration for the baseline model in seconds.
        quantized_time_s (float): The inference duration for the quantized model in seconds.

    Returns:
        None: This function renders an HTML table directly to the display output.
    """
    # Calculate the absolute difference in model size
    size_change = baseline_size - quantized_size
    
    # Initialize a default text for size change if calculation is not possible
    size_change_text = "N/A"
    
    # Check if baseline size is valid to calculate percentage changes
    if baseline_size > 0:
        # Calculate the percentage change in model size
        size_percent_change = (size_change / baseline_size) * 100
        
        # Determine if the change represents a reduction or an increase
        size_change_label = "reduction" if size_change >= 0 else "increase"
        
        # Format the size change string with absolute and percentage values
        size_change_text = f"{size_change:.2f} ({abs(size_percent_change):.1f}% {size_change_label})"

    # Calculate the absolute difference in inference time
    time_change_s = baseline_time_s - quantized_time_s
    
    # Initialize a default text for time change if calculation is not possible
    time_change_text = "N/A"
    
    # Check if baseline time is valid to calculate percentage changes
    if baseline_time_s > 0:
        # Calculate the percentage change in inference latency
        time_percent_change = (time_change_s / baseline_time_s) * 100
        
        # Determine if the change represents a reduction or an increase in speed
        time_change_label = "reduction" if time_change_s >= 0 else "increase"
        
        # Format the time change string with absolute and percentage values
        time_change_text = f"{time_change_s:.4f} ({abs(time_percent_change):.1f}% {time_change_label})"
        
    # Construct the HTML template for the comparison table
    html_string = f"""
    <p><b>Question:</b> {question}</p>
    <table border="1" style="width:100%; border-collapse: collapse; text-align: left;">
      <tr style="background-color: #4A5568; color: white;">
        <th style="padding: 8px;"></th>
        <th style="padding: 8px;">Model Size (MB)</th>
        <th style="padding: 8px;">Inference Time (s)</th>
        <th style="padding: 8px;">Answer</th>
      </tr>
      <tr>
        <td style="padding: 8px;"><b>Baseline Model</b></td>
        <td style="padding: 8px;">{baseline_size:.2f}</td>
        <td style="padding: 8px;">{baseline_time_s:.4f}</td>
        <td style="padding: 8px;">{baseline_answer}</td>
      </tr>
      <tr>
        <td style="padding: 8px;"><b>Quantized Model (Dynamic)</b></td>
        <td style="padding: 8px;">{quantized_size:.2f}</td>
        <td style="padding: 8px;">{quantized_time_s:.4f}</td>
        <td style="padding: 8px;">{quantized_answer}</td>
      </tr>
      <tr>
        <td style="padding: 8px;"><b>Change</b></td>
        <td style="padding: 8px;"><b>{size_change_text}</b></td>
        <td style="padding: 8px;"><b>{time_change_text}</b></td>
        <td style="padding: 8px;">---</td>
      </tr>
    </table>
    """

    # Render and display the generated HTML content in the notebook environment
    display(HTML(html_string))


# Shows a Jupyter widget that uploads and validates one JPG image
def upload_jpg_widget(output_image_folder="./images"):
    """
    Creates and displays a file upload widget for JPG images.

    This function facilitates the uploading of JPG files within a Jupyter
    or IPython environment. It handles file validation for format and size,
    corrects image orientation based on EXIF data, resizes large images
    while preserving aspect ratio, and saves the final image to a local
    directory.
    """
    # Uploaded images are stored in output_image_folder.
    # Create the destination directory if it does not already exist.
    os.makedirs(output_image_folder, exist_ok=True)

    # Initialize the file upload widget, accepting only JPG/JPEG files.
    uploader = widgets.FileUpload(
        accept='.jpg,.jpeg',
        multiple=False,
        description='Upload JPG (Max 5MB)'
    )

    # Create an output widget to display messages to the user.
    output_area = widgets.Output()

    def on_file_uploaded(change):
        """
        Processes the uploaded file upon a change event.

        This callback function is triggered when a file is uploaded. It validates
        the file's format and size, corrects orientation, resizes if necessary,
        and saves the processed image.

        Args:
            change: A dictionary-like object containing information about the change event.
                    The new file data is in `change['new']`.
        """
        # Retrieve the tuple of uploaded file data from the change event.
        current_uploaded_value_tuple = change['new']
        # If the tuple is empty, it means the upload was cleared, so do nothing.
        if not current_uploaded_value_tuple:
            return

        # Use the output area to display feedback.
        with output_area:
            # Clear any previous messages.
            output_area.clear_output()

            # Get the dictionary containing file data from the tuple.
            file_data_dict = current_uploaded_value_tuple[0]
            # Extract the filename and its binary content.
            filename = file_data_dict['name']
            file_content = file_data_dict['content']

            # Validate that the file has a '.jpg' or '.jpeg' extension.
            if not filename.lower().endswith(('.jpg', '.jpeg')):
                # Format an error message for invalid file types.
                error_msg_format = (
                    f"<p style='color:red;'>Error: Please upload a file with a ‘.jpg’ or ‘.jpeg’ format. "
                    f"You uploaded: '{filename}'</p>"
                )
                # Display the error message and reset the uploader.
                display(HTML(error_msg_format))
                uploader.value = ()
                return

            # Check if the file size exceeds the 5MB limit.
            if len(file_content) > 5 * 1024 * 1024:
                # Calculate file size in megabytes for the error message.
                file_size_mb = len(file_content) / (1024 * 1024)
                # Format an error message for oversized files.
                error_msg_size = (
                    f"<p style='color:red;'>Error: File '{filename}' is too large ({file_size_mb:.2f} MB). "
                    f"Please upload a file less than or equal to 5 MB.</p>"
                )
                # Display the error message and reset the uploader.
                display(HTML(error_msg_size))
                uploader.value = ()
                return

            try:
                # Open the image from its binary content.
                img = Image.open(io.BytesIO(file_content))

                # Attempt to correct image orientation using EXIF data.
                try:
                    # Map EXIF tags to their names for easier lookup.
                    orientation_map = {
                        ExifTags.TAGS[k]: k for k in ExifTags.TAGS if k in ExifTags.TAGS
                    }
                    # Retrieve the EXIF data from the image.
                    exif = img._getexif()

                    # Check if orientation data exists in the EXIF information.
                    if exif and orientation_map['Orientation'] in exif:
                        orientation = exif[orientation_map['Orientation']]
                        # Apply rotation based on the orientation value.
                        if orientation == 3:
                            img = img.transpose(Image.ROTATE_180)
                        elif orientation == 6:
                            img = img.transpose(Image.ROTATE_270)
                        elif orientation == 8:
                            img = img.transpose(Image.ROTATE_90)
                # Handle cases where EXIF data is missing or corrupt.
                except (AttributeError, KeyError, IndexError):
                    pass

                # Get the dimensions of the potentially reoriented image.
                width, height = img.size
                # Check if the image dimensions exceed the 1000x1000 pixel limit.
                if width > 1000 or height > 1000:
                    # Calculate the scaling factor to maintain the aspect ratio.
                    scaling_factor = 1000 / max(width, height)
                    # Compute the new dimensions.
                    new_width = int(width * scaling_factor)
                    new_height = int(height * scaling_factor)
                    # Resize the image using a high-quality downsampling filter.
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Create an in-memory byte stream to save the processed image.
                output_byte_stream = io.BytesIO()
                # Save the image to the byte stream in JPEG format.
                img.save(output_byte_stream, format='JPEG', quality=90)
                # Get the binary content from the stream.
                content_to_write = output_byte_stream.getvalue()

                # Define the full path to save the file.
                save_path = os.path.join(output_image_folder, filename)
                # Write the final image content to a file on disk.
                with open(save_path, 'wb') as f:
                    f.write(content_to_write)

                # Create a string representation of the path for user-friendly output.
                python_code_path = repr(save_path)
                # Format a success message for the user.
                success_message = f"""
                <p style='color:green;'>File successfully uploaded!</p>
                <p>Please use the path as <code>image_path = {python_code_path}</code></p>
                """
                # Display the success message.
                display(HTML(success_message))

            # Catch any exceptions that occur during image processing.
            except Exception as e:
                # Format and display a generic error message.
                error_msg_save = f"<p style='color:red;'>Error processing file '{filename}': {e}</p>"
                display(HTML(error_msg_save))
            # The finally block ensures the uploader is cleared regardless of success or failure.
            finally:
                uploader.value = ()

    # Register the callback function to be executed when a file is uploaded.
    uploader.observe(on_file_uploaded, names='value')
    # Display the file upload widget.
    display(uploader)
    # Display the area for output messages.
    display(output_area)


# Shows a model's layers, types and parameter counts as a styled HTML table
def display_model_architecture(model):
    """
    Traverses the provided PyTorch model and displays its architecture as a 
    styled HTML table, including hierarchy, layer types, and parameter counts.

    Arguments:
        model: The PyTorch model instance to inspect.
    """
    # Initialize a list to hold the dictionary data for each layer
    layer_data = []
    
    # Iterate through the immediate child modules of the network
    for name, module in model.named_children():
        # Calculate the total number of parameters in the current module
        params = sum(p.numel() for p in module.parameters())
        
        # Determine the layer type string
        if isinstance(module, torch.nn.Conv2d):
            layer_type = f"Conv2d {module.kernel_size}"
        else:
            layer_type = module.__class__.__name__
            
        # Append the parent layer information to the data list
        layer_data.append({
            "Layer Hierarchy": f"<b>{name}</b>",
            "Layer Type": layer_type,
            "Parameters": f"{params:,}"
        })
        
        # Check if the module is a container (Sequential) to inspect sub-layers
        if isinstance(module, torch.nn.Sequential):
            # Iterate through the sub-layers within the sequential block
            for sub_name, sub_module in module.named_children():
                # Calculate parameters for the sub-module
                sub_params = sum(p.numel() for p in sub_module.parameters())
                
                # Determine the sub-layer type string
                if isinstance(sub_module, torch.nn.Conv2d):
                    sub_type = f"Conv2d {sub_module.kernel_size}"
                else:
                    sub_type = sub_module.__class__.__name__
                
                # Append the sub-layer information with visual indentation
                layer_data.append({
                    "Layer Hierarchy": f"&nbsp;&nbsp;&nbsp;&nbsp;└─ {sub_name}",
                    "Layer Type": sub_type,
                    "Parameters": f"{sub_params:,}"
                })

    # Create a Pandas DataFrame from the collected layer data
    df = pd.DataFrame(layer_data)

    # Apply styling to the DataFrame for a clean HTML presentation
    styler = df.style.hide(axis="index")
    styler.set_table_styles([
        {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse"), ("font-family", "sans-serif")]},
        {"selector": "th", "props": [
            ("text-align", "left"), ("padding", "10px"), 
            ("background-color", "#4f4f4f"), ("color", "white"),
            ("border-bottom", "1px solid #ddd")
        ]},
        {"selector": "td", "props": [
            ("text-align", "left"), ("padding", "8px"), 
            ("border-bottom", "1px solid #ddd")
        ]},
    ])
    
    # Render the styled table to HTML
    table_html = styler.to_html()

    # Calculate the total number of parameters in the entire network
    total_params = sum(p.numel() for p in model.parameters())
    
    # Create a summary HTML block for total parameters
    summary_html = f"""
    <div style="margin-top: 15px; font-family: monospace; font-size: 1.1em;">
        <p><b>Total Parameters:</b> {total_params:,}</p>
        <hr>
    </div>
    """

    # Combine the table and summary HTML
    final_html = table_html + summary_html
    
    # Display the final HTML output in the notebook
    display(HTML(final_html))


# Display the 4 main layers of the TranslationEncoder model
def show_model_layers(model):
    """
    Display the 4 main layers of the TranslationEncoder model.
    """
    print("\n" + "=" * 70)
    print(f" {model.__class__.__name__} - Main Layers")
    print("=" * 70)
    print(f"\n{'Layer':<30} {'Type':<25} {'Parameters':>15}")
    print("-" * 70)
    
    # Show the 4 main layers
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_type = module.__class__.__name__
        print(f"{name:<30} {module_type:<25} {params:>15,}")
    
    print("-" * 70)
    total = sum(p.numel() for p in model.parameters())
    print(f"{'TOTAL':<30} {'':<25} {total:>15,}")
    print("=" * 70)



# =============================================================================
# PLOT STYLING AND GENERIC PLOTS
# =============================================================================


# Returns the DeepLearning.AI color map and a matplotlib style dict
def apply_dlai_style():
    # Global plot style
    PLOT_STYLE = {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "font.family": "sans",  # "sans-serif",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 3,
        "lines.markersize": 6,
    }

    # mpl.rcParams.update(PLOT_STYLE)

    # Custom colors (reusable)
    color_map = {
        "pink": "#F65B66",
        "blue": "#1C74EB",
        "yellow": "#FAB901",
        "red": "#DD3C66",
        "purple": "#A12F9D",
        "cyan": "#237B94",
    }
    return color_map, PLOT_STYLE


# Default DeepLearning.AI colors and style, built once so other functions can use them as defaults
color_map, PLOT_STYLE = apply_dlai_style()

# Creates a subplot grid and always returns the axes as a flat array
def get_grid(num_rows, num_cols, figsize=(16, 8)):
    """
    Creates a grid of subplots and ensures the axes object is consistently formatted.

    Args:
        num_rows (int): The number of rows in the subplot grid.
        num_cols (int): The number of columns in the subplot grid.
        figsize (tuple): The dimensions of the overall figure.

    Returns:
        fig (matplotlib.figure.Figure): The generated matplotlib figure object.
        axes (list): The formatted axes grid, structured as an iterable or 2D list.
    """
    # Create a figure and a set of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Handle the case where there is only one row.
    if num_rows == 1:
        # Ensure the axes object is iterable for consistency.
        axes = [axes]
    # Handle the case where there is only one column.
    elif num_cols == 1:
        # Ensure the axes object is a 2D list for consistent indexing.
        axes = [[ax] for ax in axes]
        
    # Return the figure and the formatted axes grid.
    return fig, axes


# Plots a bar chart of parameter counts per layer
def plot_counting(counting_params):
    import matplotlib.pyplot as plt

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.bar(counting_params.keys(), counting_params.values())
    plt.xlabel("Layer Name")
    plt.ylabel("Number of Parameters")
    plt.title("Number of Parameters in Each Terminal Layer")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Displays side-by-side plots for training loss and test accuracy over epochs
def plot_metrics(train_loss, test_acc):
    """
    Displays side-by-side plots for training loss and test accuracy over epochs.

    Args:
        train_loss (list): A list of floating-point numbers representing the
                           average training loss for each epoch.
        test_acc (list): A list of floating-point numbers representing the
                         test accuracy for each epoch.
    """
    # Get the number of epochs from the length of the loss list
    num_epochs = len(train_loss)
    # Create a 1-based epoch range for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Training Loss ---
    ax1.plot(epochs, train_loss, marker='o', linestyle='-', color='royalblue')
    ax1.set_title('Training Loss Over Epochs', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True)
    # Ensure the x-axis ticks are integers
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Plot 2: Test Accuracy ---
    ax2.plot(epochs, test_acc, marker='o', linestyle='-', color='red')
    ax2.set_title('Test Accuracy Over Epochs', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True)
    # Ensure the x-axis ticks are integers
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust layout to prevent overlap and display the plots
    plt.tight_layout()
    plt.show()


# Plots the training loss over a series of epochs
def plot_loss(losses):
    """
    Plots the training loss over a series of epochs.

    Args:
        losses: A list or array of loss values, where each value
                corresponds to an epoch.
    """
    
    # Create a new figure with a specific size for the plot.
    plt.figure(figsize=(8, 8))
    # Plot the loss values against the epoch number.
    plt.plot(losses)
    # Set the title and labels for clarity.
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # Add a grid to make the plot easier to read.
    plt.grid(True)
    # Display the plot.
    plt.show()


# Plots training and validation loss from a history dict
def plot_loss_history(history):
    """
    Plot training and validation loss from training history.
    
    Args:
        history: Dictionary containing 'train_loss' and 'val_loss' lists
    """
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'],
             label='Training Loss', marker='o')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'],
             label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print final results
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Best Validation Loss: {min(history['val_loss']):.4f}")
    best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    print(f"Best Validation Loss at Epoch: {best_epoch}")


# Plots training and test accuracy from a history dict against a baseline
def plot_accuracy_history(history, initial_accuracy=50.0):
    """
    Generates a visualization of model performance metrics over the training duration.

    Args:
        history: A dictionary containing 'train_acc' and 'test_acc' lists.
        initial_accuracy: The baseline accuracy value to be used as a reference point.

    Returns:
        None. This function displays a plot and prints a summary to the console.
    """
    # Initialize the figure for plotting performance curves
    plt.figure(figsize=(8, 5))
    # Define the range of epochs based on the length of the training history
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Plot the training accuracy values over the defined epochs
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    # Plot the testing accuracy values over the defined epochs
    plt.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    # Draw a horizontal reference line representing the starting accuracy level
    plt.axhline(y=initial_accuracy, color='gray', linestyle='--', label='Initial Accuracy')
    
    # Configure the horizontal axis label as the training epoch
    plt.xlabel('Epoch')
    # Configure the vertical axis label as the accuracy percentage
    plt.ylabel('Accuracy (%)')
    # Set the main title for the performance visualization
    plt.title('Model Training Progress')
    # Add a legend to distinguish between the different data series
    plt.legend()
    # Enable a background grid with partial transparency for better readability
    plt.grid(True, alpha=0.3)
    # Adjust the layout to ensure all elements fit within the figure boundaries
    plt.tight_layout()
    # Render the visualization to the display
    plt.show()
    
    # Output a structured text summary of the performance metrics
    print("Accuracy Summary:")
    # Display the starting accuracy value provided to the function
    print(f"  Started at: {initial_accuracy:.2f}% (untrained)")
    # Display the final accuracy value recorded in the testing history
    print(f"  Final test accuracy: {history['test_acc'][-1]:.2f}%")
    # Calculate and display the difference between the final and initial accuracy
    print(f"  Total improvement: +{history['test_acc'][-1] - initial_accuracy:.2f}%")


# Plots loss and accuracy curves and marks the best epoch
def plot_training_history(history, model_name="Custom DenseNet"):
    """Visualizes the training and validation history of a model.

    This function generates and displays two plots: one for training and
    validation loss, and another for validation accuracy. It also highlights
    the epoch where the highest validation accuracy was achieved.

    Args:
        history (dict): A dictionary containing the model's training history.
                        It must include the keys 'val_accuracy', 'val_loss',
                        and 'train_loss'.
        model_name (str, optional): The name of the model, used for plot
                                    titles and labels. Defaults to "Custom DenseNet".
    """
    # Find the index of the epoch with the highest validation accuracy.
    best_epoch_idx = np.argmax(history['val_accuracy'])
    # Get the best validation accuracy and the corresponding validation loss.
    best_val_acc = history['val_accuracy'][best_epoch_idx]
    best_val_loss = history['val_loss'][best_epoch_idx]

    # Print a summary of the model's performance at the best epoch.
    print("---------- Best Epoch Performance ----------")
    print(f"Model: {model_name}")
    print(f"Epoch: {best_epoch_idx + 1}")
    print(f"Validation Accuracy: {best_val_acc:.2%}")
    print(f"Validation Loss:     {best_val_loss:.4f}")
    print("------------------------------------------\n")

    # Set up the figure and subplots for displaying the history.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # Define colors for plot elements to ensure consistency.
    train_color = 'blue'
    val_color = 'red'
    best_epoch_color = 'green'

    # Plot training and validation loss on the first subplot.
    ax1.plot(history['train_loss'], label=f'{model_name} Train Loss', color=train_color, linestyle='-')
    ax1.plot(history['val_loss'], label=f'{model_name} Val Loss', color=val_color, linestyle='--')

    # Highlight the validation loss at the best-accuracy epoch with a marker.
    ax1.plot(best_epoch_idx, best_val_loss, marker='o', color=best_epoch_color, markersize=8, label='Loss When Best Acc Was Achieved')
    # Annotate the marker with its precise value.
    ax1.annotate(f'{best_val_loss:.4f}',
                 xy=(best_epoch_idx, best_val_loss),
                 xytext=(best_epoch_idx, best_val_loss + 0.1),
                 ha='center', color=best_epoch_color,
                 arrowprops=dict(arrowstyle="->", color=best_epoch_color))

    # Set titles and labels for the loss subplot.
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy on the second subplot.
    ax2.plot(history['val_accuracy'], label=f'{model_name} Val Accuracy', color=val_color)

    # Highlight the best validation accuracy with a marker.
    ax2.plot(best_epoch_idx, best_val_acc, marker='o', color=best_epoch_color, markersize=8, label='Best Accuracy Achieved')
    # Annotate the marker with its value.
    ax2.annotate(f'{best_val_acc:.2%}',
                 xy=(best_epoch_idx, best_val_acc),
                 xytext=(best_epoch_idx, best_val_acc - 0.05),
                 ha='center', color=best_epoch_color,
                 arrowprops=dict(arrowstyle="->", color=best_epoch_color))

    # Set titles and labels for the accuracy subplot.
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Determine an appropriate interval for x-axis ticks for readability.
    num_epochs = len(history['train_loss'])
    if num_epochs > 10:
        x_ticks_interval = 2
    else:
        x_ticks_interval = 1

    # Generate tick locations (0-indexed) and corresponding labels (1-indexed).
    tick_locations = np.arange(0, num_epochs, x_ticks_interval)
    tick_labels = np.arange(1, num_epochs + 1, x_ticks_interval)

    # Apply the custom x-axis ticks to both subplots.
    ax1.set_xticks(ticks=tick_locations, labels=tick_labels)
    ax2.set_xticks(ticks=tick_locations, labels=tick_labels)

    # Adjust subplot parameters for a tight layout and display the plot.
    plt.tight_layout()
    plt.show()


# Plots and compares the training history of two models
def plot_training_logs(history1, history2, model_name1="PlainCNN Model", model_name2="SimpleResNet Model"):
    """
    Plots and compares the training history of two models.

    Args:
        history1 (dict): The training history dictionary for the first model.
        history2 (dict): The training history dictionary for the second model.
        model_name1 (str, optional): The name of the first model for labels.
                                     Defaults to "Plain CNN Model".
        model_name2 (str, optional): The name of the second model for labels.
                                     Defaults to "ResNet Model".
    """
    # Extract the final validation accuracy for each model from the history.
    final_acc1 = history1['val_accuracy'][-1]
    final_acc2 = history2['val_accuracy'][-1]

    # Display a summary of the final validation metrics.
    print("---------- Final Validation Accuracies ---------")
    print(f"{model_name1:<15}     |  Accuracy: {final_acc1:.2%}")
    print(f"{model_name2:<15}  |  Accuracy: {final_acc2:.2%}")
    print("------------------------------------------------\n")

    # Create a figure with two side-by-side subplots for loss and accuracy.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    color1, color2 = 'red', 'blue'

    # Plot the training and validation loss curves for both models.
    ax1.plot(history1['train_loss'], label=f'{model_name1} Train Loss', color=color1, linestyle='-')
    ax1.plot(history1['val_loss'], label=f'{model_name1} Val Loss', color=color1, linestyle='--')
    ax1.plot(history2['train_loss'], label=f'{model_name2} Train Loss', color=color2, linestyle='-')
    ax1.plot(history2['val_loss'], label=f'{model_name2} Val Loss', color=color2, linestyle='--')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot the validation accuracy curves for both models.
    ax2.plot(history1['val_accuracy'], label=f'{model_name1} Val Accuracy', color=color1)
    ax2.plot(history2['val_accuracy'], label=f'{model_name2} Val Accuracy', color=color2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Configure the x-axis ticks to be displayed at a dynamic interval.
    num_epochs = len(history1['train_loss'])
    if num_epochs > 10:
        x_ticks_interval = 2
    else:
        x_ticks_interval = 1
    
    # Define the locations for the ticks (0-indexed) and their labels (1-indexed).
    tick_locations = np.arange(0, num_epochs, x_ticks_interval)
    tick_labels = np.arange(1, num_epochs + 1, x_ticks_interval)

    # Apply the custom ticks and labels to both subplots.
    ax1.set_xticks(ticks=tick_locations, labels=tick_labels)
    ax2.set_xticks(ticks=tick_locations, labels=tick_labels)
    
    # Adjust layout to prevent titles from overlapping and render the plot.
    plt.tight_layout()
    plt.show()


# Plots the training and validation metrics from a model training process
def plot_training_metrics(metrics):
    """
    Plots the training and validation metrics from a model training process.

    This function generates two side-by-side plots:
    1. Training Loss vs. Validation Loss.
    2. Validation Accuracy.

    Args:
        metrics (list): A list or tuple containing three lists:
                        [train_losses, val_losses, val_accuracies].
    """
    # Unpack the metrics into their respective lists
    train_losses, val_losses, val_accuracies = metrics
    
    # Determine the number of epochs from the length of the training losses list
    num_epochs = len(train_losses)
    # Create a 1-indexed range of epoch numbers for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Configure the first subplot for training and validation loss ---
    # Select the first subplot
    ax1 = axes[0]
    # Plot training loss data
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    # Plot validation loss data
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    # Set the title and axis labels for the loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Display the legend
    ax1.legend()
    # Add a grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configure the second subplot for validation accuracy ---
    # Select the second subplot
    ax2 = axes[1]
    # Plot validation accuracy data
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Accuracy')
    # Set the title and axis labels for the accuracy plot
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    # Display the legend
    ax2.legend()
    # Add a grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # --- Apply dynamic and consistent styling to both subplots ---
    # Calculate a suitable interval for the x-axis ticks to avoid clutter
    x_interval = (num_epochs - 1) // 10 + 1

    # Loop through each subplot to apply common axis settings
    for ax in axes:
        # Set the y-axis to start at 0 and the x-axis to span the epochs
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)
        
        # Set the major tick locator for the x-axis using the dynamic interval
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        # Set the font size for the tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust subplot parameters for a tight layout
    plt.tight_layout()
    # Display the plots
    plt.show()


# Compares the training histories of two models side by side
def compare_models(history1, history2,
                   model1_name="Custom Model",
                   model2_name="PyTorch Model",
                   model1_desc="Custom implementation",
                   model2_desc="Built-in implementation",
                   figsize=(14, 5)):
    """
    Compares the training and testing performance of two models side by side.

    Args:
        history1: Dictionary containing training and testing accuracy for the first model.
        history2: Dictionary containing training and testing accuracy for the second model.
        model1_name: String label for the first model.
        model2_name: String label for the second model.
        model1_desc: Brief architectural description of the first model.
        model2_desc: Brief architectural description of the second model.
        figsize: Tuple defining the dimensions of the comparison plot.

    Returns:
        None. This function displays a comparative plot and prints a summary report.
    """
    # Calculate the number of completed epochs for both training histories
    epochs1 = len(history1['train_acc'])
    epochs2 = len(history2['train_acc'])
    # Identify the shortest duration to maintain a synchronized comparison window
    epochs = min(epochs1, epochs2)
    
    # Initialize a figure with two subplots for side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # Define the range of epochs for the x-axis
    epoch_range = range(1, epochs + 1)
    
    # Plot training and testing accuracy for the first model
    ax1.plot(epoch_range, history1['train_acc'][:epochs], 'b-', label='Train', linewidth=2)
    ax1.plot(epoch_range, history1['test_acc'][:epochs], 'r-', label='Test', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(model1_name)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([40, 100])
    # Apply integer ticks if the number of epochs is small
    if epochs <= 10:
        ax1.set_xticks(range(1, epochs + 1))
    
    # Determine the highest accuracy and corresponding epoch for the first model
    if 'best_epoch' in history1 and history1['best_epoch'] <= epochs:
        best_epoch1 = history1['best_epoch']
        best_acc1 = history1['best_test_acc']
    else:
        best_acc1 = max(history1['test_acc'][:epochs])
        best_epoch1 = history1['test_acc'][:epochs].index(best_acc1) + 1
    
    # Highlight the best performance point on the first plot
    ax1.plot(best_epoch1, best_acc1, 'r*', markersize=12)
    ax1.annotate(f'Best: {best_acc1:.1f}%',
                xy=(best_epoch1, best_acc1),
                xytext=(best_epoch1, best_acc1-5),
                fontsize=9, ha='center')
    
    # Draw a vertical indicator at the selected model's epoch
    ax1.axvline(x=best_epoch1, color='green', linestyle='--', alpha=0.3, label='Selected Model')
    
    # Plot training and testing accuracy for the second model
    ax2.plot(epoch_range, history2['train_acc'][:epochs], 'b-', label='Train', linewidth=2)
    ax2.plot(epoch_range, history2['test_acc'][:epochs], 'r-', label='Test', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(model2_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([40, 100])
    if epochs <= 10:
        ax2.set_xticks(range(1, epochs + 1))
    
    # Determine the highest accuracy and corresponding epoch for the second model
    if 'best_epoch' in history2 and history2['best_epoch'] <= epochs:
        best_epoch2 = history2['best_epoch']
        best_acc2 = history2['best_test_acc']
    else:
        best_acc2 = max(history2['test_acc'][:epochs])
        best_epoch2 = history2['test_acc'][:epochs].index(best_acc2) + 1
    
    # Highlight the best performance point on the second plot
    ax2.plot(best_epoch2, best_acc2, 'r*', markersize=12)
    ax2.annotate(f'Best: {best_acc2:.1f}%',
                xy=(best_epoch2, best_acc2),
                xytext=(best_epoch2, best_acc2-5),
                fontsize=9, ha='center')
    
    # Draw a vertical indicator for the second selected model
    ax2.axvline(x=best_epoch2, color='green', linestyle='--', alpha=0.3, label='Selected Model')
    
    # Finalize the layout and display the visual plots
    plt.tight_layout()
    plt.show()
    
    # Retrieve performance metrics at the optimal epoch for both models
    selected_test_acc1 = history1['test_acc'][best_epoch1-1]
    selected_train_acc1 = history1['train_acc'][best_epoch1-1]
    selected_test_acc2 = history2['test_acc'][best_epoch2-1]
    selected_train_acc2 = history2['train_acc'][best_epoch2-1]
    
    # Retrieve final epoch metrics for baseline comparison
    final_test_acc1 = history1['test_acc'][epochs-1]
    final_train_acc1 = history1['train_acc'][epochs-1]
    final_test_acc2 = history2['test_acc'][epochs-1]
    final_train_acc2 = history2['train_acc'][epochs-1]
    
    # Calculate the accuracy gap between the two selected models
    difference = selected_test_acc2 - selected_test_acc1
    
    # Calculate the epoch at which models reached 90% of their peak accuracy
    convergence_threshold = 0.9 * max(best_acc1, best_acc2)
    conv_epoch1 = next((i for i, acc in enumerate(history1['test_acc'][:epochs])
                        if acc >= convergence_threshold), epochs) + 1
    conv_epoch2 = next((i for i, acc in enumerate(history2['test_acc'][:epochs])
                        if acc >= convergence_threshold), epochs) + 1
    
    # Generate the text-based comparison summary report
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"Training Duration: {epochs} epochs")
    
    # Print metrics specifically for the first model
    print(f"\n{model1_name}:")
    print(f"  Architecture: {model1_desc}")
    print(f"  Selected Model from Epoch: {best_epoch1}")
    print(f"  Selected Model Test Accuracy: {selected_test_acc1:.2f}%")
    print(f"  Selected Model Train Accuracy: {selected_train_acc1:.2f}%")
    if best_epoch1 != epochs:
        print(f"  (Final epoch accuracy was: {final_test_acc1:.2f}%)")
    print(f"  Convergence: Epoch {conv_epoch1}")
    
    # Print metrics specifically for the second model
    print(f"\n{model2_name}:")
    print(f"  Architecture: {model2_desc}")
    print(f"  Selected Model from Epoch: {best_epoch2}")
    print(f"  Selected Model Test Accuracy: {selected_test_acc2:.2f}%")
    print(f"  Selected Model Train Accuracy: {selected_train_acc2:.2f}%")
    if best_epoch2 != epochs:
        print(f"  (Final epoch accuracy was: {final_test_acc2:.2f}%)")
    print(f"  Convergence: Epoch {conv_epoch2}")
    
    # Analyze and compare the relative performance of both models
    print(f"\n" + "-"*60)
    print("PERFORMANCE ANALYSIS")
    print("-"*60)
    print(f"Difference in selected model accuracy: {difference:+.2f}%")
    
    # Identify which model converged to the target threshold more quickly
    if conv_epoch1 < conv_epoch2:
        print(f"Faster convergence: {model1_name} (by {conv_epoch2-conv_epoch1} epochs)")
    elif conv_epoch2 < conv_epoch1:
        print(f"Faster convergence: {model2_name} (by {conv_epoch1-conv_epoch2} epochs)")
    else:
        print(f"Both models converged at the same speed")
    
    # Provide a qualitative interpretation of the quantitative differences
    print(f"\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)
    
    if abs(difference) < 2:
        print("✓ Both models perform similarly! The implementations are comparable.")
    elif difference > 0:
        print(f"✓ {model2_name} performs better by {difference:.2f}%")
        if difference > 5:
            print("  This is a significant improvement.")
    else:
        print(f"✓ {model1_name} performs better by {-difference:.2f}%")
        if difference < -5:
            print("  This is a significant improvement.")
    
    # Analyze the training-testing accuracy gap to identify potential overfitting
    overfit1 = selected_train_acc1 - selected_test_acc1
    overfit2 = selected_train_acc2 - selected_test_acc2
    
    print("\n⚠️  Overfitting Analysis (for selected models):")
    print(f"  {model1_name}: {overfit1:.1f}% gap (train-test) at epoch {best_epoch1}")
    print(f"  {model2_name}: {overfit2:.1f}% gap (train-test) at epoch {best_epoch2}")
    
    # Evaluate if the overfitting gap exceeds acceptable thresholds
    if overfit1 > 10 or overfit2 > 10:
        if overfit1 > overfit2:
            print(f"  → {model1_name} shows more overfitting")
        else:
            print(f"  → {model2_name} shows more overfitting")
    else:
        print("  → Both models show acceptable generalization (gap < 10%)")
    
    # Assess the effectiveness of selecting the best epoch over the final epoch
    if best_epoch1 < epochs or best_epoch2 < epochs:
        print("\n📊 Early Stopping Analysis:")
        if best_epoch1 < epochs:
            prevented_overfit1 = (final_train_acc1 - final_test_acc1) - overfit1
            print(f"  {model1_name}: Early stopping at epoch {best_epoch1} prevented {prevented_overfit1:.1f}% additional overfitting")
        if best_epoch2 < epochs:
            prevented_overfit2 = (final_train_acc2 - final_test_acc2) - overfit2
            print(f"  {model2_name}: Early stopping at epoch {best_epoch2} prevented {prevented_overfit2:.1f}% additional overfitting")
    
    print("="*60)


# Generates a performance plot with annotations from a dictionary of results
def plot_performance_summary(results_dict, title, xlabel, ylabel="Stable Time per Epoch (ms)"):
    """
    Generates a performance plot with annotations from a dictionary of results.

    Args:
        results_dict: A dictionary where keys are parameters and values are performance metrics.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
    """
    # Create a new dictionary to store only valid results
    valid_times = {k: v for k, v in results_dict.items() if v != float("inf")}

    # Check if there are any valid results to plot
    if not valid_times:
        # Print a message if no valid data is found
        print("No valid results to plot.")
        # Exit the function if there is no data
        return

    # Convert the time values from seconds to milliseconds
    valid_times_ms = {k: v * 1000 for k, v in valid_times.items()}

    # Create a new figure with a specified size
    plt.figure(figsize=(10, 5))

    # Plot the data points with a line and markers
    (line,) = plt.plot(
        list(valid_times_ms.keys()),
        list(valid_times_ms.values()),
        marker="o",
        linestyle="--",
        color="#237B94",
    )

    # Set the title and labels for the plot axes
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Set the x-axis ticks to match the keys in the dictionary
    plt.xticks(list(valid_times_ms.keys()))
    # Add a grid to the plot for better readability
    plt.grid(True)

    # Get the y-values to determine plot limits
    y_values = list(valid_times_ms.values())
    # Set the x-axis limits
    plt.xlim(left=0)
    # Set the y-axis limits with a 10% padding at the top
    plt.ylim(bottom=0, top=max(y_values) + max(y_values) * 0.1)

    # Get the color of the plotted line for annotations
    marker_color = line.get_color()
    # Calculate a vertical offset for the annotations
    vertical_offset = max(y_values) * 0.06

    # Iterate over the data points to add annotations
    for x, y in valid_times_ms.items():
        # Add a text annotation at each data point
        plt.text(
            x,
            y + vertical_offset,
            f"{y:.2f}",
            ha="center",
            color=marker_color,
        )

    # Display the plot
    plt.show()


# Visualizes a word similarity matrix as a heatmap
def plot_similarity_matrix(matrix, labels):
    """
    Visualizes a word similarity matrix as a heatmap.

    This function takes a square matrix of similarity scores and a list of
    labels to generate a color-coded heatmap, making it easy to see the
    relationships between different words.

    Args:
        matrix: A square 2D numpy array containing the similarity scores.
        labels: A list of string labels for the matrix axes.
    """
    # Create a new figure with a specific size for the plot.
    plt.figure(figsize=(10, 8))
    # Display the data as an image, where each cell's color corresponds to its value.
    # The 'bwr' colormap (blue-white-red) is great for showing similarity.
    plt.imshow(matrix, cmap='bwr')
    # Add a color bar to serve as a legend for the values.
    plt.colorbar(label='Cosine Similarity')
    # Set the chart's title and axis labels.
    plt.title('Word Similarity Matrix')
    plt.xlabel('Words')
    plt.ylabel('Words')
    # Set the tick marks on both axes to correspond to the words.
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    # Adjust the plot to ensure everything fits without overlapping.
    plt.tight_layout()
    # Display the final plot.
    plt.show()


# Visualizes 2D word embeddings using a scatter plot
def plot_embeddings(coords, labels, label_dict, title):
    """
    Visualizes 2D word embeddings using a scatter plot.

    This function plots a set of 2D coordinates, annotates each point with
    a corresponding label, and colors the points based on predefined
    categories. It dynamically assigns colors to categories for clear
    visual distinction.

    Args:
        coords: A 2D numpy array where each row represents the x, y
                coordinates of a point.
        labels: A list of string labels, one for each point in `coords`.
        label_dict: A dictionary that maps category names to lists of
                    words belonging to that category.
        title: The title for the plot.
    """
    # Use a try-except block to handle potential inconsistencies
    # between the labels and the dictionary.
    try:
        # Create a reverse mapping from words to categories for efficient lookup.
        word_to_category = {word: category for category, words in label_dict.items() for word in words}

        # Validate that every label has a corresponding category in the map.
        # This will raise a KeyError if a word is not found.
        for word in labels:
            _ = word_to_category[word]

        # --- Dynamic Color Assignment ---
        # Define a base list of visually distinct colors for consistency.
        fixed_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Use a set for efficient tracking of assigned colors to prevent duplicates.
        used_colors = set(fixed_colors)
        
        # Get a list of all unique category names.
        unique_categories = list(label_dict.keys())
        # Initialize a dictionary to map each category to a unique color.
        category_to_color = {}

        # Iterate through each unique category to assign a color.
        for i, category in enumerate(unique_categories):
            # Assign colors from the predefined list first.
            if i < len(fixed_colors):
                category_to_color[category] = fixed_colors[i]
            else:
                # Generate a unique random color for any additional categories.
                random_color = None
                while random_color is None or random_color in used_colors:
                    # Generate a random hex color string (e.g., '#a1c3f7').
                    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    random_color = f'#{r:02x}{g:02x}{b:02x}'
                
                # Assign the new unique color to the category.
                category_to_color[category] = random_color
                # Add the new color to the set of used colors.
                used_colors.add(random_color)

        # --- Plotting and Annotation ---
        # Set up the figure size for the plot.
        plt.figure(figsize=(8, 8))

        # Plot points for each category to associate them with the correct legend entry.
        for category in unique_categories:
            # Find the indices of words belonging to the current category.
            indices = [i for i, word in enumerate(labels) if word_to_category[word] == category]
            # Get the coordinates for the current category using the indices.
            category_coords = coords[indices]
            
            # Create a scatter plot for the current category's points.
            plt.scatter(
                category_coords[:, 0],
                category_coords[:, 1],
                color=category_to_color[category],
                s=120,
                alpha=0.9,
                label=category
            )

        # Add text annotations for each individual data point.
        for i, word in enumerate(labels):
            plt.annotate(word, (coords[i, 0], coords[i, 1]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=12, fontweight='bold')

        # --- Final Plot Adjustments ---
        # Set the plot title and axis labels.
        plt.title(title, fontsize=16)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        # Add a grid for better readability.
        plt.grid(True, alpha=0.3)
        # Display the legend to show category-color mappings.
        plt.legend(loc='upper right', title="Categories")
        # Adjust plot to ensure everything fits without overlapping.
        plt.tight_layout()
        # Display the final visualization.
        plt.show()

    except KeyError as e:
        # Handle cases where a label is not found in the categorization dictionary.
        print(f"Error: Word {e} from the label list was not found in the dictionary.")
        print("Please ensure that the label list and the categorization dictionary are consistent.")


# Plots a confusion matrix as a seaborn heatmap
def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using Seaborn's heatmap.

    Args:
        cm: A confusion matrix numpy array.
        class_names: A list of class names for the labels.
    """
    # Create a new figure with a specified size
    plt.figure(figsize=(8, 6))
    # Generate a heatmap from the confusion matrix data
    sns.heatmap(
        cm, 
        # Annotate cells with the numeric value
        annot=True,     
        # Use a general format to disable scientific notation
        fmt='g',        
        # Set the color map to 'Blues'
        cmap='Blues',   
        # Set the labels for the x-axis ticks
        xticklabels=class_names, 
        # Set the labels for the y-axis ticks
        yticklabels=class_names
    )
    # Set the label for the x-axis
    plt.xlabel('Predicted Labels')
    # Set the label for the y-axis
    plt.ylabel('True Labels')
    # Set the title of the plot
    plt.title('Confusion Matrix')
    # Display the plot
    plt.show()


# Prints per-class accuracy and then plots the confusion matrix
def plot_confusion_matrix_with_class_accuracy(cm_np, labels):
    """Calculates and displays per-class accuracy, then plots a confusion matrix.

    This function first computes the accuracy for each individual class from the
    provided confusion matrix. It displays these scores with a progress bar, then
    uses scikit-learn's ConfusionMatrixDisplay to visualize the full matrix.

    Args:
        cm_np (numpy.ndarray): The confusion matrix to be plotted, where rows
                               represent true labels and columns represent
                               predicted labels.
        labels (list of str): A list of class names that correspond to the
                              matrix indices.
    """
    # --- Per-Class Accuracy Calculation ---
    correct_predictions = cm_np.diagonal()
    total_samples_per_class = cm_np.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.nan_to_num(correct_predictions / total_samples_per_class)
    class_accuracies = {label: acc for label, acc in zip(labels, per_class_acc)}

    # --- Display Per-Class Accuracy with a Progress Bar ---
    print("--- Per-Class Accuracy ---")
    for class_name, acc in tqdm(class_accuracies.items(), desc="Calculating Metrics"):
        print(f"{class_name:<20} | Accuracy: {acc:.2%}")
        time.sleep(0.05)
    print("-" * 40 + "\n")

    # --- Confusion Matrix Plotting ---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=labels)
    
    # Create a figure and axes object with the desired size
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Render the confusion matrix plot on the created axes
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    # Rotate the x-axis tick labels for better readability with long names.
    plt.xticks(rotation=45, ha="right")
    
    # Set the plot's title and axis labels.
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Display the finalized plot.
    plt.show()


# Analyzes model performance, visualizes F1 scores, and selects the best model
def plot_and_select_best_model(all_trained_data, bar_color=color_map["blue"]):
    """
    Analyzes model performance, visualizes F1 scores, and selects the best model.

    This function takes a dictionary of trained models and their performance
    metrics, generates a bar chart comparing their validation F1 scores,
    and identifies and returns the model with the highest score.

    Args:
        all_trained_data: A dictionary where keys are model names and values
                          are tuples containing the model instance and a results
                          dictionary.
        bar_color: The hex color code for the bars in the plot.

    Returns:
        The instance of the top-performing model based on F1 score.
    """
    # Extract model names and their corresponding validation F1 scores into a dictionary.
    model_performance = {
        name: results["val_f1"] for name, (model, results) in all_trained_data.items()
    }
    # Create a pandas Series from the performance data and sort it in descending order.
    performance_series = pd.Series(model_performance).sort_values(ascending=False)

    # Convert the F1 scores to a percentage format for easier interpretation in the plot.
    performance_series_pct = performance_series * 100

    # Initialize a new figure for the plot with a specified size.
    plt.figure(figsize=(12, 8))
    # Generate a bar plot of the model performance.
    ax = performance_series_pct.plot(
        kind="bar", color=bar_color, edgecolor="black", width=0.7
    )

    # Add a horizontal grid to the plot for better readability.
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # Set the limits of the y-axis to range from 0 to 100.
    ax.set_ylim(bottom=0, top=100)

    # Iterate over each bar in the plot to add a text annotation.
    for bar in ax.patches:
        # Place the F1 score percentage in the middle of each bar.
        ax.annotate(
            f"{bar.get_height():.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
            ha="center",
            va="center",
            size=12,
            fontweight="bold",
            color="white",
        )

    # Set the title and axis labels for the plot.
    ax.set_title("Model F1 Score Comparison", fontsize=16, pad=20)
    ax.set_ylabel("Validation F1 Score (%)", fontsize=12)
    # Customize the appearance of the x-axis tick labels.
    ax.tick_params(axis="x", rotation=0, labelsize=12)

    # Adjust plot parameters to ensure everything fits without overlapping.
    plt.tight_layout()
    # Save the generated plot as a PNG image file.
    plt.savefig("f1_score_comparison.png")

    # Identify the name of the model with the highest F1 score.
    best_model_name = performance_series.index[0]
    # Retrieve the model object corresponding to the best-performing model.
    best_model_instance = all_trained_data[best_model_name][0]

    # Print the name of the best performing model to the console.
    print(f"Best Performing Model (by F1 Score): {best_model_name}\n")
    # Return the instance of the best model.
    return best_model_instance


# Handles for the single reused figure, passed back in on every update
@dataclass
class LivePlot:
    """Handles for the single reused figure, passed back in on every update."""

    fig: Figure
    ax: Axes
    line: Line2D
    x: np.ndarray       # flattened inputs
    order: np.ndarray   # indices that sort x left to right


# Which epochs to capture as animation frames
def frame_schedule(total_epochs: int, n_frames: int = 80) -> set[int]:
    """
    Which epochs to capture as animation frames.

    Spacing them evenly wastes most of the animation: the fit is ~80% settled
    within the first few hundred epochs, so evenly spaced frames show a snap
    followed by a long stretch of nothing moving. Log spacing packs frames in
    early, where the line is actually changing, and thins them out later.

    Returns fewer than n_frames, since early values collide after rounding.
    """
    spaced = np.logspace(0, np.log10(total_epochs), n_frames)
    return set(np.unique(spaced.astype(int)).tolist())


# Returns the model's current predictions as a flat NumPy array
def _predict(model: torch.nn.Module, x_norm: torch.Tensor) -> np.ndarray:
    """Returns the model's current predictions as a flat NumPy array."""
    # We only want the numbers to draw, so skip building the gradient graph
    with torch.no_grad():
        predicted = model(x_norm)
    return predicted.numpy().flatten()


# Call inside the training loop
def update_live_plot(plot: LivePlot, epoch: int, loss: torch.Tensor,
                     model: torch.nn.Module, x_norm: torch.Tensor) -> np.ndarray:
    """
    Call inside the training loop. Moves the green line to the current fit.

    Returns the predictions so the caller can record them for the GIF.
    """
    y_pred = _predict(model, x_norm)

    # Move the existing line rather than drawing a new one, sorted left to right
    plot.line.set_data(plot.x[plot.order], y_pred[plot.order])

    plot.ax.set_title(f'Epoch: {epoch + 1} | Loss: {loss.item():.4f}')

    # Push the change to the screen, then rest so the eye can follow it
    plot.fig.canvas.draw_idle()
    plot.fig.canvas.flush_events()

    # Guard the zero case: plt.pause(0) waits on the event loop forever
    if FRAME_PAUSE > 0:
        plt.pause(FRAME_PAUSE)

    return y_pred


# Plots (x, y) data points as a scatter with custom title and axis labels
def plot_data(x, y, title='Data', xlabel='x', ylabel='y', label='Actual data'):
    """
    Plots (x, y) data points as a scatter.

    Args:
        x: Values for the horizontal axis (tensor, list or array).
        y: Values for the vertical axis.
        title: Plot title.
        xlabel: Horizontal axis label.
        ylabel: Vertical axis label.
        label: Legend label of the points.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(_to_numpy(x), _to_numpy(y), color='orange', marker='o', linestyle='none', label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


# Plots de-normalized model predictions against the original data
def plot_final_fit(model, x, y, x_norm, y_std, y_mean, title='Model fit vs. actual data',
                   xlabel='x', ylabel='y', data_label='Actual data', fit_label='Model predictions'):
    """
    Plots model predictions against the original data, after undoing the target normalization.

    Args:
        model: Trained regression model that takes normalized inputs.
        x: Original inputs, for the horizontal axis.
        y: Original targets.
        x_norm: Normalized inputs fed to the model.
        y_std: Standard deviation used to normalize the targets.
        y_mean: Mean used to normalize the targets.
        title, xlabel, ylabel: Plot texts.
        data_label, fit_label: Legend labels of the data and the model line.
    """
    model.eval()
    with torch.no_grad():
        predicted_norm = model(x_norm)
    predicted = predicted_norm * y_std + y_mean
    plt.figure(figsize=(8, 6))
    plt.plot(_to_numpy(x), _to_numpy(y), color='orange', marker='o', linestyle='none', label=data_label)
    plt.plot(_to_numpy(x), _to_numpy(predicted), color='green', label=fit_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


# Plots the current model fit on normalized data during training
def plot_training_progress(epoch, loss, model, x_norm, y_norm, xlabel='Normalized x', ylabel='Normalized y'):
    """
    Plots the current model fit on normalized data during training.

    Args:
        epoch: Zero-based epoch number, shown as epoch + 1 in the title.
        loss: Current loss value (kept for call compatibility, not drawn).
        model: Model being trained.
        x_norm: Normalized inputs.
        y_norm: Normalized targets.
        xlabel, ylabel: Axis labels.
    """
    predicted_norm = model(x_norm)
    x_plot = _to_numpy(x_norm)
    y_plot = _to_numpy(y_norm)
    y_pred_plot = predicted_norm.detach().numpy()
    sorted_indices = x_plot.argsort(axis=0).flatten()
    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_plot, color='orange', marker='o', linestyle='none', label='Actual Normalized Data')
    plt.plot(x_plot[sorted_indices], y_pred_plot[sorted_indices], color='green', label='Model Predictions')
    plt.title(f'Epoch: {epoch + 1} | Normalized Training Progress')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
    time.sleep(0.05)


# Creates the figure shared by the live training plot and the saved GIF
def _build_figure(x: np.ndarray, y: np.ndarray, xlabel: str = 'Normalized x', ylabel: str = 'Normalized y') -> LivePlot:
    """Creates the figure shared by both the live view and the saved GIF."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, color='orange', marker='o', linestyle='none', label='Actual Normalized Data')
    (line,) = ax.plot([], [], color='green', label='Model Predictions')
    ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    return LivePlot(fig=fig, ax=ax, line=line, x=x, order=x.argsort())


# Opens an interactive figure that update_live_plot animates during training
def setup_live_plot(x_norm: torch.Tensor, y_norm: torch.Tensor, xlabel: str = 'Normalized x',
                    ylabel: str = 'Normalized y') -> LivePlot:
    """Call once BEFORE the training loop. Opens the window that will animate."""
    plt.ion()
    plot = _build_figure(x_norm.numpy().flatten(), y_norm.numpy().flatten(), xlabel, ylabel)
    plot.fig.show()
    return plot


# Replays recorded training frames into an animated GIF
def save_training_gif(frames: list, x_norm: torch.Tensor, y_norm: torch.Tensor, path: str = 'training_progress.gif',
                      xlabel: str = 'Normalized x', ylabel: str = 'Normalized y') -> None:
    """Call once AFTER training. Replays the recorded (epoch, loss, predictions) frames into a GIF."""
    plot = _build_figure(x_norm.numpy().flatten(), y_norm.numpy().flatten(), xlabel, ylabel)

    def draw_frame(i: int):
        """Draws frame i. FuncAnimation calls this once per frame."""
        epoch, loss_value, y_pred = frames[i]
        plot.line.set_data(plot.x[plot.order], y_pred[plot.order])
        plot.ax.set_title(f'Epoch: {epoch} | Loss: {loss_value:.4f}')
        return (plot.line,)
    animation = FuncAnimation(plot.fig, draw_frame, frames=len(frames), blit=False)
    animation.save(path, writer='pillow', fps=GIF_FPS)
    plt.close(plot.fig)
    print(f'Saved animation to {path}')


# Draws a styled scatter plot with labelled groups, hollow markers and optional color scale
def plot_grouped_scatter(x, y, groups, title='Grouped scatter', xlabel='x', ylabel='y', legend_title=None,
                         color_values=None, cmap='YlOrRd', colorbar_label=None, background='#35466A', figsize=(12, 7)):
    """
    Draws a styled scatter plot where points are split into labelled groups.

    Args:
        x: Values for the horizontal axis.
        y: Values for the vertical axis.
        groups: List of dicts, one per group, with keys
            'mask' (boolean array over the points), 'label' (legend text),
            'color' (optional, used when color_values is None) and 'filled' (optional, default True).
            Hollow markers (filled=False) are handy for a second category such as weekend vs weekday.
        title, xlabel, ylabel: Plot texts.
        legend_title: Optional legend title.
        color_values: Optional per-point values that color the points through cmap, with a colorbar.
        cmap: Matplotlib colormap name used with color_values.
        colorbar_label: Label of the colorbar.
        background: Axes background color.
        figsize: Figure size.
    """
    x, y = _to_numpy(x), _to_numpy(y)
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.set_facecolor(background)
    text_color, grid_color, neutral = 'black', '#FFFFFF', '#CCCCCC'
    if color_values is not None:
        color_values = _to_numpy(color_values)
        colormap = plt.get_cmap(cmap)
        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
    handles, bar_source = [], None
    for g in groups:
        mask = _to_numpy(g['mask']).astype(bool)
        filled = g.get('filled', True)
        color = g.get('color', neutral)
        if color_values is not None:
            vals = color_values[mask]
            if filled:
                sc = ax.scatter(x[mask], y[mask], s=80, c=vals, cmap=colormap, norm=norm, alpha=0.9)
                bar_source = sc if bar_source is None else bar_source
            else:
                ax.scatter(x[mask], y[mask], s=80, facecolors='none', edgecolors=colormap(norm(vals)),
                           linewidths=1.5, alpha=0.9)
            marker_color = neutral
        else:
            if filled:
                ax.scatter(x[mask], y[mask], s=90, c=color, alpha=0.9, edgecolor=color, linewidth=0.5)
            else:
                ax.scatter(x[mask], y[mask], s=90, facecolors='none', edgecolors=color, linewidths=1.5, alpha=0.9)
            marker_color = color
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=marker_color if filled else 'none',
                              markeredgecolor=marker_color, markeredgewidth=0.5 if filled else 1.5,
                              markersize=9, linestyle='None'))
    title_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'bold', 'size': 16}
    label_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'normal', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    ax.set_xlabel(xlabel, fontdict=label_font)
    ax.set_ylabel(ylabel, fontdict=label_font)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=grid_color, alpha=0.5)
    legend = ax.legend(handles=handles, labels=[g['label'] for g in groups], title=legend_title)
    legend.get_frame().set_facecolor('#4A6C8C')
    legend.get_frame().set_edgecolor('none')
    if legend_title:
        plt.setp(legend.get_title(), color='white', fontsize=14, weight='bold')
    for text, g in zip(legend.get_texts(), groups):
        text.set_fontsize(13)
        text.set_color('white' if color_values is not None else g.get('color', 'white'))
    if bar_source is not None:
        cbar = plt.colorbar(bar_source, ax=ax)
        cbar.set_label(colorbar_label or '', fontdict=label_font)
        cbar.ax.tick_params(colors=text_color)
    plt.show()


# Scatter plot of predictions against true values with a perfect-prediction line
def plot_predicted_vs_actual(predicted, actual, title='Actual vs. predicted', xlabel='Actual', ylabel='Predicted'):
    """
    Scatter plot of predictions against true values with a perfect-prediction diagonal.

    Args:
        predicted: Model outputs (tensor, list or array).
        actual: True targets.
        title, xlabel, ylabel: Plot texts.
    """
    background, text_color, grid_color = '#35466A', 'black', '#FFFFFF'
    points_color, line_color = '#F0F0E0', '#E64C29'
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')
    predicted, actual = _to_numpy(predicted), _to_numpy(actual)
    plt.figure(figsize=(12, 7))
    ax = plt.axes()
    ax.set_facecolor(background)
    ax.scatter(actual, predicted, s=90, c=points_color, alpha=0.7, edgecolor=points_color, linewidth=0.5,
               label='Model Predictions')
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, linestyle='-', color=line_color, alpha=0.75, zorder=0, label='Perfect Prediction')
    title_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'bold', 'size': 16}
    label_font = {'family': 'sans-serif', 'color': text_color, 'weight': 'normal', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    ax.set_xlabel(xlabel, fontdict=label_font)
    ax.set_ylabel(ylabel, fontdict=label_font)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=grid_color, alpha=0.5)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=points_color, markersize=10, linestyle='None'),
               Line2D([0], [0], color=line_color, linewidth=2)]
    legend = ax.legend(handles=handles, labels=['Model Predictions', 'Perfect Prediction'], title='Prediction Insights')
    legend.get_frame().set_facecolor('#4A6C8C')
    legend.get_frame().set_edgecolor('none')
    plt.setp(legend.get_title(), color='white', fontsize=15, weight='bold', ha='left')
    for text, color in zip(legend.get_texts(), (points_color, line_color)):
        text.set_color(color)
        text.set_fontsize(14)
    plt.show()



# =============================================================================
# IMAGE AND TENSOR UTILITIES
# =============================================================================


# Callable class that undoes a Normalize transform given the original mean and std
class Denormalize:
    def __init__(self, mean, std):

        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]

        self.denormalize = transforms.Normalize(mean=new_mean, std=new_std)

    def __call__(self, tensor):
        return self.denormalize(tensor)


# Reverses mean and std normalization on an image tensor
def unnormalize(tensor_img, mean=[0.5] * 3, std=[0.5] * 3):
    """Reverses the normalization on a tensor image."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor_img * std + mean


# Process a raw PyTorch tensor into a viewable PIL Image object
@torch.no_grad()
def tensor_to_image(tensor):
    """
    Process a raw PyTorch tensor into a viewable PIL Image object.
    
    Arguments:
        tensor: The input tensor representing an image.

    Returns:
        A PIL Image object scaled to 0-255.
    """
    # Move tensor to CPU, remove batch dimension, and reorder dimensions to HxWxC
    img = tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    
    # Normalize values to the range [0, 1] if the input range is [-1, 1]
    img = (img + 1) / 2 if img.min() < 0 else img
    
    # Ensure all pixel values fall strictly within the valid range
    img = np.clip(img, 0, 1)
    
    # Scale values to 0-255, cast to unsigned 8-bit integers, and convert to PIL format
    return Image.fromarray((img * 255).astype(np.uint8))


# Plots an image tensor or array with an optional title and extra info text
def plot_img(img, label=None, info=None, ax=None):
    """
    Plots an image with optional labels and supplementary information.

    Args:
        img (torch.Tensor or numpy.ndarray): The image data to plot.
        label (str): Optional label to display as the title.
        info (str): Optional supplementary text to display below the image.
        ax (matplotlib.axes.Axes): Optional matplotlib axes to plot on.
    """
    def add_info_text(ax, info):
        """
        Adds supplementary text below the plot on a given axes.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            info (str): The text to be added.
        """
        # Add text to the axes at a specified position.
        ax.text(
            0.5, -0.1, info, transform=ax.transAxes, ha="center", va="top", fontsize=10
        )
        # Set the x-axis label position to the top.
        ax.xaxis.set_label_position("top")

    # Create a new figure and axes if none are provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Check if a label is provided to determine how to display the image.
    if label:
        # Create a title string with the provided label.
        title = f"Label: {label}"
        # Display the image with the generated title.
        show_titled_image((img, title), ax=ax)
    else:
        # Display the image without a title.
        show_image(img, ax=ax)

    # Check if supplementary information is provided.
    if info:
        # Add the information as text below the image.
        add_info_text(ax, info)

    # If no axes were passed in, display the newly created plot.
    if ax is None:
        plt.show()


# Plots a tensor image using matplotlib
def plot_image(image_tensor, title=None, **kwargs):
    """
    Plots a tensor image using matplotlib.

    Args:
        image_tensor (torch.Tensor): A tensor representing an image with shape (C, H, W).
        title (str, optional): The title of the plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to `plt.imshow`.
    """
    image_np = image_tensor.squeeze(0).numpy()
    img_transposed = np.transpose(image_np, (1, 2, 0))
    
    # Use the 'nearest' interpolation to enhance pixelation
    plt.imshow(img_transposed, interpolation='nearest', **kwargs)
    plt.title(title)
    plt.axis('off')


# Prints basic debugging information about an image tensor
def quick_debug(img):
    """
    Prints basic debugging information about an image tensor.

    Args:
        img (torch.Tensor): The image tensor to inspect.
    """
    # Print the shape of the image tensor.
    print(f"Shape: {img.shape}")  # Should be [3, 224, 224]
    # Print the data type of the tensor.
    print(f"Type: {img.dtype}")  # Should be torch.float32
    # Print the minimum and maximum pixel values in the tensor.
    print(
        f"Range of pixel values: [{img.min():.1f}, {img.max():.1f}]"
    )


# Creates a structured RGB tensor representing a simple circle shape
def get_ball():
    """
    Creates a structured RGB tensor representing a simple circle shape.

    Returns:
        np.ndarray: A numpy array of shape (3, 32, 32) representing an RGB image containing a circle.
    """
    height, width = 32, 32
    
    # Create a grid of coordinates
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    radius = min(center_y, center_x) // 2
    
    # Create a circular mask
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Initialize channels
    red_channel = np.zeros((height, width))
    green_channel = np.zeros((height, width))
    blue_channel = np.zeros((height, width))
    
    # Apply mask to create a circle in red channel
    red_channel[mask] = 1.0
    
    # Stack channels to create an RGB image
    structured_image = np.stack([red_channel, green_channel, blue_channel], axis=0)
    return structured_image


# Loads and converts an image file to RGB format
def get_query_img(image_path):
    """
    Loads and converts an image file to RGB format.

    Args:
        image_path: The file path to the image.

    Returns:
        The loaded and converted image object.
    """
    # Open the image file from the specified path and convert it to the RGB color space
    img = Image.open(image_path).convert("RGB")
    # Return the processed image
    return img


# Retrieves an original image and its label from a dataset by index
def get_image(dataset, index):
    """
    Retrieves an original image and its label from a dataset by index.

    Args:
        dataset: The dataset object (e.g., a torchvision ImageFolder)
                 that contains a 'samples' list of (path, label) tuples.
        index: The index of the item to retrieve.

    Returns:
        A tuple containing the PIL Image object (in RGB format) and
        its corresponding label.
    """
    # Get the file path and label for the given index from the dataset's samples
    path, label = dataset.samples[index]  
    # Open the image file using its path and convert it to RGB format
    original_img = Image.open(path).convert("RGB")
    # Return the loaded image and its label
    return original_img, label


# Loads and transforms an image file into a normalized tensor suitable for model inference
def preprocess_image(image_path, device="cpu"):
    """
    Loads and transforms an image file into a normalized tensor suitable for model inference.

    Arguments:
        image_path: The file path to the input image.
        device: The computing device (CPU or GPU) to place the tensor on.
    """
    # Define the sequence of image transformations including resizing, cropping, and normalization
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    # Load the image, apply the transformations, add a batch dimension, and transfer to the specified device
    image_tensor = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    return image_tensor


# Displays an image with its corresponding label and title
def display_image(image, label, title, num_ticks=6, show_values=True):
    """
    Displays an image with its corresponding label and title.

    This function handles different image formats (PIL Image and PyTorch Tensor),
    normalizes the display range, and optionally overlays pixel values on the image.

    Args:
        image: The image data to be displayed. Can be a PIL Image or a PyTorch Tensor.
        label: The label associated with the image.
        title: The title for the plot.
        num_ticks (int, optional): The number of ticks to display on the color bar. Defaults to 6.
        show_values (bool, optional): If True, overlays the numerical value of each pixel on the image. Defaults to True.
    """
    # Initialize variables for value range and image data.
    vmin_val, vmax_val = None, None
    image_data = None

    # Check if the input is a PIL Image.
    if isinstance(image, Image.Image):
        # Set the value range for a standard 8-bit image.
        vmin_val = 0
        vmax_val = 255
        # Convert the PIL Image to a NumPy array.
        image_data = np.array(image)
    # Check if the input is a PyTorch Tensor.
    elif isinstance(image, torch.Tensor):
        # Convert the tensor to a NumPy array and remove any single-dimensional entries.
        image_np = image.numpy().squeeze()
        # Determine the min and max values from the tensor for normalization.
        vmin_val = image_np.min()
        vmax_val = image_np.max()
        # Assign the NumPy array to image_data.
        image_data = image_np
    # Handle unsupported image types.
    else:
        print("Warning: Unsupported image type.")
        return

    # Create a new figure for the plot.
    plt.figure(figsize=(9, 9))
    # Display the image data as a grayscale image.
    plt.imshow(image_data, cmap='gray', vmin=vmin_val, vmax=vmax_val)
    # Set the title of the plot with the provided title and label.
    plt.title(f'{title} | Label: {label}')

    # Check if pixel values should be displayed on the image.
    if show_values:
        # Calculate a threshold to determine the color of the text (black or white).
        threshold = (vmin_val + vmax_val) / 2.0
        # Get the dimensions of the image.
        height, width = image_data.shape
        
        # Iterate over each pixel to display its value.
        for y in range(height):
            for x in range(width):
                # Get the pixel value.
                value = image_data[y, x]
                # Set text color based on the pixel's brightness.
                text_color = "white" if value < threshold else "black"
                
                # Format the text to display, handling integers and floats differently.
                text_to_display = f"{value:.0f}" if isinstance(value, np.integer) else f"{value:.1f}"
                
                # Add the pixel value as text to the plot.
                plt.text(x, y, text_to_display, 
                         ha="center", va="center", color=text_color, fontsize=6)

    # Add a grid to the plot.
    plt.grid(True, color='red', alpha=0.3, zorder=2)
    # Set the x-axis ticks.
    plt.xticks(np.arange(0, 28, 4))
    # Set the y-axis ticks.
    plt.yticks(np.arange(0, 28, 4))
    
    # Add a color bar to the plot.
    cbar = plt.colorbar()
    # Create evenly spaced ticks for the color bar.
    ticks = np.linspace(vmin_val, vmax_val, num=num_ticks)
    # Set the ticks on the color bar.
    cbar.set_ticks(ticks)
    # Format the tick labels on the color bar.
    cbar.ax.set_yticklabels([f'{t:.2f}' for t in ticks])

    # Show the final plot.
    plt.show()


# Displays one or more images in a row with optional titles
def display_images(images, titles=None, figsize=None):
    """
    Displays one or more images in a row with optional titles.
    
    Args:
        images: A single image or a list of images.
        titles: A list of titles corresponding to the images (optional).
        figsize: A tuple (width, height) for the figure. 
                 If None, defaults to (5 * num_images, 5).
    """
    # 1. Normalize input: Ensure 'images' is always a list
    if not isinstance(images, (list, tuple)):
        images = [images]
    
    num_images = len(images)
    
    # 2. Smart Figure Size: Default to 5 inches wide per image if not specified
    if figsize is None:
        figsize = (5 * num_images, 5)

    # 3. Create Subplots
    fig, axs = plt.subplots(1, num_images, figsize=figsize)
    
    # 4. Handle Single Axis: Make 'axs' iterable even if there is only 1 image
    if num_images == 1:
        axs = [axs]
        
    # 5. Plot Loop
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')
        
        # Add title if provided and index exists
        if titles and i < len(titles):
            axs[i].set_title(titles[i])

    print() # To add space
    plt.tight_layout()
    plt.show()


# Displays a grid of images
def display_grid(images, titles=None, n_rows=None, n_cols=None, figsize=None, main_title=None, row_labels=None):
    """
    Displays a grid of images. Handles both flat lists (wrapping) and lists of lists (defined rows).

    Args:
        images: Either a flat list of images (requires n_rows/n_cols) 
                OR a list of lists of images (e.g. [row1_images, row2_images]).
        titles: Optional list of strings for subplot titles. 
                Should match the total number of images (flattened order).
        n_rows: Number of rows (required if images is a flat list).
        n_cols: Number of columns (required if images is a flat list).
        figsize: Tuple (width, height). Defaults to auto-calculated based on grid size.
        main_title: Super title for the entire figure.
        row_labels: Optional list of strings for row labels (left side).
    """
    # 1. Detect Input Type: Is it a flat list or a list of rows?
    if isinstance(images[0], list): 
        # Input is a list of lists (Snippet 2 style)
        n_rows = len(images)
        n_cols = len(images[0])
        flat_images = [img for row in images for img in row]
    else: 
        # Input is a flat list (Snippet 1 style)
        flat_images = images
        if n_rows is None or n_cols is None:
            raise ValueError("For flat lists, n_rows and n_cols must be provided.")

    # 2. Setup Figure
    if figsize is None:
        # Heuristic: 3 inches per column, 3.5 inches per row
        figsize = (3 * n_cols, 3.5 * n_rows)
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if main_title:
        fig.suptitle(main_title, fontsize=16)

    # 3. Normalize Axes: Ensure axes is always a 2D array [row, col]
    # This handles edge cases like 1xN or Nx1 grids automatically
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    # 4. Plotting Loop
    for i, img in enumerate(flat_images):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        
        ax.imshow(img)
        ax.axis('off')
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)

    # 5. Add Row Labels (if provided)
    if row_labels:
        for row_idx, label in enumerate(row_labels):
            if row_idx < n_rows:
                # Add label to the first column of the row
                axes[row_idx, 0].set_ylabel(label, rotation=90, size='large', labelpad=20)
                # Note: For labels to show when axis is 'off', we might need specific matplotlib backend 
                # behavior or manual text. If using standard styles, keeping the axis off might hide this.
                # However, this preserves the logic from your original Snippet 2.

    print() # To add space
    plt.tight_layout()
    plt.show()


# Creates a single grid image from a batch of images
def make_image_grid(images, grid_size=None):
    """
    Creates a single grid image from a batch of images.

    Args:
        images (np.ndarray or torch.Tensor): Batch of images with shape (N, H, W).
        grid_size (int, optional): The number of images per row/column. Defaults to ceil(sqrt(N)).

    Returns:
        np.ndarray: A single large image composed of the input images in a grid.
    """
    # images: (N, H, W)
    num_images, H, W = images.shape
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(num_images)))
    grid = np.zeros((grid_size * H, grid_size * W))
    for idx in range(num_images):
        row = idx // grid_size
        col = idx % grid_size
        grid[row*H:(row+1)*H, col*W:(col+1)*W] = images[idx]
    return grid


# Creates a tiled grid image from a subset of channels in a feature map
def grid_from_layer(feat, num_filters=4):
    """
    Creates a tiled grid image from a subset of channels in a feature map.

    This function selects a specified number of filters from the input feature
    tensor at equal intervals and arranges them into a square grid for visualization.
    The resulting grid is normalized to the [0, 1] range.

    Arguments:
        feat: The input feature map tensor (typically from a CNN layer).
        num_filters: The number of channels to select and display in the grid.
    """
    # Remove single-dimensional entries, move to CPU, and convert to a NumPy array
    feat = feat.squeeze().cpu().numpy()
    
    # Unpack the channel, height, and width dimensions
    C, H, W = feat.shape
    
    # Select indices for the filters to visualize, spaced equally across the channels
    idxs = np.linspace(0, C - 1, num_filters, dtype=int)

    # Calculate the side length of the square grid needed to hold the filters
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    
    # Initialize an empty array for the final grid image
    grid = np.zeros((grid_size * H, grid_size * W))

    # Iterate through the selected filter indices to populate the grid
    for i, c in enumerate(idxs):
        # Calculate row and column positions for the current filter
        r, col = divmod(i, grid_size)
        
        # Insert the feature map into the calculated position within the grid
        grid[
            r * H : (r + 1) * H, col * W : (col + 1) * W
        ] = feat[c]

    # Shift values so the minimum is 0
    grid -= grid.min()
    
    # Scale values so the maximum is 1, adding a small epsilon to prevent division by zero
    grid /= grid.max() + 1e-5
    
    return grid


# Displays a horizontal strip of upsampled feature maps
def visual_strip(upsampled):
    """
    Displays a horizontal strip of upsampled feature maps.

    This function concatenates a list of image tensors into a single grid 
    row and renders it using Matplotlib to visualize the progression 
    of features across model layers.

    Arguments:
        upsampled: A list of 4D tensors (Batch, Channel, Height, Width) 
                   representing processed feature maps.
    """
    # Create a single image grid from the list of tensors, arranged in one row with padding
    grid = make_grid(torch.cat(upsampled), nrow=5, padding=2)
    
    # Initialize the figure with a wide aspect ratio suitable for a horizontal strip
    plt.figure(figsize=(14, 3))
    
    # Transpose the tensor dimensions from (C, H, W) to (H, W, C) and move to CPU for Matplotlib compatibility
    plt.imshow(grid.cpu().permute(1, 2, 0))
    
    # Hide the axis ticks and labels
    plt.axis("off")
    
    # Set the descriptive title for the plot
    plt.title("Feature-map progression conv1 → layer4")
    
    # Render the final figure
    plt.show()


# Shows a single-channel image with pixel ticks and an optional label title
def visualize_image(img, label=None, ax=None, label_to_title=None, show_grid=True):
    """
    Shows a single-channel image with pixel-level ticks and an optional label title.

    Args:
        img: Image tensor or array. Extra channel dimensions are squeezed away.
        label: Optional label used for the title.
        ax: Optional matplotlib axis. If None, a new figure with a colorbar is shown.
        label_to_title: Optional function that turns the label into a title string.
            Defaults to 'Label: <label>'.
        show_grid: Draws pixel ticks and a light grid when True.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy().squeeze()
    elif isinstance(img, np.ndarray) and img.ndim == 3:
        img = img[:, :, 0]
    title = None
    if label is not None:
        title = label_to_title(label) if label_to_title else f'Label: {label}'
    show_colorbar = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(img, cmap='gray')
    if show_grid:
        ax.set_xticks(np.arange(0, img.shape[1], 1))
        ax.set_yticks(np.arange(0, img.shape[0], 1))
        ax.tick_params(labelsize=6)
        ax.grid(True, color='red', alpha=0.3)
    if title:
        ax.set_title(title)
    if show_colorbar:
        plt.colorbar(im, ax=ax)
        plt.show()


# Converts a tensor, list or array into a NumPy array
def _to_numpy(values):
    """Converts a tensor, list or array into a NumPy array."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


# Turns a normalized image tensor back into a displayable array
def deprocess_image(tensor, mean, std, to_uint8=False):
    """
    Turns a normalized (C, H, W) tensor back into a displayable (H, W, C) array.

    Args:
        tensor: Normalized image tensor.
        mean: Per-channel mean used for normalization.
        std: Per-channel std used for normalization.
        to_uint8: Returns values in 0..255 as uint8 when True, otherwise floats in 0..1.
    """
    image = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(np.array(std) * image + np.array(mean), 0, 1)
    return (image * 255).astype(np.uint8) if to_uint8 else image



# =============================================================================
# DATASETS AND DATALOADERS
# =============================================================================


# A custom dataset wrapper that applies specific transformations to a dataset subset
class TransformedDataset(Dataset):
    """
    A custom dataset wrapper that applies specific transformations to a dataset subset.

    Args:
        subset (Dataset): A subset of a larger dataset or a standalone dataset object.
        transform (callable): A function or transform object that takes in an image 
                              and returns a transformed version.
    """
    
    def __init__(self, subset, transform):
        """
        Initializes the dataset with a specific subset and transformation pipeline.

        Args:
            subset (Dataset): The underlying data subset to be used.
            transform (callable): The transformation logic to apply to the data.
        """
        # Assign the provided data subset to the instance
        self.subset = subset
        # Assign the transformation function to the instance
        self.transform = transform

    def __len__(self):
        """
        Calculates the total number of samples in the dataset.

        Returns:
            length (int): The number of items available in the subset.
        """
        # Return the length of the underlying subset
        return len(self.subset)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset and applies the defined transformation.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            transformed_img (Tensor/PIL Image): The data sample after applying transformations.
            label (any): The ground truth label associated with the data sample.
        """
        # Extract the raw image and label from the subset at the given index
        img, label = self.subset[idx]
        
        # Apply the transformation to the image and return it along with the label
        return self.transform(img), label


# Wraps a train and a validation dataset into DataLoaders
def create_dataloaders(train_dataset, validation_dataset, batch_size):
    """
    Creates and returns DataLoader instances for training and validation.

    Args:
        train_dataset: The dataset to be used for training.
        validation_dataset: The dataset to be used for validation.
        batch_size: The number of samples per batch to load.

    Returns:
        A tuple containing the training DataLoader and the validation DataLoader.
    """
    # Create a DataLoader for the training dataset, with shuffling enabled
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Create a DataLoader for the validation dataset, without shuffling
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    # Return the created DataLoaders
    return train_loader, val_loader


# Splits input and target sequences into train and validation DataLoaders
def create_sequence_dataloaders(inputs, targets, batch_size=32, train_split=0.9, shuffle=True):
    """
    Constructs training and validation DataLoaders from input and target sequences.

    Args:
        inputs (list): List of numerical input sequences for the dataset.
        targets (list): List of numerical target sequences for the dataset.
        batch_size (int): The number of samples per batch to load.
        train_split (float): The proportion of the dataset to include in the training split.
        shuffle (bool): Whether to reorganize the data at every epoch during training.

    Returns:
        train_loader (DataLoader): The DataLoader instance for the training set.
        val_loader (DataLoader): The DataLoader instance for the validation set, or None if no split.
        dataset (Dataset): The full initialized dataset object.
    """
    # Create the primary dataset object from the provided sequences
    dataset = ShakespeareDataset(inputs, targets)
    
    # Check if the data should be partitioned into training and validation sets
    if train_split < 1.0:
        # Calculate the number of samples designated for training
        train_size = int(train_split * len(dataset))
        # Calculate the remaining samples designated for validation
        val_size = len(dataset) - train_size
        # Perform a random split of the dataset into two subsets
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Initialize the training DataLoader with shuffling and batching enabled
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=0
        )
        
        # Initialize the validation DataLoader without shuffling
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        # Log the sizes and batch counts for the partitioned datasets
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Number of train batches: {len(train_loader)}")
        print(f"Number of val batches: {len(val_loader)}")
        
        # Return both loaders and the underlying dataset
        return train_loader, val_loader, dataset
    
    else:
        # Initialize a single DataLoader for the entire dataset when no split is required
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=0
        )
        
        # Log the total dataset size and the resulting number of batches
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of batches: {len(train_loader)}")
        
        # Return the training loader, None for validation, and the full dataset
        return train_loader, None, dataset


# Splits a full dataset into training and validation sets
def create_dataset_splits(full_dataset, train_split_percentage=0.8):
    """
    Splits a full dataset into training and validation sets.

    Args:
        full_dataset: The complete PyTorch Dataset to be split.
        train_split_percentage: The percentage of the dataset to allocate
                                for the training set. Defaults to 0.8.

    Returns:
        A tuple containing the training dataset and validation dataset.
    """
    # Calculate the number of samples for the training set.
    train_size = int(train_split_percentage * len(full_dataset))
    # Calculate the number of samples for the validation set.
    val_size = len(full_dataset) - train_size

    # Randomly split the full dataset into training and validation subsets.
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Return the two dataset splits.
    return train_dataset, val_dataset


# Loads an ImageFolder and makes a stratified train and validation split with their transforms
def create_datasets(dataset_path, train_transform, val_transform, train_split=0.8, seed=42):
    """
    Initializes and splits an image dataset into training and validation sets 
    using a stratified approach to maintain class balance.

    Args:
        dataset_path: Path to the directory containing image folders.
        train_transform: Transformations to apply to the training set.
        val_transform: Transformations to apply to the validation set.
        train_split: Proportion of the dataset to include in the train split.
        seed: Random seed for reproducibility.
    """
    
    class TransformedDataset(Dataset):
        """
        A wrapper class to apply specific transformations to a dataset subset.

        Args:
            subset: The dataset subset to wrap.
            transform: The transformation pipeline to apply to the data.
        """
        def __init__(self, subset, transform):
            # Initialize the subset and the transformation pipeline
            self.subset = subset
            self.transform = transform
            # Inherit class metadata from the original dataset
            self.classes = subset.dataset.classes
            self.class_to_idx = subset.dataset.class_to_idx

        def __len__(self):
            """
            Returns the total number of samples in the subset.
            """
            return len(self.subset)

        def __getitem__(self, idx):
            """
            Retrieves a sample and its label, applying the transformation.

            Args:
                idx: The index of the item to retrieve.
            """
            img, label = self.subset[idx]
            return self.transform(img), label

    # Load the base dataset from the specified directory
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=None)
    
    # Map indices to their respective class labels for stratified splitting
    class_indices = defaultdict(list)
    for idx, target in enumerate(full_dataset.targets):
        class_indices[target].append(idx)

    train_indices = []
    val_indices = []

    # Configure the random seed for shuffling consistency
    random.seed(seed)
    
    # Distribute samples from each class into train and validation lists
    for label, indices in class_indices.items():
        # Shuffle indices within the current class
        random.shuffle(indices)
        # Determine the cutoff point based on the split ratio
        split_point = int(train_split * len(indices))
        # Assign indices to the respective splits
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    # Create subset objects based on the calculated indices
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # Wrap the subsets with their specific transformation logic
    train_dataset = TransformedDataset(subset=train_subset, transform=train_transform)
    val_dataset = TransformedDataset(subset=val_subset, transform=val_transform)

    return train_dataset, val_dataset


# Loads training and validation datasets from specified subdirectories
def load_datasets(dataset_path, train_transform=None, val_transform=None):
    """
    Loads training and validation datasets from specified subdirectories.

    This function uses torchvision's ImageFolder to load datasets from
    'sub_train_dataset' and 'sub_validation_dataset' folders within
    the provided root path.

    Args:
        dataset_path: The root directory path containing the dataset subfolders.
        train_transform: An optional torchvision transform to be applied
                         to the training dataset.
        val_transform: An optional torchvision transform to be applied
                       to the validation dataset.

    Returns:
        A tuple containing the training dataset and the validation dataset.
    """
    # Load the training dataset from the 'sub_train_dataset' subdirectory
    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "sub_train_dataset"), transform=train_transform
    )
    # Load the val dataset from the 'sub_validation_dataset' subdirectory
    validation_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "sub_validation_dataset"), transform=val_transform
    )
    
    # Return the loaded datasets
    return train_dataset, validation_dataset


# Splits an ImageFolder dataset 80/20 and returns train and validation loaders
def get_dataloaders(dataset_path, transformations=[transform_train, transform_val], batch_size=32):
    """
    Creates and returns training and validation DataLoaders from an ImageFolder dataset.

    Args:
        dataset_path (str): The file path to the root directory of the image dataset.
        transformations (list): A list containing two transformation objects, where 
                                index 0 is for training and index 1 is for validation.
        batch_size (int): The number of samples per batch to load.

    Returns:
        train_loader (DataLoader): The DataLoader instance for the training dataset.
        val_loader (DataLoader): The DataLoader instance for the validation dataset.
    """
    # Load the comprehensive image dataset from the provided directory path
    full_dataset = ImageFolder(root=dataset_path)

    # Define the proportion of the dataset to be allocated for training
    train_ratio = 0.8
    # Compute the integer count of samples intended for the training set
    train_size = int(train_ratio * len(full_dataset))
    # Compute the remaining count of samples for the validation set
    val_size = len(full_dataset) - train_size
    # Partition the full dataset into training and validation subsets randomly
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Print the resulting size of the training subset to the console
    print(f"Train samples (80%):\t\t{len(train_subset)}")
    # Print the resulting size of the validation subset to the console
    print(f"Validation samples (20%):\t{len(val_subset)}\n")

    # Wrap the training subset with the corresponding training transformations
    train_dataset = TransformedDataset(train_subset, transformations[0])
    # Wrap the validation subset with the corresponding validation transformations
    val_dataset = TransformedDataset(val_subset, transformations[1])

    # Initialize the training DataLoader with shuffling enabled and specified batch size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # Initialize the validation DataLoader with shuffling disabled for consistency
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Provide a confirmation message regarding the successful creation of the loaders
    print(f"DataLoaders created with {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # Return the training and validation data loaders as individual objects
    return train_loader, val_loader


# Counts and prints images per class in a folder of class subfolders
def dataset_images_per_class(dataset_path):
    """
    Counts and prints the number of images for each class in a dataset directory.

    Args:
        dataset_path (str): The file path to the directory containing class subfolders.

    """
    # Define the image file extensions considered valid for counting
    valid_exts = ('.jpeg', '.jpg', '.JPG', '.Jpg')
    
    # Initialize a dictionary to store the count of images for each class
    class_counts = defaultdict(int)
    
    # Iterate through the contents of the provided dataset path in alphabetical order
    for class_name in sorted(os.listdir(dataset_path)):
        # Construct the absolute path for the potential class directory
        class_dir = os.path.join(dataset_path, class_name)
        
        # Verify if the current path points to an actual directory
        if os.path.isdir(class_dir):
            # Generate a list of files that match the allowed image extensions
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(valid_exts)]
            # Assign the total count of discovered images to the class name
            class_counts[class_name] = len(image_files)
    
    # Output the total number of identified classes to the console
    print(f"Total number of classes: {len(class_counts)}\n")
    # Output the header for the individual class breakdown
    print("Number of images per class:\n")
    # Loop through the dictionary to display each class and its respective image count
    for cls, count in class_counts.items():
        print(f"{cls:25}: {count}")


# Prints image counts per class and in total for the train and val folders
def display_dataset_count(dataset_path):
    """
    Calculates and displays the number of images for each class and the total
    number of images in the train and validation sets.

    Args:
        dataset_path: The root path to the dataset directory.
    """
    # Define a maximum width for class names and the total label to ensure alignment
    class_width = 25
    total_width = 25

    # Iterate over the main directories (train, val)
    for split in ['train', 'val']:
        # Construct the full path for the current split
        split_path = os.path.join(dataset_path, split)
        
        # Check if the directory exists
        if not os.path.isdir(split_path):
            # Print a message if the directory is not found and continue to the next split
            print(f"\nDirectory not found: {split_path}")
            continue
            
        # Print a header for the current dataset split
        print(f"--- {split.capitalize()} Set ---")
        
        # Initialize a counter for the total number of images in the current split
        total_images = 0
        
        # Get class names from the subdirectories within the split path and sort them
        class_names = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        
        # Check if any class directories were found
        if not class_names:
            # Print a message if no class directories are found and continue to the next split
            print("  No class directories found.")
            continue

        # Count images in each class directory and add to total
        for class_name in class_names:
            # Construct the full path for the current class directory
            class_path = os.path.join(split_path, class_name)
            # Count the number of items (images) in the class directory
            num_images = len(os.listdir(class_path))
            # Add the number of images in the current class to the total count
            total_images += num_images
            # Use f-string with left-alignment for class name and right-alignment for image count
            print(f"  - {class_name:<{class_width}}: {num_images} images")
            
        # Print a separator line for readability
        print("—" * (class_width + total_width))
        # Print the total count for the current set with right-alignment
        print(f"  Total: {total_images:>{total_width}} images\n")


# Prints the sizes and shapes of the first batch of a data loader
def display_data_loader_contents(data_loader):
    """
    Displays the contents of the data loader including sizes and shapes.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader to 
            display.
    """
    # Begin exception handling block
    try:
        # Print the total number of images in the dataset
        print("Total number of images in dataset:", len(data_loader.dataset))
        
        # Print the total number of batches in the data loader
        print("Total number of batches:", len(data_loader))
        
        # Iterate through the batches in the data loader
        for batch_idx, (data, labels) in enumerate(data_loader):
            # Print the current batch number
            print(f"--- Batch {batch_idx + 1} ---")
            
            # Print the shape of the data tensor
            print(f"Data shape: {data.shape}")
            
            # Print the shape of the labels tensor
            print(f"Labels shape: {labels.shape}")
            
            # Exit the loop after displaying the first batch
            break
            
    # Catch iteration exceptions if the data loader is empty
    except StopIteration:
        # Print a message indicating the data loader is empty
        print("data loader is empty.")
        
    # Catch any other general exceptions
    except Exception as e:
        # Print the specific error message
        print(f"An error occurred: {e}")


# Analyzes and displays a statistical summary of an image dataset
def display_dataset_stats(base_data_dir):
    """
    Analyzes and displays a statistical summary of an image dataset.

    This function iterates through the subdirectories of a specified base
    directory, where each subdirectory is considered a distinct class. It
    counts the number of image files ('.jpg') within each class and
    renders the statistics in a styled HTML table, including a total count.

    Args:
        base_data_dir (str): The file path to the root directory of the dataset.
    """
    class_counts = {}
    
    # Iterate through the base directory to find class folders and count images.
    try:
        for class_name in os.listdir(base_data_dir):
            class_path = os.path.join(base_data_dir, class_name)
            if os.path.isdir(class_path):
                # Count only files ending with .jpg (case-insensitive)
                image_count = len([
                    f for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith('.jpg')
                ])
                class_counts[class_name] = image_count
    except FileNotFoundError:
        print(f"Error: The directory '{base_data_dir}' was not found.")
        return

    # Verify that some classes were found before proceeding.
    if not class_counts:
        print(f"No class folders with .jpg images found in '{base_data_dir}'.")
        return

    # Restructure the collected data into a list of dictionaries for the DataFrame.
    data_list = []
    for class_name, count in class_counts.items():
        data_list.append({
            'Class Name': class_name,
            'Number of Images': count
        })

    # Create a DataFrame from the list, sort it by class name, and calculate the total.
    df = pd.DataFrame(data_list).sort_values(by='Class Name').reset_index(drop=True)
    total_images = df['Number of Images'].sum()
    
    # Create a 'Total' summary row and append it to the DataFrame.
    total_row = pd.DataFrame([{
        'Class Name': '<b>Total</b>',
        'Number of Images': total_images
    }])
    df_display = pd.concat([df, total_row], ignore_index=True)

    # Apply CSS styling to the DataFrame for a professional presentation.
    styler = df_display.style.hide(axis="index")
    styler.set_table_styles(
        [
            {"selector": "table", "props": [("width", "60%"), ("margin", "0")]},
            {"selector": "td", "props": [("text-align", "left"), ("padding", "8px")]},
            {"selector": "th", "props": [
                ("text-align", "left"),
                ("padding", "8px"),
                ("background-color", "#4f4f4f"),
                ("color", "white")
            ]}
        ]
    )
    styler.set_properties(**{"white-space": "normal"})
    
    # Render the styled DataFrame.
    display(styler)


# Creates and configures a progress bar for a given dataloader
def get_dataloader_bar(dataloader, color="green"):
    """
    Creates and configures a progress bar for a given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader providing the dataset.
        color (str): The color of the progress bar.

    Returns:
        pbar (tqdm.tqdm): The configured progress bar object.
    """
    # Get the total number of samples from the dataloader's dataset.
    num_samples = len(dataloader.dataset)

    # Initialize a tqdm progress bar with specified settings.
    pbar = tqdm(
        # Set the total number of iterations for the bar.
        total=num_samples,
        # Dynamically calculate the width of the progress bar.
        ncols=int(num_samples / 10) + 300,
        # Define the format string for the progress bar's appearance.
        bar_format="{desc} {bar} {postfix}",
        # Direct the progress bar output to the standard output stream.
        file=sys.stdout,
        # Set the color of the progress bar.
        colour=color,
    )

    # Return the configured progress bar object.
    return pbar


# Updates the existing progress bar with the current batch information
def update_dataloader_bar(p_bar, batch, current_bs, n_samples):
    """
    Updates the existing progress bar with the current batch information.

    Args:
        p_bar (tqdm.tqdm): The progress bar object to update.
        batch (int): The current batch index.
        current_bs (int): The size of the current batch.
        n_samples (int): The total number of samples in the dataset.
    """
    # Advance the progress bar by the number of items in the current batch.
    p_bar.update(current_bs)
    # Set the description to show the current batch number.
    p_bar.set_description(f"Batch {batch+1}")

    # Check if the current batch is the last one.
    if (batch + 1) * current_bs > n_samples:
        # Update the postfix to show the total number of samples processed.
        p_bar.set_postfix_str(f"{n_samples} of a total of  {n_samples} samples")
    else:
        # Update the postfix to show the cumulative number of samples processed.
        p_bar.set_postfix_str(
            f"{current_bs*(batch+1)} of a total of  {n_samples} samples"
        )


# Analyzes and visualizes the efficiency of multiple data loaders
def visualize_dataloader_efficiency(loader_dict, device):
    """
    Analyzes and visualizes the efficiency of multiple data loaders.

    Args:
        loader_dict: A dictionary of data loaders to be analyzed.
        device: The device for data transfer.
    """
    # Define the number of batches to analyze
    num_batches = 52
    # Define the number of initial batches to be used for warm-up
    warmup_batches = 2

    # A list to store labels for the plot
    labels = []
    # A list to store average work times
    avg_work_times = []
    # A list to store average wait times
    avg_wait_times = []

    # Iterate through the dictionary of data loaders
    for label, loader in loader_dict.items():
        # Lists to store individual wait and work times
        wait_times, work_times = [], []
        # Initialize the end time of the previous work period
        end_of_previous_work = time.time()

        # Begin a try-except block to handle potential errors
        try:
            # Create an iterator for the data loader
            data_iter = iter(loader)
            # Loop for the specified number of batches
            for batch_idx in range(num_batches):
                # Get the next batch of data
                data = next(data_iter)
                # Record the start time of the current work period
                start_of_current_work = time.time()
                # Calculate and append the wait time
                wait_times.append(start_of_current_work - end_of_previous_work)

                # Unpack the data and labels from the batch
                inputs, labels_tensor = data
                # Move the inputs to the specified device
                inputs.to(device)
                # Move the labels to the specified device
                labels_tensor.to(device)

                # Record the end time of the current work period
                end_of_current_work = time.time()
                # Calculate and append the work time
                work_times.append(end_of_current_work - start_of_current_work)
                # Update the end time for the next iteration
                end_of_previous_work = end_of_current_work

            # Check if there are enough work times after warm-up
            if len(work_times) > warmup_batches:
                # Calculate the average work time, excluding warm-up batches
                avg_work = np.mean(work_times[warmup_batches:])
                # Calculate the average wait time, excluding warm-up batches
                avg_wait = np.mean(wait_times[warmup_batches:])

                # Append the label to the list
                labels.append(label)
                # Append the average work time
                avg_work_times.append(avg_work)
                # Append the average wait time
                avg_wait_times.append(avg_wait)
            # Handle cases where there are not enough batches
            else:
                # Print a message indicating the reason for skipping
                print(f"--> Skipping '{label}' due to insufficient batches.")

        # Catch the StopIteration exception if the dataset is exhausted
        except StopIteration:
            # Print a warning message
            print(
                f"Warning: Dataset exhausted for '{label}' before completing {num_batches} batches."
            )
        # Catch a generic RuntimeError
        except RuntimeError as e:
            # Print an error message
            print(f"\n❌ ERROR with {label}: {e}")

        # Ensure resources are cleaned up regardless of errors
        finally:
            # Delete the loader object
            del loader
            # Force garbage collection
            gc.collect()
            # If a CUDA device is available, clear its cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Check if any valid results were collected
    if not labels:
        # Print a message if there are no results to plot
        print("\nNo valid loader results to plot.")
        # Exit the function
        return

    # --- Plot Generation ---
    # Lists to store work and wait times as percentages
    work_percentages = []
    wait_percentages = []
    # Calculate the raw total time for each loader
    total_times_raw = [w + a for w, a in zip(avg_work_times, avg_wait_times)]

    # Loop through the average work and wait times
    for work, wait in zip(avg_work_times, avg_wait_times):
        # Calculate the total time for the current loader
        total_time = work + wait
        # Check if the total time is positive
        if total_time > 0:
            # Calculate and append the work percentage
            work_percentages.append((work / total_time) * 100)
            # Calculate and append the wait percentage
            wait_percentages.append((wait / total_time) * 100)
        else:
            # Append zero if the total time is not positive
            work_percentages.append(0)
            wait_percentages.append(0)

    # Create a figure with a specified size
    fig = plt.figure(figsize=(10, 8))
    # Create a grid for the subplots
    gs = gridspec.GridSpec(1, 2, width_ratios=[8, 2])

    # Add a subplot to the figure grid
    ax = fig.add_subplot(gs[0, 0])
    # Create an array of indices for the x-axis
    indices = np.arange(len(labels))

    # Plot the work percentages as a bar chart
    p1 = ax.bar(indices, work_percentages, color="#1C74EB", label="GPU Active Time (%)")
    # Plot the wait percentages as a stacked bar chart
    p2 = ax.bar(
        indices,
        wait_percentages,
        bottom=work_percentages,
        color="#FAB901",
        label="GPU Idle / Waiting Time (%)",
    )

    # Set the y-axis limits
    ax.set_ylim(bottom=0, top=100)

    # Determine a dynamic font size for the annotations
    dynamic_fontsize = max(8, 20 - len(labels))
    # Loop through the bars to add text annotations
    for i in range(len(labels)):
        # Check if the work percentage is greater than zero
        if work_percentages[i] > 0:
            # Add a text annotation with the work percentage
            ax.text(
                i,
                work_percentages[i] + 0.5,
                f"{work_percentages[i]:.1f}%",
                ha="center",
                va="bottom",
                color=p1[i].get_facecolor(),
                fontsize=dynamic_fontsize,
                weight="bold",
            )

    # Set the label for the y-axis
    ax.set_ylabel("Percentage of Average Time per Batch (%)")
    # Set a title for the entire figure
    fig.suptitle(
        "DataLoader Performance Comparison (Efficiency)", fontsize=16, weight="bold"
    )
    # Set the x-axis ticks
    ax.set_xticks(indices)
    # Set the x-axis tick labels and rotate them
    ax.set_xticklabels(labels, rotation=45, ha="right")
    # Add a grid to the plot
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust the plot layout
    plt.tight_layout()
    # Display the plot
    plt.show()


# Computes class weights inversely proportional to class frequencies
def compute_class_weights(dataset):
    """
    Computes class weights inversely proportional to class frequencies.

    Args:
        dataset (torch.utils.data.Dataset): A PyTorch dataset, expected to have
                                             a 'targets' attribute (like torchvision ImageFolder)
                                             or provide labels when iterated.

    Returns:
        torch.Tensor: A tensor containing the weight for each class.
    """
    # Try to get labels directly if the dataset supports it (like ImageFolder)
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'): # Handle datasets with a 'labels' attribute
         labels = dataset.labels
    else:
        # Fallback: Iterate through the dataset to get labels (slower)
        print("Dataset has no 'targets' or 'labels' attribute, iterating to get labels...")
        labels = [label for _, label in dataset]

    # Count the frequency of each class label
    class_counts = Counter(labels)

    # Sort counts by class index (0, 1, 2, ...) to ensure correct weight order
    sorted_counts = [class_counts[i] for i in sorted(class_counts)]

    # Calculate weights: total_samples / (num_classes * count_for_this_class)
    # This is a common formula for inverse frequency weighting
    total_samples = len(labels)
    num_classes = len(sorted_counts)

    # Calculate weight for each class
    weights = []
    for count in sorted_counts:
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    # Convert weights list to a PyTorch tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    return weights_tensor


# Loads the CIFAR10 dataset, applies transformations, and initializes data loaders
def load_cifar10():
    """
    Loads the CIFAR10 dataset, applies transformations, and initializes data loaders.

    Args:

    Returns:
        trainloader (torch.utils.data.DataLoader): The data loader for the training dataset 
                                                   containing augmented images.
        testloader (torch.utils.data.DataLoader): The data loader for the testing dataset 
                                                  containing normalized images.
    """
    # Define a series of transformations to apply to the training images.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Define a simpler set of transformations for the test images.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Specify the local directory to store the dataset.
    data_path = './CIFAR10_data'
    
    # Check if the data directory already exists to avoid re-downloading.
    if os.path.exists(data_path) and os.path.isdir(data_path):
        # If the folder exists, set download to False.
        download = False  
        print("CIFAR10 Data folder found locally. Loading from local.\n")
    else:
        # If the folder doesn't exist, set download to True.
        download = True   
        print("CIFAR10 Data folder not found locally. Downloading data.\n")

    # Set the number of images to be processed in one batch.
    batch_size = 128

    # Load the CIFAR10 training dataset and apply the defined training transformations.
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=download, transform=transform_train)
    
    # Create a data loader for the training set, which will shuffle the data and serve it in batches.
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Load the CIFAR10 test dataset and apply the defined test transformations.
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=download, transform=transform_test)
    
    # Create a data loader for the test set. Shuffling is not necessary for the test set.
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Return the training and testing data loaders.
    return trainloader, testloader


# Downloads the CIFAR-10 training dataset and applies transformations
def download_and_load_cifar10():
    """
    Downloads the CIFAR-10 training dataset and applies transformations.

    Returns:
        The transformed CIFAR-10 training dataset.
    """
    # Define a series of transformations to apply to the images
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),  # The mean for each channel
                (0.5, 0.5, 0.5),  # The standard deviation for each channel
            ),
        ]
    )

    # Specify the local directory for the dataset
    data_path = "./cifar10_data"

    # Check if the dataset directory already exists
    if os.path.exists(data_path) and os.path.isdir(data_path):
        # Set the download flag to False if the directory exists
        download = False
    else:
        # Set the download flag to True if the directory does not exist
        download = True

    # Load the CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,  # Specify the training subset
        download=download,  # Control whether to download the dataset
        transform=transform,  # Apply the defined transformations
    )

    # Return the loaded dataset
    return trainset


# Loads CIFAR-100 and keeps only the chosen classes
def load_cifar100_subset(target_classes, train_transform, val_transform, root='./cifar_100'):
    """
    Loads and filters the CIFAR-100 dataset to include only specified target classes.

    This function first checks for a local copy of the CIFAR-100 dataset and
    downloads it if not found. It then filters both the training and test sets
    to retain only the images and labels corresponding to the classes specified
    in `target_classes`. The labels are remapped to be contiguous from 0.

    Args:
        target_classes: A list of class name strings to be included in the dataset subset.
        train_transform: A torchvision transform to be applied to the training dataset images.
        val_transform: A torchvision transform to be applied to the test/val dataset images.
        root: The root directory where the dataset is stored or will be downloaded.

    Returns:
        A tuple containing the filtered training dataset and the filtered test dataset.
        Returns (None, None) if a specified target class is not found.
    """
    # Construct the path to the CIFAR-100 dataset directory.
    cifar100_path = os.path.join(root, 'cifar-100-python')
    # Check if the dataset directory exists locally.
    dataset_found = os.path.isdir(cifar100_path)
    if dataset_found:
        print(f"Dataset found in '{root}'. Loading from local files.")
    # If not found, inform the user that it will be downloaded.
    else:
        print(f"Dataset not found in '{root}'. Downloading...")

    # Load the full CIFAR-100 training dataset.
    train_dataset_full = torchvision.datasets.CIFAR100(
        root=root,
        train=True,
        download=not dataset_found,
        transform=train_transform
    )

    # Load the full CIFAR-100 test dataset.
    test_dataset_full = torchvision.datasets.CIFAR100(
        root=root,
        train=False,
        download=not dataset_found,
        transform=val_transform
    )
    print("Dataset loaded successfully.")

    # Get the list of all class names from the dataset.
    all_classes = train_dataset_full.classes
    try:
        # Get the original integer indices for the target class names.
        target_indices = [all_classes.index(cls) for cls in target_classes]
    # Handle the case where a specified class name is not in the dataset.
    except ValueError as e:
        print(f"Error: One of the target classes not found in CIFAR-100. {e}")
        return None, None
        
    # Create a mapping from the original class indices to new, contiguous indices (0, 1, 2, ...).
    label_map = {old_label: new_label for new_label, old_label in enumerate(target_indices)}

    # Define a helper function to filter a dataset based on the target classes.
    def _filter_dataset(dataset):
        # Convert the list of targets to a NumPy array for efficient boolean indexing.
        targets_np = np.array(dataset.targets)
        # Create a boolean mask to identify which samples belong to the target classes.
        indices_to_keep = np.isin(targets_np, target_indices)
        
        # Filter the dataset's image data using the boolean mask.
        dataset.data = dataset.data[indices_to_keep]
        
        # Get the original labels of the samples that are being kept.
        original_targets_to_keep = targets_np[indices_to_keep]
        # Remap the original labels to the new contiguous labels.
        dataset.targets = [label_map[target] for target in original_targets_to_keep]
        
        # Update the dataset's class list to only include the target classes.
        dataset.classes = target_classes
        return dataset

    print(f"\nFiltering for {len(target_classes)} classes...")
    # Apply the filtering logic to the full training dataset.
    train_dataset_subset = _filter_dataset(train_dataset_full)
    # Apply the filtering logic to the full test dataset.
    test_dataset_subset = _filter_dataset(test_dataset_full)
    print("Filtering complete. Returning training and validation datasets.")
    
    # Return the filtered training and test subsets.
    return train_dataset_subset, test_dataset_subset


# Keeps CSV rows whose text matches the keywords of exactly one category
def filter_dataset_by_keywords(input_path, output_path, category_keywords, text_column, columns_to_keep=None,
                               category_column='category', parse_as_list=False):
    """
    Keeps only the CSV rows whose text matches the keywords of exactly one category.

    Args:
        input_path: CSV file to read.
        output_path: CSV file to write the filtered rows to.
        category_keywords: Dict mapping a category name to a list of lowercase keywords,
            for example {'fruit': ['apple', 'mango'], 'vegetable': ['carrot']}.
        text_column: Column whose text is searched. Lists stored as strings are joined first when parse_as_list is True.
        columns_to_keep: Columns written to the output. All columns plus the category when None.
        category_column: Name of the new column that stores the category.
        parse_as_list: True when text_column holds a Python list written as a string.

    Returns:
        The filtered DataFrame, or None if the input file was not found.
    """
    print(f"Loading the raw dataset from '{input_path}'...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at '{input_path}'")
        return None

    def categorize(raw_text):
        """Returns the single matching category, or 'other' when none or several match."""
        try:
            text = ' '.join(ast.literal_eval(raw_text)).lower() if parse_as_list else str(raw_text).lower()
        except (ValueError, SyntaxError):
            return 'other'
        matches = [name for name, words in category_keywords.items() if any(word in text for word in words)]
        return matches[0] if len(matches) == 1 else 'other'
    print('Categorizing rows based on keywords...')
    df[category_column] = df[text_column].apply(categorize)
    filtered = df[df[category_column] != 'other'].copy()
    if columns_to_keep is not None:
        filtered = filtered[columns_to_keep]
    for name in category_keywords:
        print(f"Found {(filtered[category_column] == name).sum()} rows for '{name}'.")
    filtered.to_csv(output_path, index=False)
    print(f"Saved the filtered dataset to '{output_path}'.")
    return filtered


# Reads text files from data_dir/{train,test}/<class folder> into texts and labels
def load_text_folder_dataset(data_dir, class_dirs=None, max_train_samples=2000, max_test_samples=500):
    """
    Reads text files from a folder layout of data_dir/{train,test}/<class folder>/*.txt.

    Args:
        data_dir: Root folder with 'train' and 'test' sub-folders.
        class_dirs: Dict mapping each class folder name to its integer label. Defaults to {'pos': 1, 'neg': 0}.
        max_train_samples: Total number of training files, split evenly over the classes.
        max_test_samples: Total number of test files, split evenly over the classes.

    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    class_dirs = class_dirs or {'pos': 1, 'neg': 0}

    def read_split(split, limit):
        texts, labels = [], []
        for folder, label in class_dirs.items():
            directory = os.path.join(data_dir, split, folder)
            files = sorted(os.listdir(directory))[:limit // len(class_dirs)]
            for filename in tqdm(files, desc=f'Loading {split} {folder}', leave=False):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
        return texts, labels
    print('\nProcessing Training Data...')
    train_texts, train_labels = read_split('train', max_train_samples)
    print('Processing Test Data...')
    test_texts, test_labels = read_split('test', max_test_samples)
    print(f'\nSuccessfully loaded {len(train_texts)} training and {len(test_texts)} test samples.')
    return train_texts, train_labels, test_texts, test_labels


# Dataset wrapper that applies a transform to the image fields of tuple samples
class TransformedTupleDataset(Dataset):
    """
    Wraps a dataset of tuples, such as (anchor, positive, negative) or (img_a, img_b, label),
    and applies a transform to the image fields only.

    Args:
        subset: Dataset or Subset that returns tuples.
        transform: Transform applied to each image field. Nothing is applied when None.
        num_images: How many leading fields are images. All fields when None.
    """

    def __init__(self, subset, transform=None, num_images=None):
        self.subset = subset
        self.transform = transform
        self.num_images = num_images

    def __getitem__(self, index):
        sample = tuple(self.subset[index])
        if self.transform is None:
            return sample
        n = len(sample) if self.num_images is None else self.num_images
        return tuple(self.transform(item) if i < n else item for i, item in enumerate(sample))

    def __len__(self):
        return len(self.subset)


# Randomly splits a tuple dataset into train and validation sets with their own transforms
def split_dataset_with_transforms(full_dataset, train_split, train_transform, val_transform, num_images=None,
                                  seed=None, require_untransformed=True):
    """
    Randomly splits a tuple dataset into train and validation sets, each with its own transform.

    Args:
        full_dataset: Dataset that returns tuples and was built without a transform.
        train_split: Fraction of samples used for training.
        train_transform: Transform for the training images.
        val_transform: Transform for the validation images.
        num_images: How many leading tuple fields are images. All fields when None.
        seed: Optional seed for a reproducible split.
        require_untransformed: Raises an error if full_dataset already has a transform.

    Returns:
        (train_dataset, val_dataset)
    """
    if require_untransformed and getattr(full_dataset, 'transform', None) is not None:
        raise ValueError("The 'full_dataset' must be initialized with transform=None for this function to work correctly.")
    train_size = int(train_split * len(full_dataset))
    kwargs = {'generator': torch.Generator().manual_seed(seed)} if seed is not None else {}
    train_subset, val_subset = random_split(full_dataset, [train_size, len(full_dataset) - train_size], **kwargs)
    return (TransformedTupleDataset(train_subset, train_transform, num_images),
            TransformedTupleDataset(val_subset, val_transform, num_images))


# Computes class weights inversely proportional to class frequency
def compute_balanced_class_weights(labels):
    """
    Computes class weights inversely proportional to class frequency.

    Args:
        labels: Sequence of integer class labels for the training samples.

    Returns:
        Float tensor with one weight per class, in sorted class order.
    """
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(weights, dtype=torch.float32)


# Pairs images from two parallel folder trees that share sub-folder names
def index_paired_folders(dir_a, dir_b, id_prefix='', extension='.jpg', min_a=1, min_b=1):
    """
    Pairs up images that live in two parallel folder trees with the same sub-folder names.

    For example dir_a/ID_1/*.jpg and dir_b/ID_1/*.jpg.

    Args:
        dir_a: First root folder. Its sub-folders define the groups.
        dir_b: Second root folder with matching sub-folder names.
        id_prefix: Only sub-folders whose name starts with this prefix are used.
        extension: File extension to collect.
        min_a: Minimum number of images required in the group of dir_a.
        min_b: Minimum number of images required in the group of dir_b.

    Returns:
        Dict mapping group name to {'a': [paths], 'b': [paths]}.
    """
    if not os.path.isdir(dir_a):
        print(f'Error: Directory not found at: {dir_a}')
        return {}
    group_map = {}
    for group_id in sorted(os.listdir(dir_a)):
        if not group_id.startswith(id_prefix):
            continue
        files_a = glob.glob(os.path.join(dir_a, group_id, f'*{extension}'))
        group_b = os.path.join(dir_b, group_id)
        files_b = glob.glob(os.path.join(group_b, f'*{extension}')) if os.path.isdir(group_b) else []
        if len(files_a) >= min_a and len(files_b) >= min_b:
            group_map[group_id] = {'a': files_a, 'b': files_b}
    return group_map


# Finds image pairs that share a filename in two sibling folders of each category
def index_matched_pairs(base_dir, categories, dir_a='Before', dir_b='After'):
    """
    Finds image pairs that share a filename between two sibling folders inside each category folder.

    For example base_dir/<category>/Before/x.jpg and base_dir/<category>/After/x.jpg.

    Args:
        base_dir: Root folder with one sub-folder per category.
        categories: Category folder names to scan.
        dir_a: Name of the first folder inside each category.
        dir_b: Name of the second folder inside each category.

    Returns:
        Dict mapping category to a list of (path_a, path_b) tuples.
    """
    pair_map = {category: [] for category in categories}
    for category in categories:
        folder_a = os.path.join(base_dir, category, dir_a)
        folder_b = os.path.join(base_dir, category, dir_b)
        if not os.path.isdir(folder_a):
            continue
        for filename in os.listdir(folder_a):
            path_b = os.path.join(folder_b, filename)
            if os.path.exists(path_b):
                pair_map[category].append((os.path.join(folder_a, filename), path_b))
    return pair_map


# Maps each class index to the dataset positions that hold that class
def _group_indices_by_class(dataset, num_classes):
    """
    Maps each class index to the positions in the dataset that hold that class.

    Works for plain datasets with .targets, torch Subset objects, and wrappers that keep a .subset.
    """
    inner = getattr(dataset, 'subset', dataset)
    if hasattr(inner, 'indices') and hasattr(inner.dataset, 'targets'):
        targets = [inner.dataset.targets[i] for i in inner.indices]
    elif hasattr(inner, 'targets'):
        targets = list(inner.targets)
    else:
        targets = [label for _, label in dataset]
    groups = {i: [] for i in range(num_classes)}
    for position, label in enumerate(targets):
        groups[int(label)].append(position)
    return groups



# =============================================================================
# DATA EXPLORATION AND SAMPLE VISUALISATION
# =============================================================================


# Groups all files under a folder by their file extension
def explore_extensions(root_dir):
    """
    Explores a directory tree to group all file paths by their file extensions.

    Args:
        root_dir (str): The starting directory path to search.

    Returns:
        extensions (dict): A dictionary mapping file extensions to a list of matching file paths.
    """
    # Initialize a dictionary to store file paths, grouped by extension.
    extensions = {}
    # Walk through the directory tree starting from the root directory.
    for dirpath, _, filenames in os.walk(root_dir):
        # Iterate over each file in the current directory.
        for filename in filenames:
            # Extract the file extension and convert it to lowercase.
            ext = os.path.splitext(filename)[1].lower()
            # If the extension has not been seen before, add it to the dictionary.
            if ext not in extensions:
                # Initialize a new list for this extension.
                extensions[ext] = []
            # Append the full path of the file to the list for its extension.
            extensions[ext].append(os.path.join(dirpath, filename))
    # Return the dictionary of extensions and their corresponding file paths.
    return extensions


# Prints the folder tree of a directory down to a given depth
def print_data_folder_structure(root_dir, max_depth=1):
    """
    Prints the directory tree structure for a given root directory.

    Args:
        root_dir (str): The starting path for the directory tree.
        max_depth (int): The maximum depth for the tree traversal.
    """
    # Define the configuration settings for displaying the directory tree.
    config_tree = {
        # Specify the starting path for the directory tree.
        "dirPath": root_dir,
        # Set to False to include both files and directories.
        "onlyDirs": False,
        # Set the maximum depth for the tree traversal.
        "maxDepth": max_depth,
        # Specify a sorting option (100 typically means no specific sort).
        "sortBy": 100,
    }
    # Create and display the tree structure using the unpacked configuration.
    DisplayTree(**config_tree)


# Shows one random image per class from a dataset
def show_sample_images(dataset, class_names, mean=None, std=None):
    """
    Displays a grid of sample images from the dataset.

    This function creates a plot showing one randomly selected image from each
    class and uses the provided `class_names` list for the titles.

    Args:
        dataset (Dataset): The dataset to visualize. Must have a '.classes'
                           attribute and support subset indexing.
        class_names (list of str): A list of formatted class names for the plot titles.
        mean: Optional per-channel mean used for normalization. Images are shown as is when None.
        std: Optional per-channel std used for normalization.
    """
    # Get the total number of classes from the dataset.
    num_classes = len(dataset.classes)
    
    # Validate that the number of class names matches the number of classes.
    assert len(class_names) == num_classes, "Length of class_names list must match the number of classes in the dataset."

    # Create a mapping of class index to all its image indices.
    class_to_indices = _group_indices_by_class(dataset, num_classes)

    # Dynamically calculate the grid size for the plot.
    ncols = min(7, num_classes)
    nrows = (num_classes + ncols - 1) // ncols  # Ceiling division for rows.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, nrows * 2.2), squeeze=False)

    # Loop through each class to display one random sample.
    for i, ax in enumerate(axes.flatten()):
        # Hide axes for any empty subplots.
        if i >= num_classes:
            ax.axis('off')
            continue

        # Set the plot title using the provided class names.
        class_name = class_names[i]

        # Pick a random image from the current class.
        random_image_idx = random.choice(class_to_indices[i])

        # Retrieve the image and label from the dataset.
        image, label = dataset[random_image_idx]

        # Un-normalize the image for proper display.
        # Assumes an 'unnormalize' function is available.
        if mean is not None and std is not None:
            image = unnormalize(image, mean, std)

        # Prepare the image tensor for plotting.
        npimg = image.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(class_name)
        ax.axis('off')

    # Apply a tight layout and show the plot.
    plt.tight_layout()
    plt.show()


# Displays a specified number of random images from a given dataset category
def show_random_images(split, category, dataset_path, num_images=5, seed=6):
    """Displays a specified number of random images from a given dataset category.

    Args:
        split: The dataset split to choose images from (e.g., 'train', 'test').
        category: The category or class of images to display.
        dataset_path: The root path to the dataset directory.
        num_images: The number of random images to display.
        seed: The random seed for reproducibility of image selection.
    """
    # Construct the path to the target folder
    folder = os.path.join(dataset_path, split, category)
    # Set the random seed for reproducibility
    random.seed(seed)
    # Randomly sample a list of image filenames from the folder
    image_files = random.sample(os.listdir(folder), num_images)

    # Create a figure to display the images
    plt.figure(figsize=(15, 3))
    # Loop through the selected image files and display them
    for i, fname in enumerate(image_files):
        # Get the full path of the image
        img_path = os.path.join(folder, fname)
        # Open and convert the image to RGB
        img = Image.open(img_path).convert("RGB")
        # Create a subplot for the current image
        plt.subplot(1, num_images, i + 1)
        # Display the image
        plt.imshow(img)
        # Assign the category to a variable for the title
        group = category
        # Set the title for the subplot
        plt.title(f"{split}/{group}")
        # Turn off the axes for a cleaner look
        plt.axis("off")
    # Show the final plot with all images
    plt.show()


# Displays a grid of random images from the dataset
def display_random_images(path):
    """
    Displays a grid of random images from the dataset.
    
    This function shows two random images for each class, ensuring the rows
    always follow the order: NORMAL, BACTERIAL_PNEUMONIA, VIRAL_PNEUMONIA.

    Args:
        path: The path to the training data directory (e.g., './chest_xray/train').
    """
    try:
        # Define the exact order for class display
        class_names = ['NORMAL', 'BACTERIAL_PNEUMONIA', 'VIRAL_PNEUMONIA']
        # Get the number of classes
        num_classes = len(class_names)
        
        # Create a figure with a row for each class and two columns for images
        fig, axs = plt.subplots(num_classes, 2, figsize=(7, 7))

        # Loop through each class in the predefined order
        for i, class_name in enumerate(class_names):
            # Construct the path to the current class directory
            class_path = os.path.join(path, class_name)
            
            # Check if the class directory actually exists
            if not os.path.isdir(class_path):
                # Print a warning and turn off the axes for the missing directory
                print(f"Warning: Directory for class '{class_name}' not found. Skipping row.")
                # Turn off the axis for the first subplot in the current row
                axs[i, 0].axis('off')
                # Turn off the axis for the second subplot in the current row
                axs[i, 1].axis('off')
                # Continue to the next iteration of the loop
                continue
            
            # Get a list of all image files in the class directory
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Check if there are at least two images in the class directory
            if len(image_files) < 2:
                # Print a warning and turn off the axes if there are not enough images
                print(f"Warning: Class '{class_name}' has fewer than 2 images. Skipping.")
                # Turn off the axis for the first subplot in the current row
                axs[i, 0].axis('off')
                # Turn off the axis for the second subplot in the current row
                axs[i, 1].axis('off')
                # Continue to the next iteration of the loop
                continue

            # Randomly sample two unique image files
            selected_images = random.sample(image_files, 2)
            
            # Display the two selected images
            for j, image_file in enumerate(selected_images):
                # Construct the full path to the image file
                image_path = os.path.join(class_path, image_file)
                # Open the image using the PIL library
                image = Image.open(image_path)
                
                # Select the appropriate subplot for the current image
                ax = axs[i, j]
                # Display the image on the subplot with a grayscale colormap
                ax.imshow(image, cmap='gray')
                # Turn off the axis to hide ticks and labels
                ax.axis('off')
                # Set the title of the subplot to the class name
                ax.set_title(class_name, fontsize=12)

        # Adjust the subplot parameters for a tight layout
        plt.tight_layout()
        # Display the figure with the images
        plt.show()

    except FileNotFoundError:
        # Handle the case where the main directory is not found
        print(f"Error: The directory '{path}' was not found.")
    except Exception as e:
        # Handle any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")


# Shows a few sample images for each label of a dataset
def display_some_images(dataset):
    plt.figure(figsize=(15, 10))
    
    # Get 3 images for each label
    images_per_label = 3
    num_labels = 3
    
    for label in range(num_labels):
        # Find indices of images with this label
        label_indices = [i for i in range(len(dataset)) if dataset[i][1] == label]
        # Take first 3 images of this label
        selected_indices = label_indices[:images_per_label]
        
        for i, img_idx in enumerate(selected_indices):
            # Get image and label
            img, label = dataset[img_idx]
            
            # Convert tensor to numpy array and transpose to correct dimensions
            img = img.permute(1, 2, 0).numpy()
            
            # Denormalize the image
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # Plot in grid - each row is a label
            plt.subplot(3, 3, label * images_per_label + i + 1)
            plt.imshow(img)
            plt.title(f'Label: {dataset.classes[label]}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()


# Visual exploration of the dataset by displaying random samples in a grid
def visual_exploration(dataset, num_rows=2, num_cols=4):
    """Visual exploration of the dataset by displaying random samples in a grid."""
    # Calculate total number of samples to display
    total_samples = num_rows * num_cols

    # Randomly select indices from the dataset without replacement
    # This ensures we get a diverse sample of the dataset
    indices = np.random.choice(len(dataset), total_samples, replace=False)

    # Create a grid of subplots with appropriate figure size
    # Each subplot gets (3 x 4) inches per image for good visibility
    fig, axes = get_grid(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 4))

    # Iterate through each axis and corresponding random index
    for ax, idx in zip(axes.flatten(), indices):
        # Load image and label from dataset at the random index
        image, label = dataset[idx]

        # Get human-readable description for the label
        description = dataset.get_label_description(label)

        # Create a combined label string with both number and description
        label = f"{label} - {description}"

        # Create info string showing index and image dimensions
        info = f"Index: {idx} Size: {image.size}"

        # Plot the image on the current axis with label and info
        plot_img(image, label=label, info=info, ax=ax)

    # Display the complete grid of images
    plt.show()


# Shows a grid with one random dataset image per class
def visualise_images(dataset, grid):
    """
    Displays a grid of images from a dataset, with one random image per class.

    Args:
        dataset: The dataset object containing the images and labels.
        grid (tuple): A tuple specifying the number of rows and columns for the image grid.
    """

    # Create a shallow copy of the dataset to avoid modifying the original
    dataset_copy = copy.copy(dataset)
    # Set the transform on the copied dataset to convert images to tensors
    dataset_copy.transform = torchvision.transforms.ToTensor()

    # Create a DataLoader to handle batching and shuffling of the data
    loader = DataLoader(dataset_copy, batch_size=64, shuffle=True)

    # Unpack the grid dimensions from the input tuple
    rows, cols = grid
    # Calculate the total number of images to display in the grid
    num_images_to_show = rows * cols

    # Get the dataset object from the DataLoader
    dataset_to_show = loader.dataset

    # Create a dictionary to store lists of indices for each class
    class_indices = defaultdict(list)
    # Iterate through the dataset to populate the class_indices dictionary
    for idx, target in enumerate(dataset_to_show.targets):
        class_indices[target].append(idx)
        
    # Get the list of class names from the dataset
    class_names = dataset_to_show.classes

    # Create a figure and a set of subplots for the grid layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Iterate over each subplot in the grid
    for i, ax in enumerate(axes.flat):
        # If the current index is out of bounds, turn off the subplot axis
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Set the class label based on the current iteration index
        class_label = i
        
        # Get the list of image indices for the current class
        indices_for_class = class_indices[class_label]
        # If there are no images for this class, turn off the subplot axis
        if not indices_for_class:
            ax.axis('off')
            continue

        # Choose a random image index from the list for the current class
        random_image_index = random.choice(indices_for_class)
        
        # Retrieve the image tensor and its corresponding label from the dataset
        image_tensor, _ = dataset_to_show[random_image_index]
        
        # Convert the tensor to a NumPy array and transpose dimensions for display
        img_to_display = image_tensor.numpy().transpose((1, 2, 0))
        
        # Get the name of the class corresponding to the class label
        class_name = class_names[class_label]
        
        # Display the image on the current subplot
        ax.imshow(img_to_display)
        
        # Set the title of the subplot to the capitalized class name
        ax.set_title(class_name.capitalize(), fontsize=16)
        # Turn off the axis for a cleaner look
        ax.axis('off')

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Display the plot
    plt.show()

    # Clean up the copied dataset to free up memory
    del dataset_copy


# Shows a grid with one random image per class taken from a data loader
def visualise_images_from_loader(loader, grid):
    """
    Visualizes a grid of random images from a dataset, showing one image per class.

    Args:
        loader: The DataLoader object containing the dataset.
        grid: A tuple specifying the grid dimensions as (rows, cols).
    """
    # Unpack the grid dimensions from the input tuple
    rows, cols = grid
    # Calculate the total number of images to display in the grid
    num_images_to_show = rows * cols

    # Get the dataset object from the DataLoader
    dataset_to_show = loader.dataset

    # Create a dictionary to store lists of indices for each class
    class_indices = defaultdict(list)
    # Iterate through the dataset to populate the class_indices dictionary
    for idx, target in enumerate(dataset_to_show.targets):
        class_indices[target].append(idx)
        
    # Get the list of class names from the dataset
    class_names = dataset_to_show.classes

    # Create a figure and a set of subplots for the grid layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Iterate over each subplot in the grid
    for i, ax in enumerate(axes.flat):
        # If the current index is out of bounds, turn off the subplot axis
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Set the class label based on the current iteration index
        class_label = i
        
        # Get the list of image indices for the current class
        indices_for_class = class_indices[class_label]
        # If there are no images for this class, turn off the subplot axis
        if not indices_for_class:
            ax.axis('off')
            continue

        # Choose a random image index from the list for the current class
        random_image_index = random.choice(indices_for_class)
        
        # Retrieve the image tensor and its corresponding label from the dataset
        image_tensor, _ = dataset_to_show[random_image_index]
        
        # Convert the tensor to a NumPy array and transpose dimensions for display
        img_to_display = image_tensor.numpy().transpose((1, 2, 0))
        
        # Rescale the pixel values from the normalized range to [0, 1] for proper visualization
        min_val = img_to_display.min()
        max_val = img_to_display.max()
        img_to_display = (img_to_display - min_val) / (max_val - min_val)
        
        # Get the name of the class corresponding to the class label
        class_name = class_names[class_label]
        
        # Display the image on the current subplot
        ax.imshow(img_to_display)
        
        # Set the title of the subplot to the capitalized class name
        ax.set_title(class_name.capitalize(), fontsize=16)
        # Turn off the axis for a cleaner look
        ax.axis('off')

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Display the plot
    plt.show()


# Displays a 3x3 grid of images from a dataloader, reversing normalization for visualization
def show_image_grid(dataloader, class_names):
    """
    Displays a 3x3 grid of images from a dataloader, reversing normalization for visualization.

    Args:
        dataloader (DataLoader): The PyTorch DataLoader containing the image batches.
        class_names (list): A list of strings representing the names of the dataset classes.

    Returns:
        None: This function displays a plot and does not return a value.
    """
    # Extract a single batch of images and their corresponding labels from the provider
    images, labels = next(iter(dataloader))
    
    # Define the specific mean values used during the initial normalization process
    mean = torch.tensor([0.630, 0.554, 0.489])
    # Define the specific standard deviation values used during the initial normalization process
    std = torch.tensor([0.248, 0.271, 0.319])

    # Initialize a figure with a 3 by 3 grid of subplots for image display
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    
    # Iterate through the grid axes to populate each subplot with an image
    for i, ax in enumerate(axes.flatten()):
        # Validate that the current index does not exceed the available images in the batch
        if i < len(images):
            # Select the image tensor at the current index
            image = images[i]
            # Select the label index at the current index
            label = labels[i]
            
            # Reverse the normalization by multiplying by standard deviation and adding the mean
            # Reshaping is performed to align dimensions for broadcasting across the tensor
            unnormalized_image = image * std[:, None, None] + mean[:, None, None]
            
            # Reorder the tensor dimensions from (Channels, Height, Width) to (Height, Width, Channels)
            img_display = unnormalized_image.permute(1, 2, 0).numpy()
            
            # Constrain the pixel values to the standard range of 0 to 1
            img_display = np.clip(img_display, 0, 1)
            
            # Render the processed image onto the current subplot axis
            ax.imshow(img_display)
            # Set the subplot title to the human-readable class name
            ax.set_title(class_names[label])
            # Disable the axis markers for a cleaner visual representation
            ax.axis("off")
    
    # Adjust the layout to prevent overlapping elements in the grid
    plt.tight_layout()
    # Execute the command to render the final plot
    plt.show()


# Shows random images from several dataset sub-folders, one row per folder
def display_train_images(dataset_path, folders=('train/real', 'train/fake'), images_per_folder=3,
                         thumbnail_size=(256, 256), valid_exts=('.jpg', '.jpeg', '.png')):
    """
    Shows random images from several sub-folders of a dataset, one row per folder.

    Args:
        dataset_path: Root folder of the dataset.
        folders: Sub-folders (relative to dataset_path) to sample from. Each one is a titled row.
        images_per_folder: Number of random images shown per folder.
        thumbnail_size: Maximum size each image is shrunk to before drawing.
        valid_exts: File extensions that count as images.
    """
    images_to_plot = []
    for title in folders:
        directory = os.path.join(dataset_path, *title.split('/'))
        try:
            with os.scandir(directory) as entries:
                image_files = [e.path for e in entries if e.is_file() and e.name.lower().endswith(valid_exts)]
        except FileNotFoundError:
            print(f"Error: Directory not found at '{directory}'.")
            return
        if len(image_files) < images_per_folder:
            print(f"Warning: Not enough images in '{directory}'. Found {len(image_files)}, need {images_per_folder}.")
            continue
        for img_path in random.sample(image_files, images_per_folder):
            images_to_plot.append((img_path, title))
    if len(images_to_plot) != len(folders) * images_per_folder:
        print('Could not gather enough images to display. Aborting.')
        return
    fig, axes = plt.subplots(len(folders), images_per_folder, figsize=(5 * images_per_folder, 5 * len(folders)))
    for ax, (img_path, title) in zip(np.atleast_1d(axes).ravel(), images_to_plot):
        try:
            with Image.open(img_path) as img:
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')
        except (UnidentifiedImageError, FileNotFoundError):
            ax.set_title(f'Error loading image\n{os.path.basename(img_path)}', color='red')
            ax.axis('off')
    plt.tight_layout()
    plt.show()


# Shows one random image from each class folder of a dataset
def plot_samples_from_dataset(dataset_path, class_names=None, grid=None, extensions=('.jpg', '.jpeg', '.png')):
    """
    Shows one random image from each class folder of a dataset.

    Args:
        dataset_path: Root folder that contains one sub-folder per class.
        class_names: Folder names to show. All sub-folders in sorted order when None.
        grid: Optional (rows, cols). Defaults to rows of up to 3 images.
        extensions: File extensions that count as images.
    """
    if class_names is None:
        class_names = sorted(d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)))
    n = len(class_names)
    rows, cols = grid if grid else (int(np.ceil(n / min(3, n))), min(3, n))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis('off')
    for ax, class_name in zip(np.atleast_1d(axes).ravel(), class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            ax.text(0.5, 0.5, f'{class_name}\nNot Found', ha='center', va='center')
            continue
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(extensions)]
        if not image_files:
            ax.text(0.5, 0.5, 'No Images', ha='center', va='center')
            continue
        ax.imshow(plt.imread(os.path.join(class_dir, random.choice(image_files))))
        ax.set_title(class_name, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Plots grouped bar charts of image counts per group for two image sets
def plot_group_counts(group_map, labels=('A', 'B'), colors=('royalblue', 'salmon'), chunk_size=17, title_prefix='Counts for'):
    """
    Plots grouped bar charts of how many images each group has in its two folders.

    Args:
        group_map: Output of index_paired_folders.
        labels: Legend names of the two image sets.
        colors: Bar colors of the two image sets.
        chunk_size: Maximum number of groups per chart.
        title_prefix: Text before the group range in each chart title.
    """
    if not group_map:
        print('No valid groups found.')
        return

    def sort_key(group_id):
        numbers = re.findall(r'\d+', group_id)
        return (int(numbers[-1]) if numbers else 0, group_id)
    ids = sorted(group_map, key=sort_key)
    counts_a = [len(group_map[g]['a']) for g in ids]
    counts_b = [len(group_map[g]['b']) for g in ids]
    print(f'Found {len(ids)} valid groups.')
    print(f'   - Total {labels[0]} images: {sum(counts_a)}')
    print(f'   - Total {labels[1]} images: {sum(counts_b)}\n')
    for start in range(0, len(ids), chunk_size):
        end = min(start + chunk_size, len(ids))
        x = np.arange(end - start)
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 8))
        rects_a = ax.bar(x - width / 2, counts_a[start:end], width, label=f'{labels[0]} images', color=colors[0])
        rects_b = ax.bar(x + width / 2, counts_b[start:end], width, label=f'{labels[1]} images', color=colors[1])
        ax.set_ylabel('Image Count')
        ax.set_title(f'{title_prefix} groups {start + 1}-{end}', fontsize=18, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(ids[start:end], rotation=45, ha='right')
        ax.legend()
        ax.bar_label(rects_a, padding=3, fontsize=12)
        ax.bar_label(rects_b, padding=3, fontsize=12)
        fig.tight_layout()
        plt.show()


# Shows a dict of counts as a styled table with a total row
def display_counts_table(counts, key_header='Category', value_header='Count'):
    """
    Shows a dict of counts as a styled table with a total row.

    Args:
        counts: Dict mapping a name to a number.
        key_header: Header of the name column.
        value_header: Header of the number column.
    """
    if not counts:
        print('Nothing to display.')
        return
    df = pd.DataFrame({key_header: list(counts.keys()), value_header: list(counts.values())})
    df = df.sort_values(by=key_header).reset_index(drop=True)
    total_row = pd.DataFrame([{key_header: '<b>Total</b>', value_header: df[value_header].sum()}])
    styler = pd.concat([df, total_row], ignore_index=True).style.hide(axis='index')
    styler.set_table_styles([{'selector': 'table', 'props': [('width', '60%'), ('margin', '0')]},
                             {'selector': 'td', 'props': [('text-align', 'left'), ('padding', '8px')]},
                             {'selector': 'th', 'props': [('text-align', 'left'), ('padding', '8px')]}])
    display(styler)


# Shows one random image from each of two image sets of a random group
def show_random_group_pair(group_map, labels=('Group A', 'Group B'), cmap='gray', title='Comparison for {group}'):
    """
    Shows one random image from each of the two image sets of a random group.

    Args:
        group_map: Output of index_paired_folders.
        labels: Titles of the two images.
        cmap: Colormap for grayscale images.
        title: Figure title. '{group}' is replaced by the group name.
    """
    if not group_map:
        print('No valid groups were found.')
        return
    group = random.choice(list(group_map.keys()))
    path_a = random.choice(group_map[group]['a'])
    path_b = random.choice(group_map[group]['b'])
    try:
        img_a, img_b = Image.open(path_a), Image.open(path_b)
    except FileNotFoundError as e:
        print(f'Error: Could not find image file. {e}')
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title.format(group=group), fontsize=16)
    for ax, img, label in zip(axes, (img_a, img_b), labels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label)
        ax.axis('off')
    plt.show()


# Shows one random image pair per category, stacked vertically
def show_random_pair_per_category(pair_map, labels=('Before', 'After')):
    """
    Shows one random image pair for each category, stacked vertically.

    Args:
        pair_map: Output of index_matched_pairs, mapping category to a list of (path_a, path_b).
        labels: Titles of the left and right image.
    """
    if not any(pair_map.values()):
        print('The pair map is empty. Cannot display image pairs.')
        return
    categories = list(pair_map.keys())
    fig = plt.figure(figsize=(8, 5 * len(categories)))
    outer_grid = GridSpec(len(categories), 1, figure=fig, hspace=0.2)
    for i, category in enumerate(categories):
        inner = outer_grid[i].subgridspec(2, 2, height_ratios=[0.05, 1], wspace=0.05, hspace=0.1)
        ax_title = fig.add_subplot(inner[0, :])
        ax_title.text(0.5, 0.5, category.replace('_', ' '), ha='center', va='center', fontsize=18, weight='bold')
        ax_title.axis('off')
        axes = [fig.add_subplot(inner[1, 0]), fig.add_subplot(inner[1, 1])]
        for ax, label in zip(axes, labels):
            ax.set_title(label, fontsize=16)
            ax.axis('off')
        pairs = pair_map[category]
        if not pairs:
            for ax in axes:
                ax.text(0.5, 0.5, 'No Images Found', ha='center', va='center', fontsize=12)
            continue
        try:
            for ax, path in zip(axes, random.choice(pairs)):
                ax.imshow(Image.open(path))
        except FileNotFoundError:
            for ax in axes:
                ax.text(0.5, 0.5, 'Image Not Found', ha='center', va='center', fontsize=12)
    plt.show()


# Shows the first sample of the next batch as images, with an optional label name
def show_random_batch_sample(dataloader, image_titles, mean=None, std=None, label_names=None,
                             suptitle='Example sample'):
    """
    Shows the first sample of the next batch: its image fields side by side, plus an optional label.

    Works for triplets (anchor, positive, negative) and for pairs with a label (img_a, img_b, label).

    Args:
        dataloader: DataLoader that yields tuples of batched tensors.
        image_titles: One title per image field. The number of titles sets how many fields are drawn as images.
        mean: Optional per-channel mean to undo normalization. Images are drawn as is when None.
        std: Optional per-channel std to undo normalization.
        label_names: Optional dict mapping the integer label that follows the images to a name.
        suptitle: Figure title. '{label}' is replaced by the label name or value.
    """
    if not dataloader:
        print('DataLoader is not available.')
        return
    batch = next(iter(dataloader))
    n = len(image_titles)
    label = batch[n][0].item() if len(batch) > n else None
    if label is not None and label_names is not None:
        label = label_names.get(label, 'Unknown')
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(suptitle.format(label=label), fontsize=16)
    for ax, field, title in zip(np.atleast_1d(axes), batch[:n], image_titles):
        sample = field[0]
        if mean is not None and std is not None:
            image = deprocess_image(sample, mean, std)
        else:
            image = sample.detach().cpu().permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



# =============================================================================
# MODEL BUILDING, INSPECTION AND SAVING
# =============================================================================


# Builds an EfficientNet-B0 embedding network from local weights
def get_efficientnet_embedding_backbone(embedding_dim=128, pretrained=True,
    weights_path="./pretrained_efficientnet_weights/efficientnet_b0_rwightman-7f5810bc.pth"):
    """
    Builds an embedding network from an EfficientNet-B0 model, loading weights offline.

    This function configures an EfficientNet-B0 model to act as a feature extractor
    by replacing its final classification layer with a linear layer that outputs
    an embedding of a specified dimension. It supports loading weights from a local file.

    Args:
        embedding_dim (int): The desired dimension of the output embedding vector.
        pretrained (bool): If True, loads weights from the local `weights_path`.
        weights_path (str): The local file path to the .pth weights file.

    Returns:
        torch.nn.Module: The modified EfficientNet-B0 model.
        
    Raises:
        FileNotFoundError: If `pretrained` is True and the weights file is not found.
    """
    # Instantiate the EfficientNet-B0 model architecture with random weights.
    model = tv_models.efficientnet_b0(weights=None)

    # If specified, load the pretrained weights from a local file.
    if pretrained:
        # Raise an error if the weights file is missing to prevent silent failures.
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at the specified path: {weights_path}")
        
        # Load the state dictionary from the path, mapping to CPU to avoid device errors.
        state_dict = torch.load(weights_path, map_location='cpu')
        # Load the state dictionary into the model.
        model.load_state_dict(state_dict)
    
    # Modify the final classification layer to produce an embedding.
    # Get the number of input features from the original classifier layer.
    num_ftrs = model.classifier[1].in_features
    # Replace the layer with a new linear layer of the desired embedding dimension.
    model.classifier[1] = nn.Linear(num_ftrs, embedding_dim)
    
    return model


# Initializes a DenseNet-121 model and prepares it for transfer learning
def load_pretrained_densenet(num_classes, seed=None, pretrained=True, train_classifier_only=True,
                             weights_path="./pretrained_densenet_weights/densenet121-a639ec97.pth"):
    """Initializes a DenseNet-121 model and prepares it for transfer learning.

    This function loads the DenseNet-121 architecture, optionally with weights
    pre-trained on ImageNet. It then replaces the final classifier layer to
    adapt the model for a new task with a specified number of classes and can
    freeze the feature extraction layers.

    Args:
        num_classes (int): The number of output classes for the new classifier head.
        seed (int, optional): A random seed for reproducible weight initialization
                              of the new classifier. Defaults to None.
        pretrained (bool, optional): If True, loads pre-trained weights from the
                                     specified path. Defaults to True.
        train_classifier_only (bool, optional): If True, freezes the feature extractor
                                                layers so only the new classifier is trained.
                                                Defaults to True.
        weights_path (str, optional): The local file path to the pre-trained weights.

    Returns:
        torch.nn.Module: The configured DenseNet-121 model.
    """
    
    # Set the manual seed for PyTorch for reproducible weight initialization for the classifier head.
    if seed is not None:
        torch.manual_seed(seed)
    
    # Instantiate the DenseNet-121 model from torchvision without its own pretrained weights.
    model = tv_models.densenet121(weights=None)

    # Conditionally load pretrained weights from a local file if the flag is set.
    if pretrained:
        # Check if the specified weights file exists and raise an error if not.
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at path: {weights_path}")

        # Load the state dictionary from the file, mapping it to the CPU to prevent device mismatches.
        state_dict = torch.load(weights_path, map_location='cpu')
        # Apply the loaded weights to the model architecture.
        model.load_state_dict(state_dict)

    # If configured for transfer learning, freeze the parameters of the feature extractor.
    if train_classifier_only:
        # Iterate through all model parameters and disable gradient calculations.
        for param in model.parameters():
            param.requires_grad = False

    # Retrieve the number of input features for the model's original classifier.
    num_ftrs = model.classifier.in_features
    # Replace the original classifier with a new, untrained linear layer.
    # The new layer will have the correct number of output units for the new task.
    # By default, its parameters will have requires_grad=True.
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # Return the modified and configured model.
    return model


# Initializes a ResNet18 model and loads weights from a local file
def load_resnet18(path='./pretrained_resnet18_weights/resnet18-f37072fd.pth'):
    """
    Initializes a ResNet18 model and loads weights from a local file.

    Args:
        path: The file system path to the model weight file.
    """
    # Instantiate the model architecture without downloading weights automatically
    model = tv_models.resnet18(weights=None)
    
    # Define the local source for the model parameters
    weights_path = path
    # Load the state dictionary from the specified file path
    state_dict = torch.load(weights_path)
    
    # Map the loaded parameters into the model architecture
    model.load_state_dict(state_dict)

    # Return the prepared model for inference or training
    return model


# Modifies the final fully connected layer of the model to match a new number of classes
def replace_final_layer(model, num_classes):
    """
    Modifies the final fully connected layer of the model to match a new number of classes.

    Args:
        model: The neural network model whose final layer is to be replaced.
        num_classes: The number of output classes for the new classification task.
    """
    # Identify the number of input features entering the final layer
    num_ftrs = model.fc.in_features
    
    # Initialize a new linear layer with the correct output dimensions
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Return the updated model architecture
    return model


# Freezes a pretrained model and replaces its last layer for a new number of classes
def adapt_model_for_transfer_learning(model, num_classes):
    """
    Modifies a pre-trained model by freezing its weights and replacing the final 
    classification layer to match a new number of output classes.

    Args:
        model (torch.nn.Module): The pre-trained neural network model to be adapted.
        num_classes (int): The number of output classes for the new task.

    Returns:
        model (torch.nn.Module): The modified model with frozen base layers and a 
                                 newly initialized final linear layer.
    """
    # Iterate through all parameters in the model to disable gradient updates
    for param in model.parameters():
        # Set the gradient requirement to False to freeze the layer weights
        param.requires_grad = False
    
    # Retrieve the number of input features entering the existing final fully connected layer
    num_ftrs = model.fc.in_features
    
    # Define a new linear layer with the correct number of outputs and assign it to the model
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Return the updated model architecture
    return model


# Calculates the size of a model's state_dict in megabytes (MB) using a temporary file
def get_model_size(model):
    """
    Calculates the size of a model's state_dict in megabytes (MB) using a temporary file.

    Args:
        model (torch.nn.Module): The model whose storage size is to be measured.

    Returns:
        size_mb (float): The size of the model weights in megabytes.
    """

    # Create a temporary file using the tempfile module
    with tempfile.NamedTemporaryFile(delete=False, suffix=".p") as temp_file:
        # Retrieve the path of the created temporary file
        temp_file_path = temp_file.name
        # Save the model's state_dict to the temporary file
        torch.save(model.state_dict(), temp_file_path)

    # Get the size of the saved file in bytes
    size_bytes = os.path.getsize(temp_file_path)

    # Convert the size from bytes to megabytes
    size_mb = size_bytes / (1024 * 1024)

    # Clean up by deleting the temporary file from the system
    os.remove(temp_file_path)

    # Return the calculated size
    return size_mb


# Measures the average inference time of a PyTorch model in milliseconds
def measure_average_inference_time_ms(model, input_shape=(1, 3, 32, 32), num_runs=100):
    """
    Measures the average inference time of a PyTorch model in milliseconds.
    
    This function forces execution on the CPU to ensure compatibility with 
    quantized models and provide a consistent baseline for latency.

    Args:
        model (torch.nn.Module): The neural network model to be benchmarked.
        input_shape (tuple): The shape of the dummy input tensor to use for timing.
        num_runs (int): The number of inference iterations to perform for averaging.

    Returns:
        avg_time (float): The average time taken for a single forward pass in milliseconds.
    """
    
    # Force the entire measurement to run on the CPU
    device = torch.device("cpu")
    
    # Move the model to the CPU
    model.to(device)
    
    # Create a random input tensor on the CPU
    input_tensor = torch.rand(input_shape).to(device)
    
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for the warm-up and measurement phases
    with torch.no_grad():
        # Execute warm-up runs to ensure system caches and operations are initialized
        for _ in range(10):
            model(input_tensor)

    # Initialize a list to store the duration of each run
    timings = []
    
    # Begin the measurement phase
    with torch.no_grad():
        # Iterate for the specified number of runs
        for _ in range(num_runs):
            # Record the start time of the forward pass
            start_time = time.time()
            
            # Execute a single inference pass
            model(input_tensor)
            
            # Record the end time of the forward pass
            end_time = time.time()
            
            # Calculate duration and convert seconds to milliseconds
            timings.append((end_time - start_time) * 1000)

    # Calculate the average inference time across all measured runs
    avg_time = sum(timings) / len(timings)
    
    # Return the resulting average time
    return avg_time


# Measures the average time per epoch for data loading and processing
def measure_average_epoch_time(loader, device):
    """
    Measures the average time per epoch for data loading and processing.

    Args:
        loader: The data loader to be measured.
        device: The device to which data is moved.

    Returns:
        The average time in seconds for one epoch.
    """
    # A list to store the time taken for each epoch
    epoch_times = []
    # Set the total number of epochs to run
    total_epochs = 5
    # Set the number of initial epochs to be considered as warm-up
    warmup_epochs = 2

    # Iterate over the total number of epochs
    for epoch in tqdm(range(total_epochs), desc="Overall Progress"):
        # Record the start time of the epoch
        start_time = time.time()
        # Iterate over the data in the loader
        for i, data in enumerate(loader, 0):
            # Unpack the data and labels
            inputs, labels = data
            # Move the inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
        # Record the end time of the epoch
        end_time = time.time()

        # Calculate the duration of the current epoch
        current_time = end_time - start_time
        # Add the current epoch's time to the list
        epoch_times.append(current_time)

    # Print a new line for better formatting
    print("")
    # Iterate through the collected epoch times to print them
    for epoch, current_time in enumerate(epoch_times):
        # Create a note if the epoch is a warm-up epoch
        warmup_note = "(warm-up)" if epoch < warmup_epochs else ""
        # Print the time taken for the current epoch
        print(
            f"  Epoch {epoch+1}/{total_epochs} | Time: {current_time:.2f} seconds {warmup_note}"
        )

    # Slice the list to get the times from the measured epochs
    measured_epochs = epoch_times[warmup_epochs:]
    # Calculate the average time of the measured epochs
    average_time = np.mean(measured_epochs)

    # Calculate the number of epochs used for the average
    num_measured_epochs = total_epochs - warmup_epochs
    # Print the final average execution time
    print(
        f"\nAverage execution time (avg of last {num_measured_epochs}): {average_time:.2f} seconds\n"
    )
    # Return the calculated average time
    return average_time


# Displays a detailed overview of the model architecture and its configuration
def print_summary(model, vocab_size=None, embedding_dim=128, num_heads=None):
    """
    Displays a detailed overview of the model architecture and its configuration.

    Args:
        model: The neural network module to be summarized.
        vocab_size: The total count of unique tokens in the vocabulary.
        embedding_dim: The dimensionality of the vector space for embeddings.
        num_heads: The count of parallel attention heads used in the architecture.

    Returns:
        None. This function outputs the summary information directly to the console.
    """
    # Determine the most efficient hardware available for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device available: {device}")
    
    # Transfer all model parameters and buffers to the identified device
    model = model.to(device)
    print(f"\nModel moved to {device}")
    print("Ready for training!")
    
    # Attempt to determine the vocabulary size from the model's internal layers if not explicitly provided
    if vocab_size is None:
        try:
            # Iterate through all sub-modules to locate the embedding layer
            for module in model.modules():
                if isinstance(module, torch.nn.Embedding):
                    # Retrieve the number of embeddings from the specific layer
                    vocab_size = module.num_embeddings
                    break
        except:
            # Revert to a fallback value if retrieval fails
            vocab_size = "Unknown"

    # Attempt to determine the number of attention heads from the model layers if not provided
    if num_heads is None:
        try:
            # Iterate through sub-modules to find MultiheadAttention instances
            for module in model.modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    # Extract the head count from the layer attributes
                    num_heads = module.num_heads
                    break
        except:
            # Set to unknown if the head count cannot be inferred
            num_heads = "Unknown"
    
    # Identify the specific class name of the provided model instance
    model_name = model.__class__.__name__
    
    # Compute the total count of elements across all parameter tensors
    total_params = sum(p.numel() for p in model.parameters())
    
    # Output the structured model specification report
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Vocabulary size: {vocab_size if vocab_size else 'Unknown'}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of attention heads: {num_heads}")
    print(f"Total parameters: {total_params:,}")
    print("\nThe model is now ready to be trained!")


# Prints the direct child modules of a model under a title
def list_children(module, title):
    print(f"\n== {title} ==")
    for name, child in module.named_modules():
        # show only immediate children of top-level sequentials
        if name in {"stem", "block", "head"}:
            print(f"\n[{name}]")
            for i, sub in enumerate(child.children()):
                print(f"  {i}: {sub.__class__.__name__}")


# Counts how many fused modules a model contains
def count_fused_layers(model):
    names = []
    for m in model.modules():
        cls = m.__class__.__name__
        if any(k in cls for k in ["ConvReLU2d", "ConvBn2d", "ConvBnReLU2d", "LinearReLU"]):
            names.append(cls)
    return {k: names.count(k) for k in sorted(set(names))}


# Times the forward pass of a model over several iterations on random input
def bench(m, iters=20, shape = (16, 3, 224, 224), device="cpu"):
    torch.manual_seed(17)
    m.eval()
    x = torch.randn(shape).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = m(x)
    return (time.perf_counter() - start) / iters


# Computes receptive field and feature map size for each conv and pool layer
def calculate_receptive_field(model, input_size=224):
    """
    Calculates the receptive field and feature map size for each Conv2d/Pool2d layer in a PyTorch model.

    Args:
        model (nn.Module): A PyTorch nn.Module (for example, ResNet50).
        input_size (int): The spatial size of the square input image (for example, 224 for 224x224).

    Returns:
        list: A list of dictionaries. Each dict contains:
            'layer': Name of the layer
            'rf': Receptive field size up to this layer
            'jump': Effective stride up to this layer
            'map_size': Feature map spatial size (width/height) at this layer
    """
    receptive_field = 1   # Start: Each input pixel only "sees" itself
    jump = 1              # Effective jump/stride (pixels in output to input)
    curr_size = input_size
    results = []

    layer_names = set()   # Helps ensure unique names (if layers repeat)

    def get_layer_name(name, module):
        base = f"{name} ({type(module).__name__})"
        n = 1
        candidate = base
        while candidate in layer_names:
            candidate = f"{base}_{n}"
            n += 1
        layer_names.add(candidate)
        return candidate

    # This function runs for each relevant layer as the model processes a dummy input
    def layer_hook(name):
        def hook(module, inp, outp):
            nonlocal receptive_field, jump, curr_size

            # Get layer parameters (assume square kernels and strides)
            k = getattr(module, "kernel_size", 1)
            s = getattr(module, "stride", 1)
            p = getattr(module, "padding", 0)
            d = getattr(module, "dilation", 1)
            if isinstance(k, tuple): k = k[0]
            if isinstance(s, tuple): s = s[0]
            if isinstance(p, tuple): p = p[0]
            if isinstance(d, tuple): d = d[0]

            # Calculate output spatial size (assume square input/output)
            map_size = outp.shape[-1]

            # Save current layer info
            results.append({
                "layer": get_layer_name(name, module),
                "rf": receptive_field,
                "jump": jump,
                "map_size": curr_size
            })

            # Update receptive field and jump for next layer
            # rf_next = rf + (k-1) * d * jump
            receptive_field = receptive_field + (k - 1) * d * jump
            jump = jump * s
            curr_size = map_size
        return hook

    # Register hooks for all Conv2d and Pool2d layers recursively
    hooks = []
    def register_hooks(net, prefix=''):
        for name, module in net.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                hooks.append(module.register_forward_hook(layer_hook(full_name)))
            register_hooks(module, full_name)
    register_hooks(model)

    # Send a dummy input through the model (to activate hooks)
    dummy = torch.zeros(1, 3, input_size, input_size)
    model.eval()
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    return results


# Visualizes a sample of weights for specified model layers
def show_weights(model, layer_names):
    """
    Visualizes a sample of weights for specified model layers.

    Args:
        model: The neural network model containing the layers.
        layer_names: A string or list of strings representing the layer attributes to inspect.
    """
    # Convert single layer name to a list for uniform processing
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    # Process each requested layer
    for layer_name in layer_names:
        try:
            # Retrieve the layer attribute from the model
            layer = getattr(model, layer_name)

            # Ensure the layer contains a weight parameter
            if not hasattr(layer, 'weight'):
                print(f"Layer '{layer_name}' has no 'weight' attribute.")
                continue

            print(f"--- Weights for '{layer_name}' ---")
            # Detach the weights and convert to a NumPy array
            weights = layer.weight.detach().cpu().numpy()

            # Handle visualization for convolutional layers
            if isinstance(layer, nn.Conv2d):
                if weights.shape[0] >= 2 and weights.shape[1] >= 2:
                    # Concatenate four 3x3 kernels into a 6x6 grid
                    kernel_0_0 = weights[0, 0]
                    kernel_0_1 = weights[0, 1]
                    kernel_1_0 = weights[1, 0]
                    kernel_1_1 = weights[1, 1]
                    grid_6x6 = np.block([[kernel_0_0, kernel_0_1], [kernel_1_0, kernel_1_1]])
                    print(grid_6x6)
                else:
                    # Fallback for layers with small channel dimensions
                    print(weights[0, 0])

            # Handle visualization for linear layers
            elif isinstance(layer, nn.Linear):
                rows, cols = weights.shape
                # Define a slice with a maximum size of 6x6
                slice_rows = min(6, rows)
                slice_cols = min(6, cols)
                # Output the top-left corner of the weight matrix
                print(weights[:slice_rows, :slice_cols])

            else:
                print(f"Visualization for layer type {type(layer)} is not implemented.")

            print("-" * 50)
            print() 

        except AttributeError:
            # Handle cases where the layer name does not exist in the model
            print(f"Layer '{layer_name}' not found in the model.")


# Compute per-layer and global sparsity (fraction of zeros) over Conv2d/Linear weights
def sparsity_report(model):
    """
    Compute per-layer and global sparsity (fraction of zeros) over Conv2d/Linear weights.
    Returns
    -------
    dict
        {
          "layers": { "<name>.weight": 0.52, ... },
          "global_sparsity": 0.47
        }
    """
    layers = {}
    zeros_total = 0
    elems_total = 0

    for name, module in _iter_prunable_modules(model):
        if not hasattr(module, "weight"):
            continue
        w = module.weight.detach()
        z = (w == 0).sum().item()
        n = w.numel()
        layers[f"{name}.weight"] = (z / n) if n > 0 else 0.0
        zeros_total += z
        elems_total += n

    global_sparsity = (zeros_total / elems_total) if elems_total > 0 else 0.0
    return {"layers": layers, "global_sparsity": global_sparsity}


# Yield (qualified_name, module) for Conv2d and Linear layers only
def _iter_prunable_modules(model):
    """Yield (qualified_name, module) for Conv2d and Linear layers only."""
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m


# Downloads a base transformer model and tokenizer from the Hugging Face Hub
def download_bert(model_name="distilbert-base-uncased", local_path="./distilbert-local-base"):
    """
    Downloads a base transformer model and tokenizer from the Hugging Face Hub.

    Args:
        model_name: The name of the model on the Hugging Face Hub.
        local_path: The local directory where the model and tokenizer will be saved.
    """
    # Check if the target directory already exists.
    if os.path.isdir(local_path):
        # If the directory exists, notify the user and skip downloading.
        print(f"Base model '{model_name}' already available at {local_path}")
    else:
        # If the directory does not exist, proceed with the download.
        print(f"Downloading base model '{model_name}' to {local_path}...")
        # Download the pre-trained tokenizer associated with the model.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Download the base pre-trained model without a specific classification head.
        model = AutoModel.from_pretrained(model_name)

        # Save the tokenizer's files to the specified local path.
        tokenizer.save_pretrained(local_path)
        # Save the model's files to the specified local path.
        model.save_pretrained(local_path)
        # Confirm that the model has been downloaded and saved.
        print("Base model downloaded and saved successfully.")


# Loads a base transformer from disk and adds a new classification head
def load_bert(local_path="./distilbert-local-base", num_classes=2):
    """
    Loads a base transformer model from a local directory and adds a new
    sequence classification head.

    Args:
        local_path: The local directory where the base model files are stored.
        num_classes: The number of output classes for the new classification head.

    Returns:
        A tuple containing the loaded model (with the new head) and the tokenizer.
    """
    # Announce the model loading and head configuration process.
    print(
        f"Loading base model from {local_path} and adding a new head with {num_classes} classes.\n"
    )

    # Load the pre-trained tokenizer from the specified local directory.
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    # Load the base pre-trained model and add a new, untrained sequence
    # classification head on top with the specified number of labels.
    model = AutoModelForSequenceClassification.from_pretrained(
        local_path, num_labels=num_classes
    )

    # Confirm that the model and tokenizer have been successfully loaded.
    print("\nModel and tokenizer loaded successfully.")

    # Return the complete model and its tokenizer.
    return model, tokenizer


# Loads a plain BERT model and its tokenizer from a local folder
def load_bert_base_and_tokenizer(save_directory='./bert_model'):
    """
    Loads a pre-trained BERT model and its tokenizer from a local directory.

    Args:
        save_directory: The path to the directory where the model and
                        tokenizer files are stored.

    Returns:
        A tuple containing the loaded tokenizer and model objects.
    """
    print(f"Loading BERT model from '{save_directory}'...")
    # Load the tokenizer and model from the specified local path.
    tokenizer = BertTokenizer.from_pretrained(save_directory)
    model = BertModel.from_pretrained(save_directory)
    print("Model loaded successfully.")
    
    return tokenizer, model


# Ensures the BLIP VQA model is available locally and loads the model and processor
def get_blip_vqa_model_and_processor():
    """
    Ensures the BLIP VQA model is available locally and loads the model and processor.

    Args:

    Returns:
        model (BlipForQuestionAnswering): The loaded BLIP model for question answering tasks.
        processor (BlipProcessor): The associated processor for image and text inputs.
    """
    # Define the model identifier from the Hugging Face Hub
    model_id = "Salesforce/blip-vqa-base"
    
    # Define the local directory where the model will be stored and loaded from
    local_path = "./blip-vqa-base-local"

    # Check if the model directory does not exist on the local filesystem
    if not os.path.exists(local_path):
        # Notify that the model is being downloaded to the local path
        print(f"Local model not found. Downloading '{model_id}' to local path: '{local_path}'")
        
        # Create the target directory and handle existing directory cases
        os.makedirs(local_path, exist_ok=True)
        
        try:
            # Download the pre-trained model weights from the hub
            temp_model = BlipForQuestionAnswering.from_pretrained(model_id)
            
            # Download the processor using the fast tokenizer implementation
            temp_processor = BlipProcessor.from_pretrained(model_id, use_fast=True)
            
            # Save the downloaded model weights to the local directory
            temp_model.save_pretrained(local_path)
            
            # Save the processor configuration and files to the local directory
            temp_processor.save_pretrained(local_path)
            
            # Confirm successful download and save operation
            print("Model and processor downloaded and saved successfully.")
        except Exception as e:
            # Handle potential network or filesystem errors during the download process
            print(f"An error occurred during download: {e}")
            return None, None
    else:
        # Confirm that the local model directory was found
        print(f"Model already exists at local path: '{local_path}'.")

    try:
        # Indicate that the loading process from disk has started
        print(f"Loading model and processor from local path: {local_path}")
        
        # Load the model weights and configuration from the local filesystem
        model = BlipForQuestionAnswering.from_pretrained(local_path)
        
        # Load the processor and its tokenizer from the local filesystem
        processor = BlipProcessor.from_pretrained(local_path, use_fast=True)
        
        # Confirm that the objects are successfully initialized in memory
        print("Model and processor loaded successfully.")
        
        # Return the initialized model and processor
        return model, processor
    except Exception as e:
        # Handle errors during the loading process from local storage
        print(f"Failed to load model from local path: {e}")
        return None, None


# Saves the student's trained model and metadata to a file
def save_student_model(model, filename="trained_student_model.pth"):
    """
    Saves the student's trained model and metadata to a file.

    Args:
        model (torch.nn.Module): The student's trained model.
        filename (str): The filename to save to.
    """
    # Create a dictionary containing the model state
    save_dict = {"model": model}
    
    # Save the dictionary to the specified file path
    torch.save(save_dict, filename)
    
    # Print a confirmation message indicating the save location
    print(f"Model saved to {filename}")


# Serializes and saves a results object to a file using pickle
def save_training_logs(partial_results, filename="training_logs.pkl"):
    """
    Serializes and saves a results object to a file using pickle.

    Args:
        partial_results: The Python object to be saved.
        filename: The path and name of the file to save the object to.
    """
    # Open the specified file in write-binary mode.
    with open(filename, 'wb') as f:
        # Use pickle to serialize and write the object to the file.
        pickle.dump(partial_results, f)
    
    # Print a confirmation message indicating where the logs were saved.
    print(f"Training logs saved to '{filename}'")


# Saves a model state dict and its metrics dict to disk
def save_model_and_metrics(
    model, 
    metrics, 
    state_dict_filename="model_state_dict.pth", 
    metrics_filename="metrics.pkl"
):
    """
    Persists the model state dictionary and evaluation metrics to local storage.

    Args:
        model: The trained neural network model to be saved.
        metrics: A dictionary containing performance evaluation data.
        state_dict_filename: The target filename for the saved model weights.
        metrics_filename: The target filename for the serialized metrics data.
    """
    # Save the model weights and parameters to a file
    torch.save(model.state_dict(), state_dict_filename)
    print(f"Model state dictionary saved to {state_dict_filename}")

    # Open a file stream to serialize the metrics dictionary
    with open(metrics_filename, 'wb') as f:
        # Write the metrics data using pickle serialization
        pickle.dump(metrics, f)
    print(f"Model metrics saved to {metrics_filename}")


# Builds a checkpoint dict with epoch, model state, optimizer state and loss
def make_checkpoint(epoch, model, optimizer, loss, extra=None):
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": float(loss),
    }
    if extra:
        ckpt.update(extra)
    return ckpt


# Builds a ResNet50 with a 2-class head and loads saved weights
def load_model(model_path, num_classes=2, arch='resnet50', device='cpu', strict=False):
    """
    Builds a torchvision classifier with a new output layer and loads saved weights into it.

    Args:
        model_path: Path to a saved state dict.
        num_classes: Number of outputs of the final layer.
        arch: Name of a torchvision.models constructor, for example 'resnet50'.
        device: Device used to map the saved weights.
        strict: Passed to load_state_dict.

    Returns:
        The model with the weights loaded.
    """
    model = getattr(tv_models, arch)(weights=None)
    if hasattr(model, 'fc'):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Don't know how to replace the final layer of '{arch}'.")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=strict)
    return model



# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================


# Runs one training epoch and returns the average loss
def training_epoch(model, train_loader, optimizer, loss_function, device, epoch, num_epochs, emty_cache=True, silent=False,):
    """Performs a single training epoch for a given model.

    Args:
        model: The neural network model to be trained.
        train_loader: The DataLoader for the training data.
        optimizer: The optimization algorithm to update model weights.
        loss_function: The loss function used to evaluate model performance.
        device: The device (e.g., 'cuda' or 'cpu') to run the training on.
        epoch: The current epoch number.
        num_epochs: The total number of epochs for training.
        emty_cache: A boolean flag to empty the CUDA cache after each batch.
        silent: A boolean flag to disable the progress bar and print statements.
    
    Returns:
        The average loss for the epoch.
    """
    # Set the model to training mode
    model.train()

    # Initialize the total loss for the epoch
    running_loss = 0.0
    # Create a TQDM progress bar for the training loader
    train_loader_tqdm = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{num_epochs} - Training",
        leave=False,
        disable=silent,
    )

    # Iterate over the batches of data in the training loader
    for i, (images, labels) in enumerate(train_loader_tqdm):
        # Move images and labels to the specified device
        images = images.to(device)
        labels = labels.to(device)

        # Perform a forward pass to get the model's predictions
        outputs = model(images)
        # Calculate the loss between the predictions and true labels
        loss = loss_function(outputs, labels)

        # Clear the gradients from the previous iteration
        optimizer.zero_grad()
        # Perform backpropagation to compute gradients
        loss.backward()
        # Update the model's weights using the optimizer
        optimizer.step()

        # Add the current batch's loss to the running total
        running_loss += loss.item()

        # Update the progress bar with the current loss if not in silent mode
        if not silent:
            train_loader_tqdm.set_postfix(loss=loss.item(), refresh=False)

        # Print training progress at specified intervals if not in silent mode
        if ((i + 1) % 45 == 0) & (not silent):
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

        # Optionally, clear memory to prevent out-of-memory errors
        if emty_cache:
            del images, labels, outputs, loss
            torch.cuda.empty_cache()

    # Calculate the average loss for the entire epoch
    epoch_loss = running_loss / len(train_loader)
    # Return the calculated average epoch loss
    return epoch_loss


# Evaluates a model on a validation loader and returns loss and accuracy
def evaluate_model(model, val_loader, device, silent=False):
    """
    Evaluates the performance of a model on a validation dataset.

    Args:
        model: The model to be evaluated.
        val_loader: DataLoader for the validation dataset.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.
        silent: If True, suppresses the printing of the validation accuracy.

    Returns:
        The accuracy of the model on the validation dataset.
    """
    # Set the model to evaluation mode
    model.eval()
    # Initialize variables for tracking correct predictions and total samples
    correct, total = 0, 0
    # Disable gradient calculations for inference
    with torch.no_grad():
        # Iterate over the validation data
        for inputs, labels in val_loader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            # Perform a forward pass
            outputs = model(inputs)
            # Get the predicted class with the highest probability
            _, predicted = torch.max(outputs, 1)
            # Update the total number of samples
            total += labels.size(0)
            # Update the count of correctly predicted samples
            correct += (predicted == labels).sum().item()

            # Release memory to prevent memory leaks
            del inputs, labels, outputs
            # Clear the GPU cache
            torch.cuda.empty_cache()

    # Calculate the accuracy
    accuracy = correct / total
    # Check if printing the accuracy is enabled
    if not silent:
        # Print the validation accuracy
        print(f"Validation Accuracy: {100 * accuracy:.2f}%")

    # Return the calculated accuracy
    return accuracy


# Returns the classification accuracy of a model over a data loader
def compute_accuracy(model, loader, device):
    from tqdm.notebook import tqdm
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # Create progress bar for evaluation
        pbar = tqdm(loader, desc='Computing Accuracy', leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            # Update progress bar with current accuracy
            acc = correct / max(total, 1)
            pbar.set_description(f'Accuracy: {acc:.4f}')
    return correct / max(total, 1)


# Stores the given parameters as user attributes on an Optuna trial
def extract_attr(trial, transform, model, params):
    """
    Sets user-defined attributes on an Optuna trial.

    Args:
        trial: The Optuna trial object.
        transform: The data transformation to be stored.
        model: The machine learning model to be stored.
        params: The parameters of the model or experiment.
    """
    # Set a user attribute for the data transformation
    trial.set_user_attr("transform", transform)
    # Set a user attribute for the model
    trial.set_user_attr("model", model)
    # Set a user attribute for the parameters
    trial.set_user_attr("params_code", params)


# Executes a given experiment function, handling result caching and re-runs
def run_experiment(experiment_name, experiment_fcn, cases, trainset, device, rerun=False):
    """
    Executes a given experiment function, handling result caching and re-runs.

    Args:
        experiment_name (str): The name of the experiment.
        experiment_fcn (function): The function to be executed for the experiment.
        cases (any): A collection of cases or parameters for the experiment.
        trainset (any): The training dataset.
        device (any): The device to be used for computation.
        rerun (bool): If True, the experiment runs regardless of existing
                      checkpoints, overwriting previous results. Defaults to False.

    Returns:
        dict: The result of the experiment.
    """
    # Define the directory for storing experiment checkpoints
    folder_experiments = "./checkpoint_experiments"
    # Ensure the checkpoint directory exists
    os.makedirs(folder_experiments, exist_ok=True)
    file_path = f"{folder_experiments}/{experiment_name}.json"

    def run_and_save():
        """Executes the experiment, saves the results, and returns them."""
        print(f"Executing experiment '{experiment_name}'...")
        result_experiment = experiment_fcn(cases, trainset, device)
        with open(file_path, "w") as f:
            # Save the experiment results to the file in JSON format
            json.dump(result_experiment, f, indent=4)
        print(f"Results for '{experiment_name}' saved to {file_path}")
        return result_experiment

    # If 'rerun' is True, run the experiment regardless of existing checkpoints
    if rerun:
        print(f"Flag 'rerun' is True. Executing '{experiment_name}' and overwriting any existing results.")
        return run_and_save()

    # If 'rerun' is False, check for a valid checkpoint first
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                result_experiment = json.load(f)
            
            # Preserve original functionality: re-run if results contain infinity
            if any(isinstance(val, float) and val == float("inf") for val in result_experiment.values()):
                print(f"Invalid (infinite) values found in checkpoint for '{experiment_name}'. Re-running.")
                return run_and_save()
            
            # Improved print statement for loading from a valid checkpoint
            print(
                f"Experiment '{experiment_name}' has already been completed. "
                f"Loading results from checkpoint: {file_path}"
            )
            return result_experiment
        except (json.JSONDecodeError, IOError) as e:
            # Handle cases where the file is corrupted or unreadable
            print(f"Could not load checkpoint file {file_path} due to error: {e}. Re-running experiment.")
            return run_and_save()
    else:
        # If no checkpoint exists, run the experiment for the first time
        print(f"Results for experiment '{experiment_name}' not found. Running experiment.")
        return run_and_save()


# Trains a classifier with an optimizer and optional scheduler and returns the best model
def training_loop(model, train_loader, validation_loader, loss_fcn, optimizer, scheduler, device, n_epochs):
    """
    Executes a training and validation loop, returning the best model.

    This function iterates for a specified number of epochs, running one
    training phase and one validation phase per epoch. It tracks the best
    validation accuracy and returns the model state from that epoch.

    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader for the training dataset.
        validation_loader: DataLoader for the validation dataset.
        loss_fcn: The loss function to use for training.
        optimizer: The optimization algorithm.
        scheduler: An optional learning rate scheduler.
        device: The device (e.g., 'cuda' or 'cpu') to run the training on.
        n_epochs: The total number of epochs to train.
    
    Returns:
        The trained model (nn.Module) with the weights from the epoch
        that achieved the highest validation accuracy.
    """
    
    # --- Setup ---
    # Move model to the target device
    model.to(device)
    
    # Get num_classes from the dataset (common in ImageFolder)
    num_classes = len(train_loader.dataset.classes)
    # Initialize metric calculators
    train_acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    val_acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    
    # --- Best Model Tracking ---
    # Initialize the best validation accuracy found so far
    best_val_acc = 0.0
    # Initialize a tracker for the epoch number of the best model
    best_epoch = 0
    # Variable to store the state dictionary of the best model
    best_model_state = None
    # --- End Tracking ---
    
    print("--- Starting Training ---")
    
    # --- Main Epoch Loop ---
    # Iterate over the specified number of epochs
    for epoch in range(n_epochs):
        
        # --- Training Phase ---
        # Set the model to training mode
        model.train()
        # Initialize running loss for the current epoch
        running_train_loss = 0.0
        # Reset the training accuracy metric at the start of each epoch
        train_acc_metric.reset()
        
        # Create a TQDM progress bar for the training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [TRAIN]", leave=False)
        
        # Iterate over batches of training data
        for inputs, labels in train_pbar:
            # Move data to the device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Zero gradients
            # Clear previously calculated gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            # Get model predictions for the current batch
            outputs = model(inputs)
            
            # 3. Calculate loss
            # Compute the loss between predictions and actual labels
            loss = loss_fcn(outputs, labels)
            
            # 4. Backward pass
            # Compute gradients of the loss with respect to model parameters
            loss.backward()
            
            # 5. Optimizer step
            # Update model parameters based on the computed gradients
            optimizer.step()
            
            # --- Update Metrics ---
            # Accumulate the training loss
            running_train_loss += loss.item() * inputs.size(0)
            # Get the predicted class indices
            preds = torch.argmax(outputs, dim=1)
            # Update the accuracy metric
            train_acc_metric.update(preds, labels)
            
            # Update progress bar postfix
            train_pbar.set_postfix(batch_loss=loss.item())

        # Calculate average training loss and accuracy for the epoch
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = train_acc_metric.compute().item()
        
        
        # --- Validation Phase ---
        # Set the model to evaluation mode
        model.eval()
        # Initialize running validation loss
        running_val_loss = 0.0
        # Reset the validation accuracy metric
        val_acc_metric.reset()
        
        # Create a TQDM progress bar for the validation batches
        val_pbar = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{n_epochs} [VAL]", leave=False)
        
        # Disable gradient calculation for validation
        with torch.no_grad():
            # Iterate over batches of validation data
            for inputs, labels in val_pbar:
                # Move data to the device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 1. Forward pass
                # Get model predictions
                outputs = model(inputs)
                
                # 2. Calculate loss
                # Compute the validation loss
                loss = loss_fcn(outputs, labels)
                
                # --- Update Metrics ---
                # Accumulate the validation loss
                running_val_loss += loss.item() * inputs.size(0)
                # Get the predicted class indices
                preds = torch.argmax(outputs, dim=1)
                # Update the validation accuracy metric
                val_acc_metric.update(preds, labels)
                
                # Update progress bar postfix
                val_pbar.set_postfix(batch_loss=loss.item())

        # Calculate average validation loss and accuracy for the epoch
        epoch_val_loss = running_val_loss / len(validation_loader.dataset)
        epoch_val_acc = val_acc_metric.compute().item()
        

        # --- End of Epoch ---
        
        # Step the scheduler
        if scheduler is not None:
            # Check if it's ReduceLROnPlateau, which needs a metric
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                # Step the scheduler based on validation loss
                scheduler.step(epoch_val_loss)
            else:
                # Step other types of schedulers
                scheduler.step()
        
        # Log metrics for the epoch
        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # --- Check for Best Model ---
        # If current validation accuracy is better than the best seen so far
        if epoch_val_acc > best_val_acc:
            # Update the best validation accuracy
            best_val_acc = epoch_val_acc
            # Store the current epoch number
            best_epoch = epoch + 1
            # Use copy.deepcopy to save a snapshot of the model's state
            best_model_state = copy.deepcopy(model.state_dict())
            # Print a message indicating a new best model
            print(f"    ^ New best model found!")
        # --- End Check ---

    print("\n--- Training Complete ---")
    
    # --- Load Best Model ---
    # Check if a best model state was saved
    if best_model_state is not None:
        print(f"\nReturning best model from epoch {best_epoch} with {best_val_acc:.4f} validation accuracy.")
        # Load the best performing weights back into the model
        model.load_state_dict(best_model_state)
    else:
        # Warn if no improvement was seen and the last model is being returned
        print("\nWarning: No best model state was saved (e.g., validation never improved). Returning last model.")
    
    # Return the best model
    return model


# Trains and validates a model with 16-bit mixed precision
def training_loop_16_mixed(
    model, train_loader, val_loader, loss_function, optimizer, num_epochs, device, scheduler=None, save_path=None,
):
    """Executes a training and validation loop for a PyTorch model using 16-bit mixed precision.

    This function iterates through epochs, performing a training step and a
    validation step for each. It tracks performance metrics like loss and
    accuracy, and optionally saves the model with the highest validation accuracy.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): The DataLoader for validation data.
        loss_function (callable): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        num_epochs (int): The total number of training epochs.
        device (torch.device): The device to perform training on (e.g., 'cuda', 'cpu').
        scheduler (torch.optim.lr_scheduler, optional): A learning rate scheduler. Defaults to None.
        save_path (str, optional): A file path to save the best model weights. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The model with the weights that achieved the best validation accuracy.
            - history (dict): A dictionary containing training and validation loss and validation accuracy for each epoch.
            - best_cm (numpy.ndarray): The confusion matrix from the best validation epoch.
    """
    # Determine the device type as a string for autocast compatibility.
    if device == torch.device("mps"):
        device_str = "mps"
    elif device == torch.device("cuda"):
        device_str = "cuda"
    else:
        device_str = "cpu"

    # Initialize a gradient scaler for mixed-precision training.
    # Gradient scaling is not supported on MPS, so it's conditionally enabled.
    use_scaler = device_str != "mps"
    scaler = GradScaler() if use_scaler else None

    # Create the directory to save the model if a path is provided.
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # Move the model to the specified computation device.
    model.to(device)

    # Initialize variables to track the best validation accuracy and confusion matrix.
    best_val_acc = 0.0
    best_cm = None

    # A dictionary to store the history of training and validation metrics.
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Determine the number of classes from the dataset.
    num_classes = len(train_loader.dataset.classes)
    # Initialize torchmetrics for calculating accuracy and the confusion matrix.
    val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro").to(
        device
    )
    val_cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    # Set up a single progress bar for the entire training process.
    total_steps = (len(train_loader) + len(val_loader)) * num_epochs
    pbar = tqdm(total=total_steps, desc="Overall Progress")

    # Begin the main training loop over the specified number of epochs.
    for epoch in range(num_epochs):
        # --- Training Phase ---
        # Set the model to training mode.
        model.train()
        # Initialize variables to accumulate training loss for the current epoch.
        running_train_loss = 0.0
        train_samples_processed = 0

        # Iterate over the training data loader.
        for inputs, labels in train_loader:
            # Update the progress bar description for the current phase.
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Training]")
            # Move input data and labels to the designated device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear any previously calculated gradients.
            optimizer.zero_grad(set_to_none=True)

            # Use autocast for mixed-precision forward pass.
            with autocast(device_type=device_str, dtype=torch.float16):
                # Forward pass: compute predicted outputs by passing inputs to the model.
                outputs = model(inputs)
                # Calculate the loss.
                loss = loss_function(outputs, labels)

            # If using a scaler (e.g., on CUDA), scale the loss and perform a backward pass.
            if use_scaler:
                scaler.scale(loss).backward()
                # Update the model weights using the scaled gradients.
                scaler.step(optimizer)
                # Update the scaler for the next iteration.
                scaler.update()
            # If not using a scaler (e.g., on CPU/MPS), perform a standard backward pass.
            else:
                loss.backward()
                # Update the model weights.
                optimizer.step()

            # Get the size of the current batch.
            batch_size = inputs.size(0)
            # Accumulate the loss, weighted by the batch size.
            running_train_loss += loss.item() * batch_size
            # Update the count of processed samples.
            train_samples_processed += batch_size
            # Calculate and display the running average loss.
            display_loss = running_train_loss / train_samples_processed
            pbar.set_postfix(loss=f"{display_loss:.4f}")
            # Update the progress bar for the batch.
            pbar.update(1)

        # Calculate the average training loss for the epoch.
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        # Store the epoch's training loss in the history.
        history["train_loss"].append(epoch_train_loss)

        # --- Validation Phase ---
        # Set the model to evaluation mode.
        model.eval()
        # Initialize variables to accumulate validation loss.
        running_val_loss = 0.0
        val_samples_processed = 0
        # Reset metric calculators for the new validation epoch.
        val_accuracy.reset()
        val_cm.reset()

        # Disable gradient calculations for the validation phase.
        with torch.no_grad():
            # Iterate over the validation data loader.
            for inputs, labels in val_loader:
                # Update the progress bar description for the validation phase.
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Validation]")
                # Move input data and labels to the designated device.
                inputs, labels = inputs.to(device), labels.to(device)

                # Use autocast for mixed-precision forward pass during validation.
                with autocast(device_type=device_str, dtype=torch.float16):
                    # Compute model outputs.
                    outputs = model(inputs)
                    # Calculate the validation loss.
                    loss = loss_function(outputs, labels)

                # Get the predicted class indices by finding the max logit.
                preds = outputs.argmax(dim=1)
                # Get the size of the current batch.
                batch_size = inputs.size(0)
                # Accumulate the validation loss.
                running_val_loss += loss.item() * batch_size
                # Update the count of processed validation samples.
                val_samples_processed += batch_size
                # Update metrics with the current batch's predictions and labels.
                val_accuracy.update(preds, labels)
                val_cm.update(preds, labels)

                # Compute and display the current running validation accuracy and loss.
                current_acc = val_accuracy.compute().item()
                display_loss = running_val_loss / val_samples_processed
                pbar.set_postfix(
                    acc=f"{current_acc:.2%}",
                    loss=f"{display_loss:.4f}",
                )
                # Update the progress bar.
                pbar.update(1)

        # Calculate the average validation loss for the epoch.
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        # Compute the final validation accuracy for the epoch.
        epoch_val_acc = val_accuracy.compute().item()
        # Store the epoch's validation loss and accuracy in the history.
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        # Print the summary of the epoch's performance.
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

        # --- SCHEDULER AND SAVE BEST MODEL ---
        # Adjust the learning rate based on the scheduler's logic, if one is provided.
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_acc)
            else:
                scheduler.step()

        # Check if the current model has the best validation accuracy seen so far.
        if epoch_val_acc > best_val_acc:
            # Update the best validation accuracy.
            best_val_acc = epoch_val_acc
            # Store the confusion matrix from the best performing epoch.
            best_cm = val_cm.compute().cpu().numpy()
            # Save the model's state dictionary if a save path is specified.
            if save_path:
                torch.save(model.state_dict(), save_path)
                # Print a message indicating that a new best model has been saved.
                tqdm.write(
                    f"  -> New best model saved to '{save_path}' with Val Acc: {best_val_acc:.2%}\n"
                )

    # Close the progress bar after the training loop is complete.
    pbar.close()

    # If a save path was provided, load the weights of the best performing model.
    if save_path and os.path.exists(save_path):
        tqdm.write(f"\nBest model saved to '{save_path}' with accuracy {best_val_acc:.2%}")
        model.load_state_dict(torch.load(save_path))
    
    # Return the trained model, the history of metrics, and the best confusion matrix.
    return model, history, best_cm


# Trains a Hugging Face style text classifier on dict batches and returns the model and metrics
def training_loop_transformer_classifier(model, train_loader, val_loader, loss_function, learning_rate, num_epochs, device):
    """
    Performs a full training and validation cycle for a PyTorch model.

    Args:
        model: The PyTorch model to be trained.
        train_loader: The DataLoader for the training dataset.
        val_loader: The DataLoader for the validation dataset.
        loss_function: The loss function used for training.
        learning_rate: The learning rate for the optimizer.
        num_epochs: The total number of epochs to train for.
        device: The computational device ('cuda' or 'cpu') to run on.

    Returns:
        A tuple containing the trained model and a dictionary of the final
        performance metrics from the last validation epoch.
    """
    # Move the model to the specified computational device.
    model.to(device)

    # Initialize the AdamW optimizer with the specified learning rate and weight decay.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Determine the number of classes from the model's configuration.
    num_classes = model.config.num_labels
    
    # Initialize metric objects from torchmetrics for stateful metric calculation.
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    
    # Create the main progress bar that iterates over the epochs.
    epoch_loop = tqdm(range(num_epochs), desc="Training Progress")

    # Begin the main training and validation loop.
    for epoch in epoch_loop:
        
        # --- Training Phase ---
        # Set the model to training mode, which enables layers like dropout.
        model.train()
        # Initialize the accumulated training loss for the epoch.
        train_loss_epoch = 0
        
        # Create a nested progress bar for the training batches of the current epoch.
        train_inner_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        # Iterate over the training data batches.
        for batch in train_inner_loop:
            # Unpack the batch and move all tensors to the active device.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Clear any gradients from the previous iteration.
            optimizer.zero_grad()
            
            # Perform a forward pass to get the model's raw outputs (logits).
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate the loss for the current batch.
            loss = loss_function(logits, labels)
            
            # Accumulate the training loss and perform backpropagation to compute gradients.
            train_loss_epoch += loss.item()
            loss.backward()
            
            # Update the model's weights based on the computed gradients.
            optimizer.step()
            
            # Update the inner progress bar's postfix with the current batch loss.
            train_inner_loop.set_postfix(loss=loss.item())

        # Calculate the average training loss over all batches in the epoch.
        train_loss_epoch /= len(train_loader)

        # --- Validation Phase ---
        # Set the model to evaluation mode, which disables layers like dropout.
        model.eval()
        # Initialize the accumulated validation loss for the epoch.
        val_loss_epoch = 0
        
        # Create a nested progress bar for the validation batches.
        val_inner_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False)
        # Disable gradient calculations to save memory and computations.
        with torch.no_grad():
            # Iterate over the validation data batches.
            for batch in val_inner_loop:
                # Unpack the batch and move tensors to the active device.
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Perform a forward pass to get the model's logits.
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Calculate and accumulate validation loss for the batch.
                val_loss = loss_function(logits, labels)
                val_loss_epoch += val_loss.item()
                
                # Get model predictions and update all metric objects with batch results.
                preds = torch.argmax(logits, dim=-1)
                val_accuracy.update(preds, labels)
                val_f1.update(preds, labels)
                val_cm.update(preds, labels)
        
        # Calculate the average validation loss for the epoch.
        val_loss_epoch /= len(val_loader)
        
        # --- Logging and Metric Calculation ---
        # Compute the final metrics over the entire validation set for the epoch.
        epoch_acc = val_accuracy.compute()
        epoch_f1 = val_f1.compute()

        # Reset the metric calculators for the next epoch.
        val_accuracy.reset()
        val_f1.reset()

        # Update the main progress bar with the results of the completed epoch.
        epoch_loop.set_postfix(
            train_loss=f"{train_loss_epoch:.4f}",
            val_loss=f"{val_loss_epoch:.4f}",
            val_acc=f"{epoch_acc:.4f}"
        )
        # Use tqdm.write to log metrics without interfering with the progress bars.
        tqdm.write(f"Epoch {epoch+1} Metrics -> Val Loss: {val_loss_epoch:.4f}, Val Acc: {epoch_acc:.4f}, Val F1: {epoch_f1:.4f}")

    # Indicate that the entire training process is complete.
    print("\n--- Training complete ---")

    # Compute the final confusion matrix over the entire validation dataset.
    final_confusion_matrix = val_cm.compute()

    # Store the final metrics from the last epoch in a results dictionary.
    final_results = {
        "val_loss": val_loss_epoch,
        "val_accuracy": epoch_acc.item(),
        "val_f1": epoch_f1.item(),
        "confusion_matrix": final_confusion_matrix.cpu()
    }
    
    # Return the trained model and the final results.
    return model, final_results


# Trains a text classifier on (text, offsets, labels) batches and returns the model and metrics
def training_loop_embedding_bag(model, train_loader, val_loader, loss_function, num_epochs, device):
    """
    Handles the complete training and validation loop for a PyTorch model.

    This function trains the model on the training set and evaluates it on the
    validation set for a specified number of epochs. It is designed to work
    with different model architectures by checking the model's class name to
    correctly unpack data batches. It calculates and stores key performance
    metrics for each epoch.

    Args:
        model: The PyTorch model to be trained.
        train_loader: The DataLoader for the training dataset.
        val_loader: The DataLoader for the validation dataset.
        loss_function: The loss function (e.g., nn.CrossEntropyLoss).
        num_epochs: The total number of epochs for training.
        device: The device to train on ('cuda' or 'cpu').

    Returns:
        A tuple containing the trained model and a dictionary of the final
        performance metrics from the last epoch, including validation
        accuracy, precision, recall, and F1-score.
    """
    # Move the model to the specified device (GPU or CPU).
    model.to(device)

    # Initialize the Adam optimizer with a default learning rate.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Automatically determine the number of classes from the training dataset.
    num_classes = len(train_loader.dataset.classes)

    # Specify the averaging method for multi-class metrics.
    avg_method = "macro"

    # Initialize torchmetrics objects for calculating performance metrics.
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average=avg_method).to(device)
    val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average=avg_method).to(device)
    val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average=avg_method).to(device)
    
    # Print the name of the model being trained.
    print(f"--- Training {model.__class__.__name__} ---")

    # Begin the main training loop over the specified number of epochs.
    for epoch in tqdm(range(num_epochs)):

        # Set the model to training mode.
        model.train()
        # Initialize the accumulated training loss for the epoch.
        train_loss_epoch = 0
        # Iterate over batches in the training data loader.
        for batch in train_loader:
            # Clear the gradients from the previous iteration.
            optimizer.zero_grad()

            # Handle different data unpacking based on the model type.
            if model.__class__.__name__ == "EmbeddingBagClassifier":
                # Unpack text, offsets, and labels for EmbeddingBag models.
                text, offsets, labels = batch
                # Move tensors to the specified device.
                text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)
                # Perform a forward pass.
                outputs = model(text, offsets)
            else:
                # Unpack text and labels for other model types.
                text, labels = batch
                # Move tensors to the specified device.
                text, labels = text.to(device), labels.to(device)
                # Perform a forward pass.
                outputs = model(text)

            # Calculate the loss between model outputs and true labels.
            loss = loss_function(outputs, labels)
            # Add the current batch's loss to the epoch's total.
            train_loss_epoch += loss.item()
            # Perform backpropagation to compute gradients.
            loss.backward()
            # Update the model's weights using the optimizer.
            optimizer.step()

        # Calculate the average training loss for the epoch.
        train_loss_epoch /= len(train_loader)

        # Set the model to evaluation mode.
        model.eval()
        # Initialize the accumulated validation loss for the epoch.
        val_loss_epoch = 0
        # Disable gradient calculations for the validation phase.
        with torch.no_grad():
            # Iterate over batches in the validation data loader.
            for batch in val_loader:
                # Handle different data unpacking based on the model type.
                if model.__class__.__name__ == "EmbeddingBagClassifier":
                    # Unpack validation text, offsets, and labels.
                    text, offsets, labels = batch
                    # Move validation tensors to the specified device.
                    text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)
                    # Perform a forward pass on validation data.
                    val_outputs = model(text, offsets)
                else:
                    # Unpack validation text and labels.
                    text, labels = batch
                    # Move validation tensors to the specified device.
                    text, labels = text.to(device), labels.to(device)
                    # Perform a forward pass on validation data.
                    val_outputs = model(text)

                # Add the batch validation loss to the epoch's total.
                val_loss_epoch += loss_function(val_outputs, labels).item()
                # Update validation metrics with the current batch's results.
                val_accuracy.update(val_outputs, labels)
                val_precision.update(val_outputs, labels)
                val_recall.update(val_outputs, labels)
                val_f1.update(val_outputs, labels)

        # Calculate the average validation loss for the epoch.
        val_loss_epoch /= len(val_loader)

        # Compute the final validation metrics for the epoch.
        epoch_acc = val_accuracy.compute()
        epoch_prec = val_precision.compute()
        epoch_recall = val_recall.compute()
        epoch_f1 = val_f1.compute()

        # Reset the metric calculators for the next epoch.
        val_accuracy.reset()
        val_precision.reset()
        val_recall.reset()
        val_f1.reset()

        # Print the training and validation progress at specified intervals.
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss_epoch:.4f} | "
                f"Val Loss: {val_loss_epoch:.4f} | "
                f"Val Accuracy: {epoch_acc:.4f}"
            )

    # Indicate that the training process has finished.
    print("--- Training complete ---")

    # Store the final performance metrics in a dictionary.
    final_results = {
        "val_accuracy": epoch_acc.item(),
        "val_precision": epoch_prec.item(),
        "val_recall": epoch_recall.item(),
        "val_f1": epoch_f1.item(),
    }

    # Return the trained model and the final metrics.
    return model, final_results


# Trains a classifier with Adam and a plateau scheduler and returns the model and final metrics
def training_loop_with_metrics(model, train_loader, val_loader, num_epochs, device, num_classes):
    """
    Executes a standard training and validation routine for a neural network.

    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader providing the training dataset.
        val_loader: DataLoader providing the validation dataset.
        num_epochs: Total number of iterations over the entire dataset.
        device: The processing unit (CPU or GPU) to use for computation.
        num_classes: The number of distinct categories in the dataset.
    """
    # Initialize the cross-entropy loss function for classification
    loss_function = nn.CrossEntropyLoss()
    
    # Configure the Adam optimizer
    # Update all parameters with a smaller learning rate suitable for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Set up a scheduler to decrease the learning rate when validation loss stops improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Initialize evaluation metrics for multiclass classification on the target device
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision_metric = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    recall_metric = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_score_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    
    # Transfer the model architecture to the specified hardware
    model = model.to(device)

    # Log the commencement of the training session
    print(f"--- Starting Training ---")
    
    # Iterate through the specified number of training cycles
    for epoch in range(num_epochs):
        
        # Enable gradients and specific behaviors for training
        model.train()
        # Track the total loss accumulated during the training phase
        train_loss = 0.0
        # Initialize a visual progress tracker for training batches
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        # Process each batch of training data
        for inputs, labels in train_progress_bar:
            # Move batch tensors to the active computation device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset gradient buffers from the previous iteration
            optimizer.zero_grad()
            
            # Compute model predictions based on current weights
            outputs = model(inputs)
            # Determine the error between predictions and ground truth
            loss = loss_function(outputs, labels)
            
            # Backpropagate the error to calculate parameter gradients
            loss.backward()
            # Adjust model parameters based on the calculated gradients
            optimizer.step()
            
            # Update the total training loss for this epoch
            train_loss += loss.item()
            # Display the current average loss in the progress bar
            train_progress_bar.set_postfix(loss=f"{(train_loss / (train_progress_bar.n + 1)):.4f}")
            
        # Disable gradient computation and enable evaluation behaviors
        model.eval()
        # Track the total loss accumulated during the validation phase
        val_loss = 0.0
        
        # Initialize a visual progress tracker for validation batches
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        # Ensure no weight updates occur during the validation phase
        with torch.no_grad():
            # Process each batch of validation data
            for inputs, labels in val_progress_bar:
                # Move batch tensors to the active computation device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Generate model predictions for validation samples
                outputs = model(inputs)
                # Compute the validation error
                loss = loss_function(outputs, labels)
                
                # Update the total validation loss for this epoch
                val_loss += loss.item()
                # Determine the class with the highest probability
                _, predicted = torch.max(outputs.data, 1)
                
                # Accumulate statistics for each performance metric
                accuracy_metric.update(predicted, labels)
                precision_metric.update(predicted, labels)
                recall_metric.update(predicted, labels)
                f1_score_metric.update(predicted, labels)
                
                # Display the current average validation loss in the progress bar
                val_progress_bar.set_postfix(loss=f"{(val_loss / (val_progress_bar.n + 1)):.4f}")
                
        # Calculate the mean loss for both phases over the entire epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Finalize the metric calculations for the current epoch
        accuracy = accuracy_metric.compute().item() * 100
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1_score = f1_score_metric.compute().item()
        
        # Clear metric states for the next epoch's calculations
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_score_metric.reset()
        
        # Update the learning rate if the validation loss has plateaued
        scheduler.step(avg_val_loss)
        
        # Display a summary of performance and loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Log the completion of the training process
    print("\n--- Finished Training ---")
    
    # Consolidate the final performance statistics into a single structure
    final_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
    
    # Return the optimized model and its final validation metrics
    return model, final_metrics


# Trains and tests a classifier and returns a history of loss and accuracy
def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=5, device=None):
    """
    Executes the training and validation loop for a machine learning model.

    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader providing the training dataset batches.
        test_loader: DataLoader providing the validation or test dataset batches.
        optimizer: The optimization algorithm used for weight updates.
        criterion: The loss function used to evaluate model performance.
        num_epochs: Total number of iterations over the training dataset.
        device: The computation device to use; defaults to CUDA if available.

    Returns:
        history: A dictionary containing lists of accuracies and losses for 
                 both training and testing, as well as the best model's 
                 state and epoch information.
    """
    # Initialize the computation device if one is not explicitly provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transfer the model parameters to the designated computation device
    model = model.to(device)
    
    # Initialize containers for tracking performance metrics throughout training
    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    
    # Initialize variables to track the best performance and model state
    best_test_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    print(f"Training on {device}")
    print("="*50)
    
    for epoch in range(num_epochs):
        # Set the model to training mode to enable dropout and batch normalization
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        
        # Initialize the progress bar for the training phase
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            # Transfer input and label tensors to the active device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Reset gradients and perform the forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Execute backpropagation and update model parameters
            loss.backward()
            optimizer.step()
            
            # Determine predictions and update running accuracy and loss totals
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item() * labels.size(0)
            
            # Calculate metrics and update the progress bar display
            train_acc = 100 * train_correct / train_total
            avg_loss = train_loss / train_total
            train_bar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{train_acc:.1f}%'})
        
        # Set the model to evaluation mode to disable training-specific layers
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        # Initialize the progress bar for the testing/validation phase
        test_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
        # Disable gradient calculations to reduce memory usage and increase speed
        with torch.no_grad():
            for inputs, labels in test_bar:
                # Move tensors to the active device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Perform inference and calculate the loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track correct predictions and cumulative loss
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_loss += loss.item() * labels.size(0)
                
                # Update metrics and progress bar status
                test_acc = 100 * test_correct / test_total
                test_bar.set_postfix({'acc': f'{test_acc:.1f}%'})
        
        # Finalize accuracy and loss calculations for the current epoch
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        avg_train_loss = train_loss / train_total
        avg_test_loss = test_loss / test_total
        
        # Append epoch results to the history container
        history['train_acc'].append(train_accuracy)
        history['test_acc'].append(test_accuracy)
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        
        # Determine if the current epoch produced the best results based on test accuracy
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_epoch = epoch + 1
            # Capture the state of the model and optimizer at the peak performance
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict().copy(),
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'test_loss': avg_test_loss,
                'train_loss': avg_train_loss
            }
            print(f'  🎯 New best model! Test Acc: {test_accuracy:.2f}%')
        
        # Output a summary of metrics for the current epoch
        print(f'Epoch {epoch+1}/{num_epochs} Summary:')
        print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%')
        print(f'  Test   - Loss: {avg_test_loss:.4f}, Acc: {test_accuracy:.2f}%')
        print('-'*50)
    
    # Restore model parameters from the best performing epoch identified during training
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print("\n" + "="*50)
        print(f"Training completed! Best model restored from epoch {best_epoch}")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    else:
        print("\nTraining completed!")
        print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    
    # Store the top performance metadata in the history dictionary
    history['best_epoch'] = best_epoch
    history['best_test_acc'] = best_test_acc
    history['best_model_state'] = best_model_state
    
    # Return the dictionary containing training logs and best model information
    return history


# Trains a model, saves the best checkpoint and can resume from a saved one
def train_model_with_checkpoint(model, train_loader, dev_loader, num_epochs, optimizer, device, checkpoint=None, save_path="best_model.pt"): 
    # Load from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch']
        if 'val_acc' in checkpoint:
            best_acc = checkpoint['val_acc']
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc='Epochs', position=0, leave=True)
    
    for epoch in epoch_pbar:
        # Create progress bar for batches
        batch_pbar = tqdm(train_loader, desc='Training', position=1, leave=False)
        
        # Train for one epoch
        model.train()
        total_loss, total = 0.0, 0
        for x, y in batch_pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            # Update batch progress bar
            batch_pbar.set_description(f"Batch Loss: {loss.item():.4f}")
            
        train_loss = total_loss / max(total, 1)
        batch_pbar.close()
        
        # Evaluate
        val_acc = compute_accuracy(model, dev_loader, device)
        
        # Update epoch progress bar description
        epoch_pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = make_checkpoint(epoch, model, optimizer, train_loss,
                                 extra={"val_acc": val_acc})
            torch.save(ckpt, "best_model.pt")
            epoch_pbar.write(f"New best accuracy: {val_acc:.4f}, saved model to best_model.pt")
    
    # Save final model
    ckpt = make_checkpoint(num_epochs-1, model, optimizer, train_loss,
                          extra={"val_acc": val_acc})
    torch.save(ckpt, save_path)
    
    # Final status update
    print(f"\nTraining completed:")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Final accuracy: {val_acc:.4f}")
    print(f"Final model saved to final_model.pt")
    
    return model, best_acc


# Trains a next-token sequence model with cross entropy
def train_language_model(model, vocab_size, loader, loss_fn, optimizer, epochs=10, device='cpu'):
    """
    Executes the training loop for a sequence generation model.

    Args:
        model (nn.Module): The neural network model to be trained.
        vocab_size (int): The total number of unique tokens in the vocabulary.
        loader (DataLoader): The data loader providing batches of training data.
        loss_fn (Callable): The loss function used for optimization.
        optimizer (Optimizer): The optimization algorithm for weight updates.
        epochs (int): The total number of iterations over the dataset.
        device (str): The target computation device for the tensors and model.

    Returns:
        None: The function updates the model in-place and prints progress.
    """
    # Transfer the model parameters to the specified computation hardware
    model.to(device)
    
    for epoch in range(epochs):
        # Configure the model for training mode to enable specific behaviors like dropout
        model.train()
        # Initialize a list to accumulate loss values for each batch in the epoch
        epoch_losses = []
        
        # Iterate through the data loader with a progress bar display
        with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for xb, yb in pbar:
                # Move the input and target tensors to the active computation device
                xb, yb = xb.to(device), yb.to(device)
                
                # Reset the gradients to zero before starting a new optimization step
                optimizer.zero_grad()
                
                # Generate model predictions for the current input batch
                logits = model(xb)
                
                # Flatten the logits and targets to compute the multi-class cross-entropy loss
                loss = loss_fn(
                    logits.reshape(-1, vocab_size),
                    yb.reshape(-1)
                )
                
                # Compute the gradient of the loss with respect to the model parameters
                loss.backward()
                
                # Rescale the gradients to prevent the exploding gradient problem during training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Adjust the model weights based on the computed gradients
                optimizer.step()
                
                # Store the scalar loss value for later average calculation
                epoch_losses.append(loss.item())
                # Update the progress bar information with the current batch loss
                pbar.set_postfix(loss=loss.item())
        
        # Compute the arithmetic mean of the loss across all batches in the current epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        # Log the performance summary for the completed epoch
        print(f"Epoch {epoch+1:2d}: avg loss = {avg_loss:.4f}")


# Trains an encoder-decoder translation model and reports loss per epoch
def train_translation_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
    """
    Simple training function for the translator
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Store history
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_total_loss = 0
        train_batches = 0
        
        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for src_batch, tgt_batch in train_bar:
            # Move to device
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # Prepare decoder input and target
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(src_batch, tgt_input)
            
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(outputs, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track loss
            train_total_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            avg_loss = train_total_loss / train_batches
            train_bar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                
                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]
                
                outputs = model(src_batch, tgt_input)
                outputs = outputs.reshape(-1, outputs.size(-1))
                tgt_output = tgt_output.reshape(-1)
                
                loss = criterion(outputs, tgt_output)
                
                val_total_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        train_loss = train_total_loss / train_batches
        val_loss = val_total_loss / val_batches
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}\n')
    
    return history


# Trains a Siamese model in place on triplet batches
def siamese_training_loop(model, dataloader, loss_fcn, optimizer, device, n_epochs=5):
    """
    Executes the training process for a Siamese network model **in-place**.

    This function iterates over the provided dataloader for a specified
    number of epochs, computes the loss, updates the model weights, and ensures
    the model's final state corresponds to the epoch with the lowest average loss.

    Args:
        model (nn.Module): The neural network model to be trained. **It will be modified in-place.**
        dataloader (DataLoader): The DataLoader providing the training data (e.g., triplets).
        loss_fcn (callable): The loss function used for training (e.g., TripletMarginLoss).
        optimizer (torch.optim.Optimizer): The optimization algorithm (e.g., Adam).
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to perform training on.
        n_epochs (int): The total number of training epochs.

    Returns:
        None: The model is trained in-place and its state is set to the best epoch.
    """
    # Move the model to the target device (CPU or GPU)
    model.to(device)

    # Variables to track the best performing epoch
    best_loss = float('inf') # Initialize best loss to infinity
    best_epoch = 0
    best_model_state = None # To store the state dict of the best model

    # Print a message to indicate the start of training
    print("\n--- Starting Siamese Training ---")

    # Loop over the dataset for the specified number of epochs
    for epoch in range(1, n_epochs + 1):
        # Set the model to training mode (enables dropout, batch norm updates, etc.)
        model.train()
        # Initialize a running loss counter for the current epoch
        running_loss = 0.0
        # Wrap the dataloader with tqdm for a visual progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)

        # Iterate over batches of data from the dataloader
        for data_batch in progress_bar:
            # Unpack the data batch (anchor, positive, negative samples)
            anchor, positive, negative = data_batch
            # Move all tensors in the batch to the selected device
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            # --- Training Step ---
            # 1. Clear the gradients from the previous iteration
            optimizer.zero_grad()

            # 2. Perform a forward pass to get the model's outputs (embeddings)
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)

            # 3. Calculate the loss based on the model's outputs
            loss = loss_fcn(anchor_out, positive_out, negative_out)

            # 4. Perform a backward pass to compute gradients
            loss.backward()
            # 5. Update the model's weights using the optimizer
            optimizer.step()
            # --- End Training Step ---

            # Add the loss from the current batch to the running total
            running_loss += loss.item()
            # Update the progress bar's postfix to show the current average loss
            # Use progress_bar.n + 1 for correct average calculation during iteration
            progress_bar.set_postfix(loss=f"{running_loss / (progress_bar.n + 1):.4f}")

        # Calculate the average loss for the entire epoch
        epoch_loss = running_loss / len(dataloader)
        # Print the average loss for the completed epoch
        print(f"Epoch {epoch}/{n_epochs} finished, Average Loss: {epoch_loss:.4f}")

        # --- Check if this epoch is the best so far ---
        if epoch_loss < best_loss:
            best_loss = epoch_loss # Update best loss
            best_epoch = epoch # Update best epoch number
            # Save the model's state dictionary on the CPU
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"    -> New best model saved (Epoch {best_epoch}, Loss: {best_loss:.4f})")
        # --- End Check ---

    # --- Training Complete ---
    print("\n--- Siamese Training Complete ---")

    # Load the best model weights found during training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model weights from Epoch {best_epoch} with lowest loss: {best_loss:.4f}")
    else:
        print("Warning: No best model state was saved (check training loop). Using weights from the last epoch.")


# Performs Quantization-Aware Training (QAT) fine-tuning for a given model
def train_qat(model_to_train, trainloader, device, epochs=5):
    """
    Performs Quantization-Aware Training (QAT) fine-tuning for a given model.

    Args:
        model_to_train (torch.nn.Module): The QAT-prepared model.
        trainloader (torch.utils.data.DataLoader): The data loader for training.
        device (torch.device): The device to train on.
        epochs (int): The number of epochs to train.

    Returns:
        model_to_train (torch.nn.Module): The fine-tuned model after QAT.
    """
    
    # Move the model to the specified hardware device
    model_to_train.to(device)
    
    # Set the model to training mode to enable dropout and batch normalization updates
    model_to_train.train()

    # Define the loss function for the training process
    loss_function = nn.CrossEntropyLoss()
    
    # Initialize the Stochastic Gradient Descent optimizer with momentum
    optimizer = optim.SGD(model_to_train.parameters(), lr=0.001, momentum=0.9)
    
    # Initialize the progress bar for the entire epoch sequence
    epoch_loop = tqdm(range(epochs), desc="QAT Training Progress")
    
    # Iterate through the specified number of training epochs
    for epoch in epoch_loop:
        # Initialize tracking variables for the current epoch
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize a nested progress bar for individual batches
        batch_loop = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        # Iterate through batches provided by the data loader
        for data in batch_loop:
            # Extract inputs and target labels from the batch
            inputs, labels = data
            
            # Transfer the input data and labels to the configured device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset gradients for the current optimization step
            optimizer.zero_grad()
            
            # Perform a forward pass through the model
            outputs = model_to_train(inputs)
            
            # Compute the loss based on model predictions and actual labels
            loss = loss_function(outputs, labels)
            
            # Calculate gradients through backpropagation
            loss.backward()
            
            # Update the model parameters based on the computed gradients
            optimizer.step()
            
            # Accumulate the running loss value
            running_loss += loss.item()

            # Determine the predicted class for accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            
            # Update the total count of processed samples
            total += labels.size(0)
            
            # Update the count of correctly predicted samples
            correct += (predicted == labels).sum().item()
            
            # Update the batch progress bar with the current loss value
            batch_loop.set_postfix(loss=f"{loss.item():.3f}")

        # Calculate the average loss over all training batches
        avg_epoch_loss = running_loss / len(trainloader)
        
        # Calculate the final training accuracy for the epoch
        train_accuracy = 100 * correct / total
        
        # Print the statistical summary for the current epoch
        print(f'Epoch {epoch + 1}, Training Loss: {avg_epoch_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        
        # Update the main epoch progress bar with the average loss
        epoch_loop.set_postfix(avg_loss=f"{avg_epoch_loss:.3f}")

    # Log the completion of the QAT training process
    print('\nQAT Training finished.')
    
    # Return the model following the fine-tuning process
    return model_to_train


# Evaluates the performance of a quantization-aware trained model on a dataset
def evaluate_qat(model_to_eval, dataloader):
    """
    Evaluates the performance of a quantization-aware trained model on a dataset.

    Args:
        model_to_eval (torch.nn.Module): The model currently in a quantized state for evaluation.
        dataloader (torch.utils.data.DataLoader): The data provider for the evaluation process.
        device (str): The hardware device where the evaluation is performed.

    Returns:
        accuracy (float): The calculated accuracy of the model as a percentage.
    """

    # Explicitly define the device as CPU to ensure all operations run it, as they should be.
    device="cpu"
    
    # Transfer the model parameters to the target hardware device
    model_to_eval.to(device)
    
    # Switch the model to evaluation mode to disable layers like dropout
    model_to_eval.eval() 
    
    # Initialize counters for the number of correct predictions and total samples
    correct = 0
    total = 0
    
    # Deactivate gradient calculations to conserve memory and computational resources
    with torch.no_grad():
        # Initialize a progress bar to track evaluation across batches
        progress_bar = tqdm(dataloader, desc="Evaluating QAT Model")
        
        # Iterate through images and ground truth labels in the dataset
        for images, labels in progress_bar:

            # Move the batch of images and labels to the active device
            images, labels = images.to(device), labels.to(device)
            
            # Pass the input images through the model to obtain predictions
            outputs = model_to_eval(images)
            
            # Identify the class index with the highest activation value
            _, predicted = torch.max(outputs.data, 1)
            
            # Increment the total count of processed samples
            total += labels.size(0)
            
            # Increment the count of predictions that match the labels
            correct += (predicted == labels).sum().item()

    # Calculate the final accuracy percentage
    accuracy = 100 * correct / total
    
    # Log the resulting accuracy to the console
    print(f'Accuracy of the QAT model on the test set: {accuracy:.2f}%')

    # Return the resulting accuracy value
    return accuracy


# Computes the accuracy of a classifier separately for each class
def evaluate_per_class(model, test_loader, device, num_classes=None, label_offset=0, class_names=None):
    """
    Computes the accuracy of a classifier for each class separately.

    Args:
        model: Classifier that returns one score per class.
        test_loader: DataLoader yielding (inputs, targets).
        device: Device to run the model on.
        num_classes: Number of classes. Inferred from the data when None.
        label_offset: Value subtracted from the targets, for datasets whose labels start at 1.
        class_names: Optional list of names used as keys of the result.

    Returns:
        Dict mapping class name (or index) to accuracy. Classes without samples get 0.0.
    """
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_targets.append((targets - label_offset).cpu())
    targets_np = torch.cat(all_targets).numpy()
    preds_np = torch.cat(all_preds).numpy()
    if num_classes is None:
        num_classes = int(max(targets_np.max(), preds_np.max())) + 1
    class_accuracies = {}
    for idx in range(num_classes):
        mask = targets_np == idx
        key = class_names[idx] if class_names is not None else idx
        class_accuracies[key] = float((preds_np[mask] == idx).mean()) if mask.any() else 0.0
    return class_accuracies


# Checks that a train_epoch function updates weights and lowers loss on a few batches
def verify_training_process(model_factory, train_loader, loss_function, train_epoch_fn, device,
                            weight_getter=None, num_epochs=5, num_batches=10, lr=0.0005):
    """
    Sanity-checks a train_epoch function on a few batches: weights must change and loss should drop.

    Args:
        model_factory: Function with no arguments that returns a fresh, untrained model.
        train_loader: DataLoader used to take a few batches from.
        loss_function: Loss passed to train_epoch_fn.
        train_epoch_fn: Function called as train_epoch_fn(model=, train_loader=, loss_function=, optimizer=, device=)
            that returns the epoch loss.
        device: Device to train on.
        weight_getter: Function that returns the weight tensor to watch. Defaults to the first parameter.
        num_epochs: Number of epochs to run on the small subset.
        num_batches: Number of batches to build the subset from.
        lr: Learning rate of the Adam optimizer used for the check.
    """
    print(f'--- Verifying train_epoch (training for {num_epochs} epochs) ---\n')
    weight_getter = weight_getter or (lambda m: next(m.parameters()))
    verify_model = model_factory().to(device)
    verify_optimizer = optim.Adam(verify_model.parameters(), lr=lr)
    batches = list(itertools.islice(iter(train_loader), num_batches))
    subset = TensorDataset(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches]))
    subset_loader = DataLoader(subset, batch_size=train_loader.batch_size)
    initial_weight = weight_getter(verify_model).clone()
    epoch_losses = []
    print(f'Training on {len(subset)} samples for {num_epochs} epochs:\n')
    for epoch in range(num_epochs):
        loss = train_epoch_fn(model=verify_model, train_loader=subset_loader, loss_function=loss_function,
                              optimizer=verify_optimizer, device=device)
        epoch_losses.append(loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')
    if not torch.equal(initial_weight, weight_getter(verify_model)):
        print('\nWeight Update Check:\tModel weights changed during training.')
    else:
        print('\nWeight Update Check:\tModel weights DID NOT change.')
    if epoch_losses[-1] < epoch_losses[0]:
        print(f'Loss Trend Check:\tLoss decreased from {epoch_losses[0]:.4f} to {epoch_losses[-1]:.4f}.')
    else:
        print('Loss Trend Check:\tLoss DID NOT show a decreasing trend.')


# Checks that a validate_epoch function returns floats and leaves weights unchanged
def verify_validation_process(model_factory, val_loader, loss_function, validate_epoch_fn, device,
                              weight_getter=None, num_batches=10):
    """
    Sanity-checks a validate_epoch function: it must return float loss and accuracy and leave weights untouched.

    Args:
        model_factory: Function with no arguments that returns a fresh model.
        val_loader: DataLoader used to take a few batches from.
        loss_function: Loss passed to validate_epoch_fn.
        validate_epoch_fn: Function called as validate_epoch_fn(model=, val_loader=, loss_function=, device=)
            that returns (loss, accuracy).
        device: Device to validate on.
        weight_getter: Function that returns the weight tensor to watch. Defaults to the first parameter.
        num_batches: Number of batches to build the subset from.
    """
    print('--- Verifying validate_epoch ---\n')
    weight_getter = weight_getter or (lambda m: next(m.parameters()))
    verify_model = model_factory().to(device)
    batches = list(itertools.islice(iter(val_loader), num_batches))
    subset = TensorDataset(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches]))
    subset_loader = DataLoader(subset, batch_size=val_loader.batch_size)
    initial_weight = weight_getter(verify_model).clone()
    print(f'Validating on {len(subset)} samples:\n')
    val_loss, val_accuracy = validate_epoch_fn(model=verify_model, val_loader=subset_loader,
                                               loss_function=loss_function, device=device)
    print(f'Returned Validation Loss: {val_loss:.4f}')
    print(f'Returned Validation Accuracy: {val_accuracy:.2f}%\n')
    if isinstance(val_loss, float) and isinstance(val_accuracy, float):
        print('\nReturn Types Check:\tFunction returned a float for loss and accuracy.')
    else:
        print('\nReturn Types Check:\tFunction DID NOT return the correct data types.')
    if torch.equal(initial_weight, weight_getter(verify_model)):
        print('Weight Integrity Check:\tModel weights were not changed during validation.')
    else:
        print('Weight Integrity Check:\tModel weights WERE CHANGED.')


# Step function that runs a model on (anchor, positive, negative) batches
def siamese_triplet_step(model, batch, device):
    """
    Step function for training_loop_embedding with (anchor, positive, negative) batches.

    Returns:
        ((anchor_out, positive_out, negative_out), ()) with no extra loss targets.
    """
    anchor, positive, negative = (t.to(device) for t in batch)
    return model(anchor, positive, negative), ()


# Step function that runs a model on (input_a, input_b, label) batches
def siamese_pair_step(model, batch, device, **forward_kwargs):
    """
    Step function for training_loop_embedding with (input_a, input_b, label) batches.

    Args:
        model: Model called as model(input_a, input_b, **forward_kwargs).
        batch: (input_a, input_b, labels) tensors.
        device: Device to move the tensors to.
        **forward_kwargs: Extra keyword arguments for the model call. Use functools.partial to bind them.

    Returns:
        ((output_a, output_b), (labels,)) where labels are passed to the loss as the last argument.
    """
    input_a, input_b, labels = (t.to(device) for t in batch)
    return model(input_a, input_b, **forward_kwargs), (labels,)


# Builds a validation metric that scores triplet outputs with a distance threshold
def triplet_threshold_accuracy(threshold):
    """
    Builds a val_metric_fn for training_loop_embedding that scores triplet outputs with a distance threshold.

    A positive pair counts as correct below the threshold, a negative pair at or above it.
    """
    def metric(outputs, targets):
        anchor_out, positive_out, negative_out = outputs
        dist_pos = F.pairwise_distance(anchor_out, positive_out)
        dist_neg = F.pairwise_distance(anchor_out, negative_out)
        correct = (dist_pos < threshold).sum().item() + (dist_neg >= threshold).sum().item()
        return correct, len(dist_pos) + len(dist_neg)
    return metric


# Trains and validates a Siamese, triplet or pair model and keeps the best weights
def training_loop_embedding(model, train_loader, val_loader, loss_fcn, optimizer, device, step_fn, n_epochs=10,
                            scheduler=None, save_path=None, val_metric_fn=None):
    """
    Trains and validates an embedding model (Siamese, triplet or pair) and keeps the best weights.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        loss_fcn: Called as loss_fcn(*outputs, *targets).
        optimizer: Optimizer.
        device: Device to train on.
        step_fn: Called as step_fn(model, batch, device) and returns (outputs, targets), both tuples.
            See siamese_triplet_step and siamese_pair_step.
        n_epochs: Number of epochs.
        scheduler: Optional scheduler. ReduceLROnPlateau receives the validation loss.
        save_path: Optional file for the best weights. They are reloaded at the end.
        val_metric_fn: Optional function (outputs, targets) -> (n_correct, n_total).
            When given, the best epoch is the one with the highest validation accuracy,
            otherwise the one with the lowest validation loss.

    Returns:
        The trained model.
    """
    if save_path and os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.to(device)
    best_score = -float('inf')
    print('--- Starting Training & Validation ---\n')
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs} [Training]', leave=False)
        for batch in bar:
            optimizer.zero_grad()
            outputs, targets = step_fn(model, batch, device)
            loss = loss_fcn(*outputs, *targets)
            loss.backward()
            optimizer.step()
            size = batch[0].size(0)
            running_loss += loss.item() * size
            seen += size
            bar.set_postfix(loss=f'{running_loss / seen:.4f}')
        train_loss = running_loss / len(train_loader.dataset)
        model.eval()
        running_loss, seen, correct, total = 0.0, 0, 0, 0
        bar = tqdm(val_loader, desc=f'Epoch {epoch}/{n_epochs} [Validation]', leave=False)
        with torch.no_grad():
            for batch in bar:
                outputs, targets = step_fn(model, batch, device)
                size = batch[0].size(0)
                running_loss += loss_fcn(*outputs, *targets).item() * size
                seen += size
                if val_metric_fn is not None:
                    c, t = val_metric_fn(outputs, targets)
                    correct += c
                    total += t
                bar.set_postfix(loss=f'{running_loss / seen:.4f}')
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0
        score = val_acc if val_metric_fn is not None else -val_loss
        acc_text = f' | Val Acc: {val_acc:.2%}' if val_metric_fn is not None else ''
        print(f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{acc_text}"
              f" | LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        if score > best_score:
            best_score = score
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  -> New best model saved to '{save_path}'\n")
    print('\n--- Training & Validation Complete ---')
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    return model


# Finds the embedding distance threshold that best separates same from different pairs
def find_best_distance_threshold(model, data_loader, loss_fcn, device, step_fn, label_to_binary=None, num_thresholds=100):
    """
    Finds the embedding distance threshold that best separates 'same' from 'different' pairs.

    Args:
        model: Pair model.
        data_loader: DataLoader of pair batches.
        loss_fcn: Called as loss_fcn(*outputs, *targets) to report the validation loss.
        device: Device to run on.
        step_fn: Called as step_fn(model, batch, device) and returns ((out_a, out_b), (labels,)).
            See siamese_pair_step.
        label_to_binary: Optional function that turns labels into 0 (same) and 1 (different). Identity when None.
        num_thresholds: Number of candidate thresholds between the smallest and largest distance.

    Returns:
        The threshold with the best binary accuracy. A pair is 'different' when its distance is above it.
    """
    print('--- Starting Evaluation ---\n')
    model.eval()
    all_distances, all_labels, running_loss = [], [], 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='[Validation]', leave=False):
            outputs, targets = step_fn(model, batch, device)
            all_distances.extend(F.pairwise_distance(outputs[0], outputs[1]).cpu().numpy())
            all_labels.extend(targets[-1].cpu().numpy())
            running_loss += loss_fcn(*outputs, *targets).item()
    all_distances = np.array(all_distances)
    labels = np.array(all_labels)
    truth = np.array(label_to_binary(labels)).astype(int) if label_to_binary else labels.astype(int)
    best_accuracy, best_threshold = 0.0, 0.0
    for t in np.linspace(all_distances.min(), all_distances.max(), num=num_thresholds):
        accuracy = np.mean((all_distances > t).astype(int) == truth)
        if accuracy > best_accuracy:
            best_accuracy, best_threshold = accuracy, t
    print('\n--- Final Evaluation Results ---')
    print(f'  -> Best Binary Accuracy: {best_accuracy:.2%}')
    print(f'  -> At Optimal Distance Threshold: {best_threshold:.4f}')
    print(f'  -> Validation Loss: {running_loss / len(data_loader):.4f}\n')
    return best_threshold



# =============================================================================
# INFERENCE, EMBEDDINGS AND TEXT GENERATION
# =============================================================================


# Predicts the class of one image file and shows the image with its label
def make_predictions(model, image_path, device, class_names):
    """
    Loads a single image, applies transformations, makes a prediction,
    and displays the image with its predicted class label.

    Args:
        model (torch.nn.Module): The trained model for making predictions.
        image_path (str): The file path to the input .jpg image.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        class_names (list): A list of class names corresponding to the labels.
    """
    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    # Define the mean and std for normalization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Define the transformation pipeline
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 1. Load and transform the image
    try:
        # Open the image and ensure it's in RGB format
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return
        
    image_tensor = val_transform(image)
    
    # Add a batch dimension (models expect a batch of images)
    # The shape changes from [C, H, W] to [1, C, H, W]
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # 2. Make a prediction
    with torch.no_grad():
        # Get the model's raw output (logits)
        outputs = model(input_tensor)
        # Get the index of the highest score, which is our predicted class
        _, predicted_idx = torch.max(outputs, 1)

    # Get the predicted class name
    pred_label = class_names[predicted_idx.item()]

    # 3. Display the image and its prediction
    plt.figure(figsize=(6, 6))
    
    # To display the image correctly, we need to denormalize the tensor
    img_to_display = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_to_display = std.numpy() * img_to_display + mean.numpy()
    img_to_display = np.clip(img_to_display, 0, 1) # Clip values to be between 0 and 1

    plt.imshow(img_to_display)
    plt.title(f"Prediction: {pred_label}", fontsize=14)
    plt.axis("off")
    plt.show()


# Predicts the category of one text with a fine-tuned transformer
def predict_category(model, tokenizer, text, device, id2cat):
    """
    Performs inference on a single text string to predict its category.

    Args:
        model (nn.Module): The fine-tuned PyTorch model.
        tokenizer: The Hugging Face tokenizer corresponding to the model.
        text (str): The raw input text string.
        device: The device to perform inference on ('cuda', 'cpu', etc.).
        id2cat (dict): A dictionary mapping class IDs to category names.

    Returns:
        str: The predicted category name as a string.
    """
    # Set the model to evaluation mode.
    model.eval()

    # Tokenize the input text and create PyTorch tensors.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Move the input tensors to the specified device.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Disable gradient calculations to save memory and speed up inference.
    with torch.no_grad():
        # Perform a forward pass through the model to get the outputs.
        outputs = model(**inputs)
    
    # Get the raw prediction scores (logits).
    logits = outputs.logits
    # Find the index of the highest logit, which corresponds to the predicted class ID.
    predicted_id = torch.argmax(logits, dim=-1).item()
    
    # Map the numerical prediction ID back to the human-readable category name.
    return id2cat[predicted_id]


# Computes embeddings for every sample of a dataset with a model
def get_embeddings(model_representation, labeled_dataset, device):
    """
    Computes and returns the feature embeddings for a given dataset using a model.

    Args:
        model_representation: The feature extraction model.
        labeled_dataset: The dataset containing the data samples.
        device: The computational device (e.g., 'cpu' or 'cuda').

    Returns:
        A NumPy array containing the computed embeddings for the entire dataset.
    """
    # Create a DataLoader to process the dataset in batches
    dataloader = DataLoader(labeled_dataset, batch_size=32, shuffle=False)
    # Set the model to evaluation mode (e.g., disables dropout)
    model_representation.eval()
    # Initialize an empty list to store embeddings from each batch
    embeddings = []

    # Disable gradient computation during this block for efficiency
    with torch.no_grad():
        # Iterate over all batches in the dataloader
        for img, _ in dataloader:
            # Move the batch of images to the specified device
            img = img.to(device)
            # Perform a forward pass to get the embeddings
            embedding_img = model_representation.forward(img)
            # Move embeddings to CPU, convert to NumPy, and append to the list
            embeddings.append(embedding_img.cpu().numpy())

    # Concatenate the list of batch embeddings into a single NumPy array
    embeddings = np.concatenate(embeddings, axis=0)
    # Return the complete array of embeddings
    return embeddings


# Finds the indices of the samples with the closest embeddings to the target embedding
def find_closest(embeddings, target_embedding, num_samples=5):
    """
    Finds the indices of the samples with the closest embeddings to the target embedding.

    Args:
        embeddings: A NumPy array containing the feature embeddings of all samples.
        target_embedding: A NumPy array representing the embedding of the query sample.
        num_samples: The number of closest sample indices to retrieve.

    Returns:
        A NumPy array of the indices corresponding to the closest samples.
    """
    # Calculate Euclidean distances from the target to all other embeddings
    distances = np.linalg.norm(embeddings - target_embedding, axis=1)

    # Get indices of the closest samples by sorting the distances
    # We slice from 1 to num_samples + 1 to exclude the target itself (which has distance 0)
    closest_indices = np.argsort(distances)[
        1 : num_samples + 1
    ]

    # Return the indices of the closest samples
    return closest_indices


# Performs Visual Question Answering by processing an image and a text query
def perform_vqa(model, processor, image_path, question_text):
    """
    Performs Visual Question Answering by processing an image and a text query.

    Args:
        model (torch.nn.Module): The pre-trained VQA model used for generation.
        processor (transformers.ProcessorMixin): The processor for image and text normalization.
        image_path (str): The local filesystem path to the input image.
        question_text (str): The natural language question regarding the image.

    Returns:
        answer (str): The text-based answer generated by the model.
        inference_time (float): The duration of the model generation process in seconds.
    """
    # Open the image file from the provided path and convert it to RGB format
    raw_image = Image.open(image_path).convert('RGB')
    
    # Prepare the model inputs by encoding the image and the question text
    inputs = processor(raw_image, question_text, return_tensors="pt")
    
    # Record the high-resolution start time before the model forward pass
    start_time = time.perf_counter()
    
    # Generate the output tokens from the model based on the processed inputs
    output = model.generate(**inputs)
    
    # Record the high-resolution end time after the generation is complete
    end_time = time.perf_counter()
    
    # Calculate the elapsed time specifically for the generation step
    inference_time = end_time - start_time
    
    # Decode the generated output tokens into a human-readable string
    answer = processor.decode(output[0], skip_special_tokens=True)
    
    # Return the generated text answer and the measured inference time
    return answer, inference_time


# Implements an autoregressive token generation loop with various sampling heuristics
@torch.no_grad()
def generate_tokens(model, prompt_ids, max_length=100, temperature=1.0, 
                    top_k=50, top_p=0.95, repetition_penalty=1.2, 
                    eos_token_id=None, device='cpu'):
    """
    Implements an autoregressive token generation loop with various sampling heuristics.

    Args:
        model (nn.Module): The neural network used to compute logit predictions.
        prompt_ids (list or Tensor): Initial sequence of token identifiers to start generation.
        max_length (int): The total maximum length of the sequence including the prompt.
        temperature (float): Scaling factor to adjust the sharpness of the probability distribution.
        top_k (int): The number of highest probability tokens to keep for filtering.
        top_p (float): The cumulative probability threshold for nucleus sampling.
        repetition_penalty (float): Factor used to discount the logits of previously seen tokens.
        eos_token_id (int): Specific token identifier that signifies the end of a sequence.
        device (str or torch.device): The target hardware for computation.

    Returns:
        generated_sequence (Tensor): A flattened tensor containing the original prompt and generated tokens.
    """
    # Set the model to evaluation mode to ensure consistent behavior of layers like dropout
    model.eval()
    
    # Standardize the prompt input into a 2D batch tensor on the target device
    if isinstance(prompt_ids, list):
        prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    elif len(prompt_ids.shape) == 1:
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
    else:
        prompt_ids = prompt_ids.to(device)
    
    # Create a copy of the prompt to append new tokens to
    generated = prompt_ids.clone()
    # Maintain a local list of tokens for applying repetition penalties efficiently
    past_tokens = list(prompt_ids[0].cpu().numpy())
    
    # Iterate until the maximum sequence length is reached
    for step in range(max_length - len(prompt_ids[0])):
        # Perform a forward pass, potentially using mixed precision for performance
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(generated)
        
        # Extract the logits for the most recently predicted position
        next_token_logits = logits[0, -1, :].float()
        
        # Adjust the distribution variance using the temperature parameter
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Reduce the likelihood of tokens that have already appeared in the sequence
        if repetition_penalty != 1.0:
            # Apply a penalty factor to every unique token already generated
            for token_id in set(past_tokens):
                next_token_logits[token_id] /= repetition_penalty
            
            # Apply an additional penalty for the three most recent tokens to avoid loops
            if len(past_tokens) > 3:
                for token_id in past_tokens[-3:]:
                    next_token_logits[token_id] /= 1.5
        
        # Zero out the probability of all tokens outside the top K highest scores
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, len(next_token_logits)))[0][-1]
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Filter out tokens that fall outside the cumulative probability threshold
        if top_p < 1.0:
            # Sort the logits to calculate cumulative probabilities in descending order
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Identify indices where the cumulative probability exceeds the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift indices to ensure at least one token remains available for sampling
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Mask the logits of the tokens that are to be removed
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Calculate the final probability distribution and sample the next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        # Concatenate the new token to the existing sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        # Update the history of generated tokens
        past_tokens.append(next_token.item())
        
        # Terminate the generation process if the model produces the end-of-sequence token
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    
    # Remove the batch dimension before returning the final tensor
    return generated.squeeze(0)


# Transforms a string prompt into a generated sequence of text using sampling strategies
def generate_text(model, prompt, tokenizer, word2idx, idx2word, 
                  max_length=100, temperature=0.8, top_k=50, top_p=0.95,
                  repetition_penalty=1.2, device='cpu'):
    """
    Transforms a string prompt into a generated sequence of text using sampling strategies.

    Args:
        model (nn.Module): The pre-trained model used for sequence prediction.
        prompt (str): The starting text string used to seed the generation.
        tokenizer (object): The tokenizer instance used to split the prompt into tokens.
        word2idx (dict): A dictionary mapping token strings to their numerical indices.
        idx2word (dict): A dictionary mapping numerical indices to their token strings.
        max_length (int): The maximum number of tokens to be generated.
        temperature (float): A value adjusting the randomness of the sampling distribution.
        top_k (int): The number of highest-probability tokens to consider during filtering.
        top_p (float): The cumulative probability threshold for nucleus sampling.
        repetition_penalty (float): The factor used to discourage the model from repeating tokens.
        device (str): The target hardware for executing the model operations.

    Returns:
        final_text (str): The decoded and formatted text string generated by the model.
    """
    # Evaluate the prompt and provide a default token if the input is empty or whitespace
    if not prompt or prompt.isspace():
        # Initialize with a common seed token
        prompt_tokens = ['the']
    else:
        # Split the input prompt into a list of lowercase tokens
        prompt_tokens = tokenizer(prompt.lower())
    
    # Initialize a list to store the numerical index for each prompt token
    prompt_ids = []
    # Map each string token to its corresponding vocabulary index
    for token in prompt_tokens:
        # Check if the exact token exists in the mapping dictionary
        if token in word2idx:
            prompt_ids.append(word2idx[token])
        else:
            # Attempt to find a lowercase version if the initial match fails
            token_lower = token.lower()
            if token_lower in word2idx:
                prompt_ids.append(word2idx[token_lower])
            else:
                # Use the designated unknown token for words not in the vocabulary
                prompt_ids.append(word2idx['<unk>'])
    
    # Ensure the sequence is not empty by appending a fallback token if necessary
    if not prompt_ids:
        prompt_ids = [word2idx.get('the', word2idx['<unk>'])]
    
    # Identify the index for the end-of-sequence token if it exists in the vocabulary
    eos_token_id = word2idx.get('<eos>', None)
    
    # Execute the core token generation loop with sampling parameters
    generated_ids = generate_tokens(
        model,
        prompt_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        device=device
    )
    
    # Initialize a list to hold the decoded string tokens
    tokens = []
    # Iterate through the generated numerical indices to recover strings
    for idx in generated_ids:
        # Ensure the index is a standard integer value
        idx_val = idx.item() if hasattr(idx, 'item') else idx
        # Retrieve the word associated with the index or use the unknown placeholder
        token = idx2word.get(idx_val, '<unk>')
        
        # Convert special newline tokens into actual line break characters
        if token == '<nl>' or token == '<newline>':
            tokens.append('\n')
        # Exclude technical special tokens from the final human-readable output
        elif token not in ['<pad>', '<unk>', '<eos>', '<start>']:
            tokens.append(token)
    
    # Combine the list of strings into a single sentence separated by spaces
    text = ' '.join(tokens)
    
    # Remove artificial spaces before common punctuation marks
    text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!')
    text = text.replace(' ?', '?').replace(' ;', ';').replace(' :', ':')
    # Consolidate spacing around apostrophes and quotes
    text = text.replace(' \'', '\'').replace('\' ', '\'')
    # Remove excessive spacing around line breaks
    text = text.replace(' \n ', '\n').replace('\n ', '\n')
    
    # Return the cleaned and formatted string with leading/trailing whitespace removed
    return text.strip()


# Generates text sequences for a collection of input prompts
def generate_batch(model, prompts, tokenizer, word2idx, idx2word, 
                   max_length=100, temperature=0.8, device='cpu'):
    """
    Generates text sequences for a collection of input prompts.

    Args:
        model (nn.Module): The pre-trained model used for sequence prediction.
        prompts (list): A list of strings used to seed the generation process.
        tokenizer (object): The tokenizer instance used to process the text.
        word2idx (dict): A dictionary mapping token strings to their numerical indices.
        idx2word (dict): A dictionary mapping numerical indices to their token strings.
        max_length (int): The maximum number of tokens to be generated for each prompt.
        temperature (float): A value adjusting the randomness of the sampling distribution.
        device (str): The target hardware for executing the model operations.

    Returns:
        results (list): A list of strings containing the generated text for each prompt.
    """
    # Initialize a list to store the generated text for each input prompt
    results = []
    # Iterate through each prompt provided in the input list
    for prompt in prompts:
        # Generate a text sequence based on the current prompt and parameters
        generated = generate_text(
            model, prompt, tokenizer, word2idx, idx2word,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        # Append the completed text sequence to the results collection
        results.append(generated)
    # Return the aggregated list of all generated sequences
    return results


# Turns a sequence of character images into a word using a classification model
def decode_word_imgs(word_imgs, model, device, label_to_char=None):
    """
    Decodes a sequence of character images into a string with a classification model.

    Args:
        word_imgs: Iterable of image tensors, one per character.
        model: Classifier that returns one score per class.
        device: Device to run the model on.
        label_to_char: Optional function that maps a class index to a string.
            Defaults to lowercase letters, where 0 -> 'a'.

    Returns:
        The decoded string.
    """
    if label_to_char is None:
        label_to_char = lambda idx: chr(ord('a') + idx)
    model.eval()
    decoded_chars = []
    with torch.no_grad():
        for char_img in word_imgs:
            output = model(char_img.unsqueeze(0).to(device))
            decoded_chars.append(label_to_char(output.argmax(dim=1).item()))
    return ''.join(decoded_chars)


# Predicts the class of a text with a vocabulary-based model
def predict_category_from_vocab(model, text, vocab, preprocess_text, device, id2label=None, use_offsets=None):
    """
    Predicts the class of a text with a model that works on vocabulary indices.

    Args:
        model: Text classifier.
        text: Raw input string.
        vocab: Object with an encode(tokens) method that returns a list of indices.
        preprocess_text: Function that turns the raw string into tokens for vocab.encode.
        device: Device to run the model on.
        id2label: Optional dict or list that maps class index to a label name.
        use_offsets: True for EmbeddingBag style models that take (indices, offsets).
            When None it is switched on for a class named 'EmbeddingBagClassifier'.

    Returns:
        The label name if id2label is given, otherwise the class index.
    """
    model.to(device)
    model.eval()
    if use_offsets is None:
        use_offsets = model.__class__.__name__ == 'EmbeddingBagClassifier'
    indexed = vocab.encode(preprocess_text(text))
    with torch.no_grad():
        if use_offsets:
            text_tensor = torch.tensor(indexed, dtype=torch.long).to(device)
            offsets = torch.tensor([0], dtype=torch.long).to(device)
            output = model(text_tensor, offsets)
        else:
            output = model(torch.tensor(indexed, dtype=torch.long).unsqueeze(0).to(device))
    predicted = torch.argmax(output, dim=1).item()
    return id2label[predicted] if id2label is not None else predicted


# Console loop that generates text from typed prompts with adjustable settings
def interactive_generation(model, tokenizer, word2idx, idx2word, device='cpu', title='Interactive Text Generation',
                           temperature=0.8, max_length=100):
    """
    Runs a console loop that generates text from typed prompts.

    Commands inside the loop: 'temp=0.8' sets the temperature, 'len=100' sets the length, 'quit' exits.

    Args:
        model: Trained language model.
        tokenizer, word2idx, idx2word: Text encoding objects passed to generate_text.
        device: Device to run on.
        title: Heading printed at the start.
        temperature: Starting sampling temperature.
        max_length: Starting number of tokens to generate.
    """
    print('=' * 50)
    print(title)
    print('=' * 50)
    print("Enter a prompt to generate text (or 'quit' to exit)")
    print("Commands: 'temp=0.8' to set temperature, 'len=100' to set length")
    print('-' * 50)
    while True:
        prompt = input('\nPrompt: ').strip()
        if prompt.lower() == 'quit':
            break
        if prompt.startswith('temp='):
            try:
                temperature = float(prompt[5:])
                print(f'Temperature set to {temperature}')
            except ValueError:
                print('Invalid temperature')
            continue
        if prompt.startswith('len='):
            try:
                max_length = int(prompt[4:])
                print(f'Max length set to {max_length}')
            except ValueError:
                print('Invalid length')
            continue
        generated = generate_text(model, prompt, tokenizer, word2idx, idx2word, max_length=max_length,
                                  temperature=temperature, device=device)
        print('\n' + '=' * 50)
        print('Generated:')
        print('-' * 50)
        print(generated)
        print('=' * 50)


# Jupyter dropdown that picks an image folder and shows predictions for its images
def predict_folder_widget(model, root_dir, class_names, true_label_fn=None, device='cpu', preprocess_fn=None,
                          max_images=10, grid=(2, 5), extensions=('.jpg', '.jpeg', '.png')):
    """
    Shows a dropdown of sub-folders and, for the chosen one, predictions for its first images.

    Args:
        model: Classifier that returns one score per class.
        root_dir: Folder whose sub-folders hold images.
        class_names: List of class names, indexed by class id.
        true_label_fn: Optional function mapping a folder name to its true class index. Titles show it when given.
        device: Device to run on.
        preprocess_fn: Function (image_path, device) -> input tensor of shape (1, C, H, W). Defaults to preprocess_image.
        max_images: Number of images shown per folder.
        grid: (rows, cols) of the image grid. Should hold max_images.
        extensions: File extensions that count as images.
    """
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return
    preprocess_fn = preprocess_fn or preprocess_image
    subdirs = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    dropdown = widgets.Dropdown(options=['Select directory'] + subdirs, value='Select directory',
                                description='Directory:')
    output_widget = widgets.Output()

    def update_grid(change):
        folder_name = change['new'] if isinstance(change, dict) else change
        with output_widget:
            output_widget.clear_output(wait=True)
            if folder_name == 'Select directory':
                return
            folder_path = os.path.join(root_dir, folder_name)
            image_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(extensions))[:max_images]
            true_idx = true_label_fn(folder_name) if true_label_fn else None
            fig, axes = plt.subplots(*grid, figsize=(3.6 * grid[1], 5 * grid[0]))
            fig.suptitle(f'Predictions for: {folder_name}', fontsize=18, y=0.98)
            model.eval()
            for i, ax in enumerate(np.atleast_1d(axes).ravel()):
                ax.axis('off')
                if i >= len(image_files):
                    continue
                img_path = os.path.join(folder_path, image_files[i])
                with torch.no_grad():
                    pred_idx = model(preprocess_fn(img_path, device)).argmax(dim=1).item()
                ax.imshow(Image.open(img_path).convert('RGB'))
                if true_idx is not None:
                    ax.set_title(f'True: {class_names[true_idx]}', fontsize=13, pad=25)
                color = 'black' if true_idx is None else ('green' if pred_idx == true_idx else 'red')
                ax.text(0.5, 1.05, f'Pred: {class_names[pred_idx]}', color=color, fontsize=13, weight='bold',
                        ha='center', transform=ax.transAxes)
            plt.subplots_adjust(hspace=0.5)
            plt.tight_layout()
            plt.show()
    dropdown.observe(update_grid, names='value')
    display(dropdown, output_widget)


# Returns the fraction of pixels whose HSV color lies inside a range
def calculate_color_fraction(image_array, lower_hsv=(35, 40, 40), upper_hsv=(85, 255, 255)):
    """
    Returns the fraction of pixels whose HSV color lies inside a range.

    Args:
        image_array: RGB uint8 image of shape (H, W, 3).
        lower_hsv: Lower (H, S, V) bound. The defaults select green.
        upper_hsv: Upper (H, S, V) bound.
    """
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, np.array(lower_hsv), np.array(upper_hsv))
    total_pixels = image_array.shape[0] * image_array.shape[1]
    return np.sum(mask == 255) / total_pixels if total_pixels > 0 else 0


# Classifies whether the share of a color grew, shrank or stayed similar between two images
def classify_color_change(input_before, input_after, threshold_percent, lower_hsv=(35, 40, 40), upper_hsv=(85, 255, 255),
                          mean=None, std=None, labels=('Positive', 'Negative', 'No_Change')):
    """
    Classifies whether the share of one color grew, shrank or stayed similar between two images.

    Args:
        input_before: File path or normalized image tensor of the first image.
        input_after: File path or normalized image tensor of the second image.
        threshold_percent: Minimum change in percentage points that counts as a change.
        lower_hsv: Lower (H, S, V) bound of the color. The defaults select green.
        upper_hsv: Upper (H, S, V) bound of the color.
        mean: Per-channel mean used for normalization. Required when tensors are passed.
        std: Per-channel std used for normalization. Required when tensors are passed.
        labels: Names returned for (increase, decrease, no change).

    Returns:
        One of labels, or an error string when a file is missing.
    """
    if isinstance(input_before, str):
        try:
            before = np.array(Image.open(input_before).convert('RGB'))
            after = np.array(Image.open(input_after).convert('RGB'))
        except FileNotFoundError:
            return 'Error: File not found'
    elif isinstance(input_before, torch.Tensor):
        before = deprocess_image(input_before, mean, std, to_uint8=True)
        after = deprocess_image(input_after, mean, std, to_uint8=True)
    else:
        raise TypeError('Inputs must be either file paths (str) or PyTorch tensors.')
    percent_before = calculate_color_fraction(before, lower_hsv, upper_hsv)
    change = calculate_color_fraction(after, lower_hsv, upper_hsv) - percent_before
    threshold = threshold_percent / 100.0
    if change >= threshold:
        return labels[0]
    if change <= -threshold and percent_before >= threshold:
        return labels[1]
    return labels[2]


# One-shot check of whether two images match, based on embedding distance
def verify_image_pair(model, reference_path, test_path, threshold, transform, device, embed_fn=None,
                      match_label='Match', mismatch_label='Mismatch', titles=('Reference image', 'Image to verify')):
    """
    One-shot check of whether two images belong together, based on embedding distance.

    Args:
        model: Embedding model.
        reference_path: Path of the known image.
        test_path: Path of the image to check.
        threshold: Distance below which the pair counts as a match.
        transform: Transform that turns a PIL image into a tensor.
        device: Device to run on.
        embed_fn: Function (model, batch) -> embeddings. Defaults to model.get_embedding.
        match_label: Text printed for a match.
        mismatch_label: Text printed for a mismatch.
        titles: Titles of the two displayed images.
    """
    embed_fn = embed_fn or (lambda m, x: m.get_embedding(x))
    model.eval()
    try:
        ref = transform(Image.open(reference_path).convert('RGB')).unsqueeze(0).to(device)
        test = transform(Image.open(test_path).convert('RGB')).unsqueeze(0).to(device)
    except FileNotFoundError as e:
        print(f'Error loading image: {e}')
        return
    with torch.no_grad():
        distance = F.pairwise_distance(embed_fn(model, ref), embed_fn(model, test)).item()
    print('--- Verification Result ---')
    print(f'Distance: {distance:.4f}')
    print(f'Decision Threshold: {threshold:.4f}')
    print(f"Prediction: {match_label if distance < threshold else mismatch_label}\n")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Distance: {distance:.4f}', fontsize=16)
    for ax, path, title in zip(axes, (reference_path, test_path), titles):
        ax.imshow(Image.open(path))
        ax.set_title(title)
        ax.axis('off')
    plt.show()


# Predicts whether two images differ from embedding distance and an optional classifier
def predict_pair_change(model, before_path, after_path, threshold, transform, device, pair_forward_fn=None,
                        classify_fn=None, titles=('Before', 'After'), same_label='No Change'):
    """
    Predicts whether two images differ, using embedding distance and an optional second classifier.

    Args:
        model: Pair model.
        before_path: Path of the first image.
        after_path: Path of the second image.
        threshold: Distance at or below which the pair counts as unchanged.
        transform: Transform that turns a PIL image into a tensor.
        device: Device to run on.
        pair_forward_fn: Function (model, a, b) -> (embedding_a, embedding_b). Defaults to model(a, b).
        classify_fn: Optional function (before_path, after_path) -> str that names the kind of change.
            Called only when the distance is above the threshold. 'Change' is used without it.
        titles: Titles of the two displayed images.
        same_label: Prediction text for an unchanged pair.
    """
    pair_forward_fn = pair_forward_fn or (lambda m, a, b: m(a, b))
    model.eval()
    try:
        img_a = transform(Image.open(before_path).convert('RGB')).unsqueeze(0).to(device)
        img_b = transform(Image.open(after_path).convert('RGB')).unsqueeze(0).to(device)
    except FileNotFoundError as e:
        print(f'Error loading image: {e}')
        return
    with torch.no_grad():
        emb_a, emb_b = pair_forward_fn(model, img_a, img_b)
        distance = F.pairwise_distance(emb_a, emb_b).item()
    if distance <= threshold:
        prediction = same_label
    else:
        prediction = classify_fn(before_path, after_path) if classify_fn else 'Change'
    print('--- Change Detection Result ---')
    print(f'Model Distance: {distance:.4f}')
    print(f'Decision Threshold: {threshold:.4f}')
    print(f'Prediction: {prediction}\n')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Prediction: {prediction} (Distance: {distance:.4f})', fontsize=14)
    for ax, path, title in zip(axes, (before_path, after_path), titles):
        ax.imshow(Image.open(path))
        ax.set_title(title)
        ax.axis('off')
    plt.show()



# =============================================================================
# PREDICTION AND FEATURE VISUALISATION
# =============================================================================


# Shows a grid of images with true and predicted labels
def visualize_predictions(model, data_loader, device, class_names):
    """
    Creates a visual grid of images showing their true and predicted class labels.

    Args:
        model: The trained model to be used for making predictions.
        data_loader: An object that loads the dataset in batches.
        device: The computing device on which to run the model.
        class_names: A list of class names that correspond to the labels.
    """

    # Set the model to evaluation mode to disable layers like dropout.
    model.eval()
    # Move the model to the specified device.
    model.to(device)

    # Define the pre-calculated dataset mean.
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    # Define the pre-calculated dataset standard deviation.
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])

    # Initialize lists to store all images and labels for each class.
    all_class_0_data = []
    all_class_1_data = []

    # Disable gradient calculations for faster inference.
    with torch.no_grad():
        # Iterate through the entire data loader to collect all images and their labels.
        # This is necessary for true randomness, as it ensures all images are in the pool for sampling.
        for images, labels in data_loader:
            # Move the image and label data to the specified device.
            images, labels = images.to(device), labels.to(device)

            # Find and collect all images belonging to class 0 from the batch.
            indices_class_0 = (labels == 0).nonzero(as_tuple=True)[0]
            for idx in indices_class_0:
                all_class_0_data.append((images[idx], labels[idx]))
            
            # Find and collect all images belonging to class 1 from the batch.
            indices_class_1 = (labels == 1).nonzero(as_tuple=True)[0]
            for idx in indices_class_1:
                all_class_1_data.append((images[idx], labels[idx]))
    
    # Randomly sample three images from the complete list of images for each class.
    sampled_class_0_data = random.sample(all_class_0_data, 3)
    # Randomly sample three images from the complete list of images for each class.
    sampled_class_1_data = random.sample(all_class_1_data, 3)
    
    # Combine the sampled images and labels from both classes.
    all_data = sampled_class_0_data + sampled_class_1_data
    all_images = torch.stack([item[0] for item in all_data])
    all_labels = torch.stack([item[1] for item in all_data])

    # Pass the collected images through the model to get the outputs.
    outputs = model(all_images)
    # Get the predicted class with the highest probability.
    _, predicted = torch.max(outputs, 1)

    # Move the tensors back to the CPU for plotting.
    images = all_images.cpu()
    # Move the true labels back to the CPU.
    labels = all_labels.cpu()
    # Move the predicted labels back to the CPU.
    predicted = predicted.cpu()

    # Create a new figure with a specified size.
    plt.figure(figsize=(10, 10))
    # Loop through a fixed number of images to display.
    for i in range(6):
        # Create a subplot within the figure.
        plt.subplot(2, 3, i + 1)
        # Change the tensor's dimension order for image display and convert to a numpy array.
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalize the image using the dataset mean and standard deviation.
        img = imagenet_std.numpy() * img + imagenet_mean.numpy()
        # Clip the image pixel values to ensure they are within a valid range.
        img = np.clip(img, 0, 1)
        # Display the image.
        plt.imshow(img)

        # Retrieve the true class name for the current image.
        true_label = class_names[labels[i]]
        # Retrieve the predicted class name for the current image.
        pred_label = class_names[predicted[i]]

        # Set the title color to green for a correct prediction and red otherwise.
        color = "green" if true_label == pred_label else "red"
        # Set the title of the subplot with true and predicted labels.
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=12)
        # Hide the axis.
        plt.axis("off")
    # Adjust the layout to prevent titles and labels from overlapping.
    plt.tight_layout(h_pad=0.5)
    # Display the final plot.
    plt.show()


# Shows model predictions for one random image from each class
def visualize_predictions_per_class(model, dataloader, class_names, device, mean=None, std=None):
    """Visualizes model predictions on a sample of images from a dataset.

    This function randomly selects one image from each class in the provided
    dataloader. It then performs inference using the given model and displays
    the images in a grid. Each image is titled with its true and predicted
    labels, colored green for correct predictions and red for incorrect ones.

    Args:
        model (torch.nn.Module): The trained PyTorch model to use for inference.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the dataset to visualize.
        class_names (list of str): A list mapping class indices to their names.
        device (torch.device): The device (e.g., 'cuda', 'cpu') on which to perform inference.
        mean: Optional per-channel mean used for normalization. Images are shown as is when None.
        std: Optional per-channel std used for normalization.
    """
    # Prepare the model for inference.
    model.to(device)
    model.eval()

    # --- Create a mapping from class index to a list of sample indices for that class ---
    # Initialize a dictionary to hold indices for each class.
    class_to_indices = _group_indices_by_class(dataloader.dataset, len(class_names))
    # ---

    # Create a grid of subplots to display the images.
    ncols = min(7, len(class_names))
    nrows = (len(class_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.6 * ncols, 2.7 * nrows), squeeze=False)

    # Disable gradient computations for the inference phase.
    with torch.no_grad():
        # Loop through each class and its corresponding subplot axis.
        for i, ax in enumerate(axes.flatten()):
            # If there are more subplots than classes, turn off the extra ones.
            if i >= len(class_names):
                ax.axis('off')
                continue

            # Randomly select one image index from the current class.
            random_image_idx = random.choice(class_to_indices[i])
            
            # Get the image tensor and its true label from the dataset.
            image_tensor, true_label = dataloader.dataset[random_image_idx]
            
            # Prepare the image tensor for the model by adding a batch dimension and moving it to the device.
            image_batch = image_tensor.unsqueeze(0).to(device)

            # Pass the image through the model to get the output logits.
            outputs = model(image_batch)
            # Determine the predicted class index by finding the index of the maximum logit.
            _, pred = torch.max(outputs, 1)
            predicted_label = pred.item()
            
            # Set the title color to green for correct predictions and red for incorrect ones.
            is_correct = (predicted_label == true_label)
            title_color = 'green' if is_correct else 'red'
            # Set the subplot's title with the predicted and true labels.
            ax.set_title(
                f'Predicted: {class_names[predicted_label]}\n(True: {class_names[true_label]})',
                color=title_color
            )
            
            # Reverse the normalization of the image tensor for proper visualization.
            img_to_plot = unnormalize(image_tensor, mean, std) if mean is not None and std is not None else image_tensor
            
            # Convert the tensor to a NumPy array and adjust dimensions for displaying.
            ax.imshow(np.transpose(img_to_plot.numpy(), (1, 2, 0)))
            # Display the image and hide the axis ticks.
            ax.axis('off')

    # Adjust the layout to prevent titles from overlapping and show the plot.
    plt.tight_layout()
    plt.show()


# Displays one sample image for each class from the validation data
def display_random_predictions_per_class(model, val_loader, classes, device, num_classes=7):
    """
    Displays one sample image for each class from the validation data.

    Args:
        model: The trained model to use for predictions.
        val_loader: DataLoader containing the validation dataset.
        classes: A list of class names corresponding to the model's output.
        device: The device (e.g., 'cuda' or 'cpu') to run inference on.
        num_classes: The total number of classes to look for.
    """
    # Set the model to evaluation mode (e.g., disable dropout)
    model.eval()
    # Move the model to the specified computation device
    model.to(device)

    # Dictionary to store one image tensor for each class index
    class_images = defaultdict(list)
    # Set to keep track of which class indices have been found
    found_classes = set()

    # Iterate through the validation loader to find one image per class
    # Disable gradient calculations during inference
    with torch.no_grad():
        # Loop over batches of images and labels
        for images, labels in val_loader:
            # Move data to the specified device
            images, labels = images.to(device), labels.to(device)

            # Check each image in the batch
            for i in range(images.size(0)):
                # Get the integer label for the current image
                label_idx = labels[i].item()
                # Check if this class is needed and hasn't been found yet
                if label_idx not in found_classes and label_idx < num_classes :
                    # Store the image tensor (move to CPU for storage/plotting)
                    class_images[label_idx].append(images[i].cpu())
                    # Mark this class index as found
                    found_classes.add(label_idx)

            # If an image for every class has been found, stop iterating
            if len(found_classes) == num_classes:
                # Exit the loop
                break
        # This 'else' block runs if the 'for' loop completes without breaking
        else:
            # Print a warning if not all classes were found in the dataset
            print(f"Warning: Could only find images for {len(found_classes)} out of {num_classes} classes.")


    # --- Prepare images and predictions for display ---
    # List to hold the image tensors for plotting
    selected_images = []
    # List to hold the true class names
    true_labels = []
    # List to hold the predicted class names
    pred_labels = []

    # Get a sorted list of the class indices that were actually found
    sorted_class_indices = sorted([idx for idx in class_images.keys() if idx < num_classes])

    # Perform inference on the selected images
    with torch.no_grad():
        # Loop through the found class indices in order
        for class_idx in sorted_class_indices:
            # Get the first image stored for this class
            img_tensor = class_images[class_idx][0]
            # Add it to the list for plotting
            selected_images.append(img_tensor)

            # Prepare the image for the model (add batch dimension and move to device)
            img_tensor_batch = img_tensor.unsqueeze(0).to(device)
            # Get the model's raw output (logits)
            output = model(img_tensor_batch)
            # Find the index with the highest score (the predicted class)
            _, predicted_idx = torch.max(output, 1)

            # Store the corresponding class names for true and predicted labels
            true_labels.append(classes[class_idx])
            pred_labels.append(classes[predicted_idx.item()])

    # --- Display the images and labels using subplots ---
    # Check if any images were actually selected
    if not selected_images:
        # Inform user if no images are available
        print("No images were selected to display.")
        # Exit the function
        return

    # Get the number of images that will be plotted
    num_found = len(selected_images)
    # Create a figure and a set of subplots (one for each image)
    fig, axes = plt.subplots(1, num_found, figsize=(3 * num_found, 4.5))

    # Handle the case where only one image is found (axes is not an array)
    if num_found == 1:
        # Wrap the single axis in a list to allow iteration
        axes = [axes]

    # Iterate through the selected images and their labels
    for i in range(num_found):
        # Get the specific axis for this plot
        ax = axes[i]
        # Denormalize the image (assuming normalization was mean=0.5, std=0.5)
        img = selected_images[i] * 0.5 + 0.5
        # Convert tensor from (Channel, Height, Width) to (Height, Width, Channel) for plotting
        img = np.transpose(img.numpy(), (1, 2, 0))
        # Display the image
        ax.imshow(img)

        # Determine the color for the prediction text (green for correct, red for incorrect)
        pred_color = 'green' if true_labels[i] == pred_labels[i] else 'red'

        # Set the title to show the true and predicted labels, coloring the prediction
        ax.set_title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}", fontsize=13, color=pred_color)

        # Hide the x and y axes
        ax.axis('off')

    # Adjust the layout to prevent titles and images from overlapping
    plt.tight_layout(pad=1.5)
    # Show the plot
    plt.show()


# Displays a grid of images along with their ground truth and predicted labels
def show_prediction_grid(images, labels, predictions, classes):
    """
    Displays a grid of images along with their ground truth and predicted labels.
    
    Args:
        images (numpy.ndarray): A batch of image tensors or arrays to be displayed.
        labels (list/numpy.ndarray): The actual class indices for the given images.
        predictions (list/numpy.ndarray): The raw model outputs or probabilities for each image.
        classes (list): A list of strings representing the human-readable class names.

    Returns:
        None: This function renders a matplotlib plot and does not return any value.
    """
    # Set the mean values used during the normalization process for reversal
    mean = np.array([0.630, 0.554, 0.489])
    # Set the standard deviation values used during the normalization process for reversal
    std = np.array([0.248, 0.271, 0.319])
    
    # Initialize a figure with a 3 by 3 subplot arrangement for visualization
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    # Adjust the spacing between subplots to ensure labels are legible
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    
    # Iterate through the flattened array of subplot axes
    for i, ax in enumerate(axes.flat):
        # Ensure the current index does not exceed the number of images in the batch
        if i < len(images):
            # Change the image dimension order from (C, H, W) to (H, W, C) for matplotlib compatibility
            img = images[i].transpose((1, 2, 0))
            # Reverse the normalization by multiplying by standard deviation and adding the mean
            img = std * img + mean
            # Clamp the pixel values to the valid range of 0 to 1
            img = np.clip(img, 0, 1)
            
            # Render the processed image in the current subplot
            ax.imshow(img)
            
            # Map the true label index to its corresponding class name string
            true_label = classes[labels[i]]
            # Map the predicted label (index with the highest value) to its class name string
            pred_label = classes[np.argmax(predictions[i])]
            
            # Apply a title to the subplot displaying both labels with conditional formatting
            # Titles appear green if the prediction is correct and red if it is incorrect
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label}",
                color=("green" if true_label == pred_label else "red")
            )
            
            # Disable horizontal axis tick marks for a cleaner display
            ax.set_xticks([])
            # Disable vertical axis tick marks for a cleaner display
            ax.set_yticks([])
            
    # Display the final generated grid of predictions
    plt.show()


# Shows an image next to feature map grids from several CNN layers
def display_feature_hierarchy(activations, img):
    """
    Visualizes the original image alongside a grid of feature maps from 
    selected layers of a convolutional neural network.

    This function de-normalizes the input image for display and creates a 
    multi-panel plot showing the original image and the aggregated feature 
    activations for specific layers (conv1 through layer4).

    Arguments:
        activations: A dictionary containing feature map tensors for specific layers.
        img: The preprocessed input image tensor (normalized).
    """
    # Create the main figure with specific dimensions
    plt.figure(figsize=(15, 10))

    # Move the image tensor to CPU and prepare for de-normalization
    orig = img[0].cpu()
    
    # Reverse the standard ImageNet normalization (multiply by std)
    orig = orig * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Reverse the standard ImageNet normalization (add mean)
    orig = orig + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    
    # Create the first subplot for the original image
    plt.subplot(2, 3, 1)
    
    # Permute tensor dimensions from (C, H, W) to (H, W, C) and clamp values to [0, 1] for display
    plt.imshow(orig.permute(1, 2, 0).clamp(0, 1))
    
    # Set the title for the original image plot
    plt.title("Original Image")
    
    # Hide the axis ticks and labels
    plt.axis("off")

    # Define the specific order of layers to visualize
    order = ["conv1", "layer1", "layer2", "layer3", "layer4"]
    
    # Iterate through the layers to create subplots, starting from the second position
    for sp, name in enumerate(order, start=2):
        # Generate a grid visualization from the layer's activation tensor
        grid = grid_from_layer(activations[name])
        
        # Select the specific subplot position
        plt.subplot(2, 3, sp)
        
        # Display the feature grid using the viridis colormap
        plt.imshow(grid, cmap="viridis")
        
        # Set the title including the layer name and number of filters
        plt.title(f"{name}: {activations[name].shape[1]} filters")
        
        # Hide the axis ticks and labels
        plt.axis("off")

    # Adjust the layout to prevent overlapping elements
    plt.tight_layout()
    
    # Render the final figure
    plt.show()


# Visualizes the activations of all layers in a grid format
def visualize_all_layers_grids(activations):
    """
    Visualizes the activations of all layers in a grid format.

    Args:
        activations (dict): A dictionary where keys are layer names and values are activation tensors.
    """
    n_layers = len(activations)
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 8))  # 1 column, n_layers rows

    if n_layers == 1:
        axes = [axes]
        
    for idx, (layer_name, output) in enumerate(activations.items()):
        acts = output[0]  # (C, H, W)
        n_filters = acts.shape[0]
        grid_size = int(np.ceil(np.sqrt(n_filters)))

        # Create a black canvas for the grid
        h, w = acts.shape[1], acts.shape[2]
        grid = np.zeros((h * grid_size, w * grid_size))

        for i in range(n_filters):
            row = i // grid_size
            col = i % grid_size
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = acts[i].detach().cpu()

        axes[idx].imshow(grid, cmap='gray')
        axes[idx].set_title(layer_name)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


# Plots a summary of the receptive field size and feature map size per layer
def plot_receptive_field_summary(rfinfo, figwidth=14):
    """
    Plots a summary of the receptive field size and feature map size per layer.

    Args:
        rfinfo (list): A list of dictionaries containing receptive field and feature map information.
        figwidth (int, optional): The width of the figure. Defaults to 14.
    """
    names = [x['layer'] for x in rfinfo]
    rf = np.array([x['rf'] for x in rfinfo])
    msize = np.array([x['map_size'] for x in rfinfo])

    # Subsample for key layers
    to_plot_idx = [0]
    for i in range(1, len(names)):
        if msize[i] < msize[i-1] or i == len(names)-1 or i % 6 == 0:
            to_plot_idx.append(i)
    to_plot_idx = sorted(set(to_plot_idx))
    names_short = []
    for idx in to_plot_idx:
        n = names[idx]
        n = '.'.join(n.split('.')[-3:]).replace(' (Conv2d)','').replace(' (MaxPool2d)','').replace(' (AvgPool2d)','')
        names_short.append(n)
    rf_plot = rf[to_plot_idx]
    msize_plot = msize[to_plot_idx]

    fig, ax1 = plt.subplots(figsize=(figwidth, 5))
    line_rf, = ax1.plot(rf_plot, '-o', color='tab:blue', label='Receptive Field')
    ax2 = ax1.twinx()
    line_ms, = ax2.plot(msize_plot, '--s', color='tab:red', label='Feature Map Size')

    # Downsampling points
    for i in range(1, len(msize_plot)):
        if msize_plot[i] < msize_plot[i-1]:
            ax2.plot(i, msize_plot[i], 'v', color='darkred', markersize=9, zorder=3)
            ax2.annotate(
                f"{msize_plot[i]}",
                xy=(i, msize_plot[i]), xytext=(0, 16), textcoords='offset points',
                ha='center', color='darkred', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none"), zorder=5
            )

    ax1.set_xticks(range(len(names_short)))
    ax1.set_xticklabels(names_short, rotation=50, ha='right', fontsize=9)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Receptive Field', color='tab:blue')
    ax2.set_ylabel('Feature Map Size', color='tab:red')
    plt.title("CNN Receptive Field and Feature Map Size per Layer (summary)")
    ax1.grid(axis='y', linestyle=':')
    plt.tight_layout(rect=[0,0,0.75,1])  # Make wide room for legend

    # --- Attach legend to the figure, not axes ---
    fig.legend([line_rf, line_ms], ["Receptive Field", "Feature Map Size"],
               loc="center left", bbox_to_anchor=(0.78, 0.5), frameon=True)
    plt.show()


# Visualizes receptive field overlays for certain layers given an image
def plot_rfinfo_over_image(rfinfo, img_path, input_size=224, layers_to_show=None, figsize=(18, 8)):
    """
    Visualizes receptive field overlays for certain layers given an image.

    Args:
        rfinfo (list): Output from calculate_receptive_field(model, input_size).
        img_path (str): The filename/path of the image.
        input_size (int, optional): Input spatial size (should match what was passed to calculate_receptive_field). Defaults to 224.
        layers_to_show (list, optional): List of layer names to show. If None, selects default interesting layers. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (18, 8).
    """
    orig_img = Image.open(img_path).convert("RGB")
    img_resized = orig_img.resize((input_size, input_size))
    image_np = np.array(img_resized)
    all_layer_names = [item["layer"] for item in rfinfo]

    if layers_to_show is None:
        # You may adjust this selection logic for your model!
        candidates = []
        for name in [
            "conv1 (Conv2d)",
            "maxpool (MaxPool2d)",
            "layer1.0.conv1 (Conv2d)",
            "layer1.2.conv3 (Conv2d)",
            "layer2.0.conv3 (Conv2d)",
            "layer3.3.conv3 (Conv2d)",
            "layer4.0.conv3 (Conv2d)",
            "layer4.2.conv3 (Conv2d)",
        ]:
            candidates.extend([n for n in all_layer_names if name in n])
        layers_to_show = candidates[:7]  # adjust how many layers you want to show

    def plot_one(info, subplot_idx, image, input_size=224):
        feature_size = info["map_size"]
        rf_size = info["rf"]
        name = info["layer"]

        # Special case: no grid or overlay for first full-size layer (usually conv1)
        if feature_size == input_size and rf_size == 1:
            plt.subplot(2, 4, subplot_idx)
            plt.imshow(image.astype(np.uint8))
            plt.title(
                f"{name}\nFeature Map: {feature_size}×{feature_size}\nReceptive Field: {rf_size}×{rf_size}"
            )
            plt.axis("off")
            return

        img = image.copy().astype(np.float32) / 255.0
        # Draw grid for other layers
        if feature_size > 1:
            cell_size = input_size / feature_size
            for i in range(feature_size):
                y = int(i * cell_size)
                if y < input_size:
                    img[y : y + 1, :, :] = [0, 0, 0]
                x = int(i * cell_size)
                if x < input_size:
                    img[:, x : x + 1, :] = [0, 0, 0]
        # Draw receptive field patch
        center = input_size // 2
        rf_half = min(rf_size // 2, input_size // 2)
        if rf_half > 0:
            start_x = max(0, center - rf_half)
            end_x = min(input_size, center + rf_half)
            start_y = max(0, center - rf_half)
            end_y = min(input_size, center + rf_half)
            img[start_y:end_y, start_x:end_x, 0] = 1.0
            img[start_y:end_y, start_x:end_x, 1] *= 0.7
            img[start_y:end_y, start_x:end_x, 2] *= 0.7
            thickness = max(1, input_size // 100)
            if start_y < end_y:
                img[start_y : start_y + thickness, start_x:end_x] = [1, 0, 0]
                img[end_y - thickness : end_y, start_x:end_x] = [1, 0, 0]
            if start_x < end_x:
                img[start_y:end_y, start_x : start_x + thickness] = [1, 0, 0]
                img[start_y:end_y, end_x - thickness : end_x] = [1, 0, 0]
        plt.subplot(2, 4, subplot_idx)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(
            f"{name}\nFeature Map: {feature_size}×{feature_size}\nReceptive Field: {rf_size}×{rf_size}"
        )
        plt.axis("off")

    plt.figure(figsize=figsize)
    # First: original image
    plt.subplot(2, 4, 1)
    plt.imshow(image_np)
    plt.title(f"Original Image\n({input_size}×{input_size})")
    plt.axis("off")
    # Others: overlays for selected layers
    for i, name in enumerate(layers_to_show):
        matches = [item for item in rfinfo if item["layer"] == name]
        if not matches:
            print(f"(Warning: Layer '{name}' not found in rfinfo.)")
            continue
        plot_one(matches[0], subplot_idx=i + 2, image=image_np, input_size=input_size)
    plt.suptitle("Receptive Field and Feature Map Visualization")
    plt.tight_layout()
    plt.show()


# Overlays a saliency heatmap onto the original image for visualization
def display_saliency(image_tensor, heatmap, alpha=0.5):
    """
    Overlays a saliency heatmap onto the original image for visualization.

    This function denormalizes the input image (assuming ImageNet statistics),
    clamps pixel values, and displays the heatmap using a jet colormap
    with transparency.

    Arguments:
        image_tensor: The input image tensor (1, 3, H, W).
        heatmap: The 2D saliency map tensor (H, W).
        alpha: The transparency factor for the heatmap overlay (0.0 to 1.0).
    """
    # Define the mean used for ImageNet normalization for denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # Define the standard deviation used for ImageNet normalization
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize the image tensor by multiplying by std and adding mean
    img_disp = image_tensor.detach()[0].cpu() * std + mean
    
    # Clamp values to [0, 1] range, rearrange dimensions to (H, W, C), and convert to numpy
    img_disp = img_disp.clamp(0, 1).permute(1, 2, 0).numpy()

    # Initialize the plot figure with specific dimensions
    plt.figure(figsize=(6, 6))
    
    # Display the base image
    plt.imshow(img_disp)
    
    # Overlay the heatmap with the specified colormap and transparency
    plt.imshow(heatmap.cpu(), cmap="jet", alpha=alpha)
    
    # Hide the axis ticks and labels
    plt.axis("off")
    
    # Set the title for the visualization
    plt.title("Salience Map Overlay")
    
    # Render the final plot
    plt.show()


# Overlays a Class Activation Map (CAM) onto the original image
def display_cam(image_tensor, cam_up):
    """
    Overlays a Class Activation Map (CAM) onto the original image.

    This function prepares the input image by reversing the normalization 
    process (assuming ImageNet statistics), and then renders the image 
    with the CAM heatmap superimposed to visualize regions of interest.

    Arguments:
        image_tensor: The input image tensor (1, 3, H, W).
        cam_up: The upsampled CAM heatmap tensor (H, W).
    """
    # Define the mean values used for ImageNet normalization
    means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # Define the standard deviation values used for ImageNet normalization
    stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize the image: multiply by std, add mean, clamp to [0, 1], and rearrange to (H, W, C)
    rgb = (image_tensor[0].cpu() * stds + means).clamp(0, 1).permute(1, 2, 0).numpy()

    # Initialize the figure with specific dimensions
    plt.figure(figsize=(6, 6))
    
    # Display the base RGB image
    plt.imshow(rgb)
    
    # Overlay the CAM heatmap using the jet colormap and 50% transparency
    plt.imshow(cam_up.cpu().detach().cpu(), cmap="jet", alpha=0.5)
    
    # Hide the axis ticks and labels
    plt.axis("off")
    
    # Set the title for the visualization
    plt.title("Simplified CAM Overlay")
    
    # Render the final figure
    plt.show()


# Interactive dropdown that runs Grad-CAM on the chosen image
def plot_gradcam_widget(compute_gradcam, visualize_gradcam, model, transform, device, folder="images"):
    """
    Create and display an interactive widget to visualize GradCAM results for images.

    This function scans a directory for images and generates a dropdown menu. 
    Selecting an image triggers the computation and display of the GradCAM heatmap 
    using the provided model and visualization functions.

    Args:
        compute_gradcam: A callable that takes an image path, model, transform, and device, returning results.
        visualize_gradcam: A callable that displays the image, heatmap, and prediction details.
        model: The pre-trained PyTorch model to use for inference.
        transform: The preprocessing transform pipeline to apply to input images.
        device: The compute device (CPU or GPU) where the model is loaded.
        folder: The directory path containing the input images.
    """
    def get_jpg_files(folder=folder):
        """
        Retrieve a sorted list of JPEG filenames from the specified directory.

        Args:
            folder: The directory path to search for images.

        Returns:
            A list of strings representing filenames ending in .jpg.
        """
        # Check if the specified folder exists
        if not os.path.exists(folder):
            return []
        # Return a sorted list of files that match the jpg extension, case-insensitive
        return sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])

    # Initialize a widget output area to capture and display plots
    out = widgets.Output()

    def gradcam_widget_view(image_name):
        """
        Handle the update logic when a new image is selected from the dropdown.

        This callback runs the GradCAM computation pipeline and updates the 
        output widget with the new visualization.

        Args:
            image_name: The name of the selected image file.
        """
        # Construct the full file path for the selected image
        img_path = os.path.join(folder, image_name)
        # Use the output widget context to capture prints and plots
        with out:
            # Clear the previous output to prepare for the new plot
            clear_output(wait=True)
            # Display a status message indicating processing has started
            print(f"Showing GradCAM for: {image_name}")
            # Run the provided GradCAM computation function
            img_display, heatmap, pred_class, pred_score = compute_gradcam(
                img_path, model, transform, device
            )
            # Verify if the image processing was successful
            if img_display is not None:
                # Extract the filename without extension to use as a title
                title = os.path.splitext(image_name)[0]
                # Render the visualization using the provided helper function
                visualize_gradcam(img_display, heatmap, pred_class, pred_score, title)
            else:
                # Log an error message if processing failed
                print(f"Could not process {image_name}.")

    # Fetch the list of available images from the folder
    jpg_list = get_jpg_files(folder)
    # Check if the list is empty and display a warning if so
    if not jpg_list:
        display(widgets.HTML(
            value=f"<b style='color: red;'>No .jpg files found in the <code>{folder}/</code> folder. Please add images and rerun.</b>"
        ))
    else:
        # Create a dropdown menu widget with the list of image files
        dropdown = widgets.Dropdown(
            options=jpg_list,
            description='Select image:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%'),  # Set width for better visibility
            disabled=False
        )
        # Display the header instructions
        display(widgets.HTML(f"<h3>GradCAM Visualizer</h3>Select an image from <code>{folder}</code>:"))
        # Link the dropdown widget to the callback function to trigger updates
        widgets.interact(gradcam_widget_view, image_name=dropdown)
        # Display the output container where the plots will appear
        display(out)


# Shows predictions for one random test image from each class
def display_predictions(model, test_loader, device, num_classes=None, grid=None, class_names=None, cmap='gray'):
    """
    Shows model predictions for one random test image from each class.

    Args:
        model: Classifier that returns one score per class.
        test_loader: DataLoader whose dataset yields (image, label).
        device: Device to run the model on.
        num_classes: Only classes 0..num_classes-1 are shown. All classes found when None.
        grid: Optional (rows, cols). Defaults to rows of up to 5 images.
        class_names: Optional list of names used in the titles instead of indices.
        cmap: Colormap used for single-channel images.
    """
    model.to(device)
    model.eval()
    dataset = test_loader.dataset
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[int(label)].append(idx)
    classes = sorted(class_indices) if num_classes is None else [c for c in range(num_classes) if class_indices[c]]
    chosen = [random.choice(class_indices[c]) for c in classes]
    sample_images = torch.stack([dataset[i][0] for i in chosen])
    sample_labels = [int(dataset[i][1]) for i in chosen]
    with torch.no_grad():
        predictions = model(sample_images.to(device)).argmax(dim=1).cpu().tolist()
    n = len(chosen)
    rows, cols = grid if grid else (int(np.ceil(n / min(5, n))), min(5, n))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 3))
    fig.suptitle('Model Predictions for a Sample of Each Class', fontsize=16)
    name = (lambda i: class_names[i]) if class_names is not None else str
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i >= n:
            ax.axis('off')
            continue
        image = sample_images[i].cpu()
        image = image.squeeze(0) if image.shape[0] == 1 else image.permute(1, 2, 0)
        ax.imshow(image, cmap=cmap if image.ndim == 2 else None)
        color = 'green' if sample_labels[i] == predictions[i] else 'red'
        ax.set_title(f'True: {name(sample_labels[i])}\nPred: {name(predictions[i])}', color=color)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    plt.show()


# Shows predictions for one random image per class on a grid, with optional de-normalization
def visualise_predictions(model, data_loader, device, grid, mean=None, std=None, class_names=None):
    """
    Shows predictions for one random image from each class on a grid.

    Args:
        model: Classifier that returns one score per class.
        data_loader: DataLoader whose dataset has .targets and (when class_names is None) .classes.
        device: Device to run the model on.
        grid: (rows, cols) of the figure.
        mean: Optional per-channel mean used when the images were normalized. Skips de-normalizing when None.
        std: Optional per-channel std used when the images were normalized.
        class_names: Optional list of names. Defaults to dataset.classes.
    """
    model.eval()
    dataset = data_loader.dataset
    class_names = class_names if class_names is not None else dataset.classes
    class_indices = defaultdict(list)
    for idx, target in enumerate(dataset.targets):
        class_indices[target].append(idx)
    rows, cols = grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    plt.subplots_adjust(wspace=0.3, hspace=0.8)
    for i, ax in enumerate(axes.flat):
        if i >= rows * cols or i >= len(class_names) or not class_indices[i]:
            ax.axis('off')
            continue
        image_tensor, true_label = dataset[random.choice(class_indices[i])]
        with torch.no_grad():
            predicted_label = model(image_tensor.unsqueeze(0).to(device)).argmax(dim=1).item()
        img_np = image_tensor.cpu().numpy().transpose((1, 2, 0))
        if mean is not None and std is not None:
            img_np = np.array(std) * img_np + np.array(mean)
        title_color = 'green' if true_label == predicted_label else 'red'
        ax.imshow(np.clip(img_np, 0, 1))
        ax.set_title(f'True: {class_names[true_label].capitalize()}\nPred: {class_names[predicted_label].capitalize()}',
                     color=title_color, fontsize=10, pad=5)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Prints per-class accuracy and a confusion matrix for a trained model
def per_class_acc_and_conf_matrix(model, val_loader, class_names, device=None):
    """
    Prints per-class accuracy and plots the confusion matrix of a classifier.

    Args:
        model: Classifier that returns one score per class.
        val_loader: DataLoader yielding (images, labels).
        class_names: List of class names, indexed by class id.
        device: Device to run on. Uses CUDA when available and CPU otherwise when None.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating Model', leave=False):
            outputs = model(images.to(device))
            all_preds.append(torch.argmax(outputs, dim=1))
            all_labels.append(labels.to(device))
    confmat = MulticlassConfusionMatrix(num_classes=len(class_names)).to(device)
    cm = confmat(torch.cat(all_preds), torch.cat(all_labels))
    per_class_acc = cm.diag() / cm.sum(axis=1)
    print('--- Per-Class Accuracy Report ---')
    for i, acc in enumerate(per_class_acc):
        print(f"  - Accuracy for class '{class_names[i]}': {acc.item():.4f}")
    print()
    plot_confusion_matrix(cm.cpu().numpy(), class_names)


# Shows two random validation images per class with true and predicted labels
def display_random_predictions(model, dataset, class_names=None, device=None, per_class=2, mean=None, std=None,
                               title='Validation Image Predictions'):
    """
    Shows a few random images per class with their true and predicted labels, one row per class.

    Args:
        model: Classifier that returns one score per class.
        dataset: Dataset with .targets and (when class_names is None) .classes, yielding (image, label).
        class_names: Optional list of class names. Defaults to dataset.classes.
        device: Device to run on. Uses CUDA when available and CPU otherwise when None.
        per_class: Number of random images per class.
        mean: Optional per-channel mean to undo normalization. Skipped when None.
        std: Optional per-channel std to undo normalization.
        title: Figure title.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    class_names = class_names if class_names is not None else dataset.classes
    targets = np.array(dataset.targets)
    placements = []
    for class_idx, class_name in enumerate(class_names):
        indices = np.where(targets == class_idx)[0]
        if len(indices) < per_class:
            print(f"Warning: Class '{class_name}' has fewer than {per_class} images.")
            continue
        for col, idx in enumerate(random.sample(list(indices), per_class)):
            placements.append((class_idx, col, int(idx)))
    if not placements:
        print('Could not find enough images to display.')
        return
    images = torch.stack([dataset[idx][0] for _, _, idx in placements]).to(device)
    with torch.no_grad():
        preds = torch.argmax(model(images), dim=1).cpu()
    fig, axs = plt.subplots(len(class_names), per_class, figsize=(4 * per_class, 2.7 * len(class_names)), squeeze=False)
    fig.suptitle(title, fontsize=16)
    for ax in axs.ravel():
        ax.axis('off')
    for (row, col, idx), image, pred in zip(placements, images.cpu(), preds):
        img = image.permute(1, 2, 0).numpy()
        if mean is not None and std is not None:
            img = np.array(std) * img + np.array(mean)
        ax = axs[row, col]
        ax.imshow(np.clip(img, 0, 1))
        true_label, pred_label = class_names[row], class_names[pred]
        ax.text(0.5, 1.05, f'True: {true_label}', ha='center', transform=ax.transAxes, fontsize=12)
        ax.text(0.5, -0.1, f'Predicted: {pred_label}', ha='center', transform=ax.transAxes, fontsize=12,
                color='green' if true_label == pred_label else 'red')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1.5)
    plt.show()


# Plots the confusion matrix and classification report for a Siamese change detector
def plot_confusion_matrix_and_metrics(model, data_loader, threshold, device, pair_forward_fn=None, label_to_binary=None,
                                      target_names=('Same (Class 0)', 'Different (Class 1)')):
    """
    Evaluates a pair model with a distance threshold and shows a classification report and confusion matrix.

    Args:
        model: Pair model.
        data_loader: DataLoader that yields (input_a, input_b, label) batches.
        threshold: Pairs with a distance at or above it are predicted as 'different' (class 1).
        device: Device to run on.
        pair_forward_fn: Function (model, a, b) -> (embedding_a, embedding_b). Defaults to model(a, b).
        label_to_binary: Optional function that turns labels into 0 (same) and 1 (different). Identity when None.
        target_names: Names of the two classes.
    """
    pair_forward_fn = pair_forward_fn or (lambda m, a, b: m(a, b))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for input_a, input_b, labels in tqdm(data_loader, desc='Generating Predictions'):
            out_a, out_b = pair_forward_fn(model, input_a.to(device), input_b.to(device))
            distances = F.pairwise_distance(out_a, out_b)
            all_preds.extend((distances >= threshold).long().cpu().numpy())
            binary = label_to_binary(labels) if label_to_binary else labels
            all_labels.extend(binary.long().cpu().numpy())
    print('\n--- Classification Report ---')
    print(classification_report(all_labels, all_preds, target_names=list(target_names)))
    print(f"F1 Macro Score: {f1_score(all_labels, all_preds, average='macro'):.4f}\n")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=list(target_names), yticklabels=list(target_names))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


# Shows every channel of one layer's output as a grayscale image
def visualize_output_layer(layer, activations):
    """
    Shows every channel of one layer's output as a grayscale image.

    Args:
        layer: Key of the layer inside activations.
        activations: Dict mapping layer names to output tensors of shape (channels, H, W).
    """
    output_tensor = activations[layer]
    num_images = output_tensor.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_images)))
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(output_tensor[i].detach().numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Creates an interactive widget to visualize feature maps and model predictions
def plot_feature_map_widget(model, images_folder='images', labels_file=None, class_labels=None,
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), canvas_size=(512, 512)):
    """
    Creates an interactive widget to visualize feature maps and model predictions.
    
    This widget allows selecting images from a folder and viewing the feature maps
    of different layers of the model.

    Args:
        model (nn.Module): The PyTorch model to visualize.
        images_folder: Folder with the images that can be selected.
        labels_file: Optional text file with one class name per line, to name the predicted class.
        class_labels: Optional list of class names, used instead of labels_file.
        mean: Per-channel mean used to normalize the input. Defaults to ImageNet values.
        std: Per-channel std used to normalize the input. Defaults to ImageNet values.
        canvas_size: Size the feature map images are drawn at.
    """
    # Download once: (uncomment if not already downloaded)
    # import urllib.request
    #urllib.request.urlretrieve("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    
    if class_labels is None and labels_file is not None:
        with open(labels_file) as f:
            class_labels = [line.strip() for line in f.readlines()]
    imagenet_labels = class_labels or []
    
    # --------- CONFIG ---------
    INPUT_OPTION_NAME = "🟢 Input image"
    # --------------------------
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])
    
    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    
    feature_maps = {}
    layer_info_list = []  # (pretty_name, module_obj or None)
    
    def hook_fn(module, input, output):
        feature_maps[module] = output
    
    # Layer list (first entry is input image)
    layer_info_list.append((INPUT_OPTION_NAME, None))
    for top_name, layer in model.named_children():
        if isinstance(layer, torch.nn.Sequential):
            for subname, sublayer in layer.named_children():
                pretty_name = f"{top_name}.{subname}:{type(sublayer).__name__}"
                short_name = (pretty_name[:60] + '...') if len(pretty_name) > 63 else pretty_name
                layer_info_list.append((short_name, sublayer))
                sublayer.register_forward_hook(hook_fn)
        else:
            pretty_name = f"{top_name}:{type(layer).__name__}"
            short_name = (pretty_name[:60] + '...') if len(pretty_name) > 63 else pretty_name
            layer_info_list.append((short_name, layer))
            layer.register_forward_hook(hook_fn)
    
    def make_grid(tensor, n_cols=8):
        if tensor.dim() != 4:
            return None
        n_channels = tensor.shape[1]
        n_rows = (n_channels - 1) // n_cols + 1
        grid = np.zeros((n_rows * tensor.shape[2], n_cols * tensor.shape[3]))
        for i in range(n_channels):
            row, col = divmod(i, n_cols)
            grid[row * tensor.shape[2]: (row + 1) * tensor.shape[2],
                 col * tensor.shape[3]: (col + 1) * tensor.shape[3]] = tensor[0, i].detach().numpy()
        return grid
    
    output = widgets.Output()
    layer_params_html = widgets.HTML(
        value="",
        layout=widgets.Layout(
            min_width='320px', max_width='420px',
            font_family='monospace',
            font_size='1em',
            background_color='#181818',
            border='solid 1.5px #444',
            padding='10px 15px 10px 15px',
            color='#CCE4FF',
            overflow='auto'
        )
    )
    
    def update_plot(image_path, selected_layer_idx):
        with output:
            clear_output(wait=True)
            feature_maps.clear()
            img = Image.open(os.path.join(images_folder, image_path)).convert("RGB")
            img_t = preprocess(img)
            if selected_layer_idx == 0:  # Input Image selected
                input_img = img_t.permute(1, 2, 0).cpu().numpy()
                input_img = (input_img * np.array(std) + np.array(mean))
                input_img = np.clip(input_img, 0, 1)
                pil_img = PILImage.fromarray((input_img*255).astype(np.uint8))
                pil_img = pil_img.resize(canvas_size, PILImage.NEAREST)
                plt.figure(figsize=(7, 7))
                plt.imshow(pil_img)
                plt.title("Input image (after preprocessing & crop)", fontsize=16, pad=16)
                plt.axis('off')
                plt.tight_layout(pad=2)
                plt.show()
                # Params/info for input image
                layer_params_html.value = (
                    "<b>Input image</b><br><br>"
                    "<i>Displayed as passed into the model after resizing, cropping, "
                    "and normalization (using the given mean/std).</i>"
                )
                return
            # Forward pass
            batch_t = torch.unsqueeze(img_t, 0)
            output_tensor = model(batch_t)
            layer_obj = layer_info_list[selected_layer_idx][1]
            # Show class prediction if last layer (fc/Linear)
            last_layer = list(model.named_children())[-1][1]
            if layer_obj is last_layer and isinstance(layer_obj, torch.nn.Linear):
                # 1. Get predicted label and probability (=confidence)
                probabilities = torch.nn.functional.softmax(output_tensor[0], dim=0)
                predicted_idx = torch.argmax(probabilities).item()
                predicted_label = imagenet_labels[predicted_idx] if predicted_idx < len(imagenet_labels) else f"class {predicted_idx}"
                predicted_prob = probabilities[predicted_idx].item()
                confidence_percent = f"{predicted_prob * 100:.1f}%"
                
                # 2. Show "It is a ..." in the figure (plain, not HTML)
                plt.figure(figsize=(7, 7))
                plt.axis('off')
                plt.text(0.5, 0.5, f"It is a\n{predicted_label}!",
                         ha='center', va='center', fontsize=26, color='#67d800', weight='bold', wrap=True)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.tight_layout(pad=2)
                plt.show()
                
                # 3. Show class and confidence in HTML in the parameter/sidebar
                param_str = (
                    f"<b>Output:</b><br>"
                    f"<span style='color:#FA4;font-size:1.3em;'>It is a <b>{predicted_label}</b>!</span><br>"
                    f"<b>Confidence:</b> <span style='color:#8FD14A;font-size:1.15em'>{confidence_percent}</span><br><br>"
                    f"<b>{type(layer_obj).__name__}</b><br>"
                    f"<pre>{str(layer_obj)}</pre>"
                )
                layer_params_html.value = param_str
                return
    
            fmap = feature_maps.get(layer_obj)
            if fmap is not None:
                grid_image = make_grid(fmap, n_cols=8)
                if grid_image is not None:
                    norm_grid = (grid_image - np.min(grid_image)) / (np.ptp(grid_image) + 1e-9)
                    norm_grid = (norm_grid * 255).astype(np.uint8)
                    pil_img = PILImage.fromarray(norm_grid)
                    pil_img = pil_img.resize(canvas_size, resample=PILImage.NEAREST)
                    plt.figure(figsize=(7, 7))
                    plt.imshow(pil_img, cmap='viridis')
                    plt.title(f'Feature maps from: {layer_info_list[selected_layer_idx][0]}', fontsize=16, pad=16)
                    plt.axis('off')
                    plt.tight_layout(pad=2)
                    plt.show()
            else:
                plt.figure()
                plt.title('No feature map for selected layer', fontsize=14)
                plt.axis('off')
                plt.show()
            # Param string (limit lines for very big blocks)
            param_str = str(layer_obj)
            nlines = 23
            param_lines = param_str.split('\n')
            if len(param_lines) > nlines:
                shown_text = '\n'.join(param_lines[:nlines]) + '\n...\n' + '<span style="color:#fbb;">[truncated]</span>'
            else:
                shown_text = '\n'.join(param_lines)
            layer_params_html.value = f"<b>Layer parameters:</b><br><pre style='font-size:0.98em;'>{shown_text}</pre>"
    # --- UI LAYOUT ---
    
    dropdown_style = {
        'description_width': '80px',
    }
    
    image_dropdown = widgets.Dropdown(
        options=image_files,
        description='Image',
        value=image_files[0],
        style=dropdown_style,
        layout=widgets.Layout(width='180px', margin='0px 8px 0px 0px')
    )
    
    layer_dropdown = widgets.Dropdown(
        options=[x[0] for x in layer_info_list],
        description='Layer',
        value=layer_info_list[0][0],
        style=dropdown_style,
        layout=widgets.Layout(width='320px', margin='0px 0px 0px 0px')
    )
    
    header = widgets.HTML(
        value="<div style='font-size: 1.8em; color:#55AAF2; margin-bottom: 6px; font-family:sans-serif;'><b>Feature Map Visualizer</b></div>"
    )
    caption = widgets.HTML(
        value=(
            "<div style='font-size:1.07em; color:#BBB; margin-bottom: 6px;'>"
            "<b>Select an <span style=\"color:#EEE\">image</span></b> and <b><span style=\"color:#EEE\">network layer</span></b> to view its feature maps.<br>"
            "All feature maps are shown at a fixed display size for easy comparison.</div>"
        )
    )
    
    controls = widgets.HBox([image_dropdown, layer_dropdown], layout=widgets.Layout(margin="0 0 16px 0", align_items="center"))
    
    # Here, place image and param text side by side.
    panel = widgets.HBox(
        [output, layer_params_html],
        layout=widgets.Layout(
            justify_content='space-between',
            align_items='flex-start'
        )
    )
    
    group_box = widgets.VBox(
        [panel],
        layout=widgets.Layout(
            border='solid 2px #444',
            background_color='#111',
            padding='16px',
            align_items='flex-start'
        )
    )
    
    outer_box = widgets.VBox(
        [header, caption, controls, group_box],
        layout=widgets.Layout(
            background_color="#101014",
            padding="24px 24px 10px 24px",
            border_radius="12px",
            box_shadow="0 1px 8px rgba(0,0,0,0.11)",
            min_width="850px", max_width="1200px"
        )
    )
    
    def on_any_dropdown_change(change):
        idx = layer_dropdown.options.index(layer_dropdown.value)
        update_plot(image_dropdown.value, idx)
    
    image_dropdown.observe(on_any_dropdown_change, names='value')
    layer_dropdown.observe(on_any_dropdown_change, names='value')
    
    display(outer_box)
    update_plot(image_dropdown.value, layer_dropdown.options.index(layer_dropdown.value))


# Shows one positive and one negative pair with their embedding distance bar
def show_pair_verification_examples(model, val_loader, threshold, device, mean, std, embed_fn=None,
                                    positive_name='Genuine', negative_name='Forgery'):
    """
    Shows one positive and one negative pair from a triplet validation batch with their embedding distance.

    Args:
        model: Embedding model.
        val_loader: DataLoader that yields (anchor, positive, negative) batches.
        threshold: Distance below which a pair counts as positive.
        device: Device to run on.
        mean: Per-channel mean used for normalization, to display the images.
        std: Per-channel std used for normalization.
        embed_fn: Function (model, batch) -> embeddings. Defaults to model.get_embedding.
        positive_name: Name of a correct pair in titles.
        negative_name: Name of a wrong pair in titles.
    """
    embed_fn = embed_fn or (lambda m, x: m.get_embedding(x))
    model.eval()
    try:
        anchor, positive, negative = next(iter(val_loader))
    except StopIteration:
        print('DataLoader is empty.')
        return
    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
    index = random.choice(range(anchor.size(0)))
    pairs = [(anchor[index], positive[index], f'{positive_name} Pair', 'Anchor', 'Positive', 1),
             (anchor[index], negative[index], f'{negative_name} Pair', 'Anchor', 'Negative', 0)]
    for img1, img2, pair_type, title1, title2, true_label in pairs:
        with torch.no_grad():
            distance = F.pairwise_distance(embed_fn(model, img1.unsqueeze(0)), embed_fn(model, img2.unsqueeze(0))).item()
        prediction = 1 if distance < threshold else 0
        correct = prediction == true_label
        color = 'green' if correct else 'red'
        fig = plt.figure(figsize=(8, 5.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1])
        ax1, ax2, ax_dist = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        predicted_name = positive_name if prediction == 1 else negative_name
        operator = '<' if distance < threshold else '>='
        fig.suptitle(f"Type: {pair_type}\nPrediction: {predicted_name} -> {'CORRECT' if correct else 'INCORRECT'}\n"
                     f'Dist: {distance:.2f} {operator} Thresh: {threshold:.2f}', color=color, fontsize=12, y=0.98)
        for ax, img, title in ((ax1, img1, title1), (ax2, img2, title2)):
            ax.imshow(deprocess_image(img, mean, std))
            ax.set_title(title)
            ax.axis('off')
        ax_dist.barh([0], [distance], color=color, height=0.5, zorder=1)
        ax_dist.axvline(x=threshold, color='black', linestyle='--', zorder=2)
        ax_dist.text(distance + 0.02, 0, f'{distance:.2f}', va='center', fontweight='bold')
        ax_dist.set_yticks([])
        ax_dist.set_xlim(0, max(distance, threshold) * 1.5)
        ax_dist.set_xlabel('Distance in Embedding Space')
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.show()


# Shows one random validation pair per class with prediction and embedding distance
def show_pair_examples_per_class(model, val_loader, device, class_names, decide_fn, pair_forward_fn=None,
                                 mean=None, std=None, threshold=None, titles=('Before', 'After')):
    """
    Shows one random validation pair per class with the true class, the prediction and the embedding distance.

    Args:
        model: Pair model.
        val_loader: DataLoader that yields (input_a, input_b, label) batches.
        device: Device to run on.
        class_names: Dict mapping class index to display name.
        decide_fn: Function (tensor_a, tensor_b, distance) -> predicted class index.
        pair_forward_fn: Function (model, a, b) -> (embedding_a, embedding_b). Defaults to model(a, b).
        mean: Optional per-channel mean to undo normalization for display.
        std: Optional per-channel std to undo normalization for display.
        threshold: Optional threshold, only printed in the title.
        titles: Titles of the two displayed images.
    """
    pair_forward_fn = pair_forward_fn or (lambda m, a, b: m(a, b))
    model.eval()
    samples = {c: [] for c in class_names}
    for batch_a, batch_b, batch_labels in val_loader:
        for i in range(len(batch_labels)):
            label = batch_labels[i].item()
            if label in samples:
                samples[label].append((batch_a[i], batch_b[i]))
    if not all(samples.values()):
        print('Validation set is missing samples from one or more classes.')
        return
    for true_label, pairs in samples.items():
        tensor_a, tensor_b = random.choice(pairs)
        with torch.no_grad():
            emb_a, emb_b = pair_forward_fn(model, tensor_a.unsqueeze(0).to(device), tensor_b.unsqueeze(0).to(device))
            distance = F.pairwise_distance(emb_a, emb_b).item()
        prediction = decide_fn(tensor_a.to(device), tensor_b.to(device), distance)
        correct = prediction == true_label
        thr_text = f' (Threshold {threshold:.2f})' if threshold is not None else ''
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f'True Label: {class_names[true_label]} | Prediction: {class_names.get(prediction, "Unknown")} -> '
                     f"{'CORRECT' if correct else 'INCORRECT'}\nDistance: {distance:.2f}{thr_text}",
                     color='green' if correct else 'red', fontsize=12, y=1.03)
        for ax, tensor, title in zip(axes, (tensor_a, tensor_b), titles):
            image = deprocess_image(tensor, mean, std) if mean is not None and std is not None \
                else tensor.permute(1, 2, 0).numpy()
            ax.imshow(image)
            ax.set_title(title)
            ax.axis('off')
        plt.show()



# =============================================================================
# NLP UTILITIES
# =============================================================================


# Downloads and extracts the GloVe 6B word embedding dataset
def download_glove6B(extract_directory="./glove_data"):
    """
    Downloads and extracts the GloVe 6B word embedding dataset.

    Args:
        extract_directory: The path to the directory where the GloVe
                         data will be stored.
    """
    
    # Define the path to a sample file to check for existence
    sample_file_path = os.path.join(extract_directory, 'glove.6B.100d.txt')
    
    # Check if the data already exists to avoid re-downloading
    if os.path.exists(sample_file_path):
        print(f"GloVe data already found in '{extract_directory}'. Skipping download.")
        return

    print(f"GloVe data not found. Starting download...")
    # Create the extraction directory if it does not exist
    os.makedirs(extract_directory, exist_ok=True)
    
    # URL for the GloVe 6B dataset
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    # Path to save the temporary zip file
    zip_file_path = "glove.6B.zip"
    
    # Use a try-finally block to ensure the temporary zip file is removed
    try:
        # Stream the download to handle large files and show progress
        response = requests.get(url, stream=True)
        # Raise an exception for bad status codes (like 404)
        response.raise_for_status()
        # Get the total file size from the response headers
        total_size = int(response.headers.get('content-length', 0))
        
        # Initialize tqdm for a progress bar during download
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading glove.6B.zip") as pbar:
            # Open the local file in binary write mode
            with open(zip_file_path, "wb") as f:
                # Iterate over the response content in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    # Update the progress bar
                    pbar.update(len(chunk))
                    # Write the current chunk to the file
                    f.write(chunk)
        
        print(f"\nUnzipping '{zip_file_path}'...")
        # Open the downloaded zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all its contents to the specified directory
            zip_ref.extractall(extract_directory)
        print(f"Files extracted to '{extract_directory}'.")
    
    finally:
        # Clean up by removing the downloaded zip file
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            print(f"Removed temporary file '{zip_file_path}'.")


# Loads GloVe word embeddings from a specified file
def load_glove_embeddings(file_path):
    """
    Loads GloVe word embeddings from a specified file.

    Args:
        file_path: The path to the GloVe embeddings text file.

    Returns:
        A dictionary mapping each word to its NumPy vector representation.
    """
    
    # Initialize an empty dictionary to store the embeddings
    embeddings_index = {}
    
    # Open the embeddings file for reading with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        # Iterate over each line in the file
        for line in f:
            # Split the line into a list of values
            values = line.split()
            # The first value is the word
            word = values[0]
            # The remaining values are the vector components
            vector = np.asarray(values[1:], dtype='float32')
            # Add the word and its vector to the dictionary
            embeddings_index[word] = vector
            
    # Return the dictionary containing the loaded embeddings
    return embeddings_index


# Prints unique token to ID mappings from a list of tokenized sentences, sorted by token ID
def print_unique_token_id_mappings(tokens_list, input_ids_tensor):
    """
    Prints unique token to ID mappings from a list of tokenized sentences,
    sorted by token ID.

    Args:
        tokens_list (list of list of str): A list where each inner list
                                           contains token strings for a sentence.
        input_ids_tensor (torch.Tensor or similar): A tensor where each row
                                                    contains token IDs for a sentence.
    """
    unique_token_to_id_map = {}

    # Iterate through each sentence's tokens and IDs
    for i in range(len(tokens_list)):
        sentence_tokens = tokens_list[i]
        # Get the corresponding tensor of IDs for the current sentence
        sentence_ids = input_ids_tensor[i]
        for j in range(len(sentence_tokens)):
            token_str = sentence_tokens[j]
            # .item() converts a single-element tensor to a Python number
            token_id = sentence_ids[j].item()
            # Store in dictionary to ensure uniqueness of token strings
            unique_token_to_id_map[token_str] = token_id

    # Sort the unique mappings by token ID (the value in the dictionary)
    sorted_token_mappings = sorted(unique_token_to_id_map.items(), key=lambda item: item[1])

    print("\n--- Unique Token to ID Mappings (for these sentences) ---")
    if not sorted_token_mappings:
        print("No tokens to display.")
        return

    for token, token_id in sorted_token_mappings:
        print(f"{token}\t-->\t{token_id}")
        
    print()


# Constructs a vocabulary and mapping dictionaries from the provided text
def build_vocabulary(text, vocab_size=5000, tokenizer=None):
    """
    Constructs a vocabulary and mapping dictionaries from the provided text.

    Args:
        text (str): The raw input text used to build the vocabulary.
        vocab_size (int): The maximum number of tokens to include in the vocabulary.
        tokenizer (object): An optional tokenizer instance for processing the text.

    Returns:
        vocab (list): A list of strings representing the unique vocabulary tokens.
        word2idx (dict): A dictionary mapping each token string to its integer index.
        idx2word (dict): A dictionary mapping each integer index back to its token string.
        tokenizer (object): The tokenizer instance used during the build process.
    """
    # Instantiate a default tokenizer if a specific one is not provided
    if tokenizer is None:
        tokenizer = ShakespeareTokenizer()
    
    # Process the text into a list of individual tokens
    tokens = tokenizer(text)
    # Calculate the frequency of each token in the tokenized list
    token_counts = Counter(tokens)
    
    # Define the set of reserved tokens for padding, unknown entries, and newlines
    special_tokens = ['<pad>', '<unk>', '<nl>']
    
    # Extract the most frequent tokens to fit within the specified vocabulary capacity
    most_common = token_counts.most_common(vocab_size - len(special_tokens))

    # Iterate through special tokens to ensure they are not duplicated in the common list
    for st in special_tokens:
        # Check if the special token is present in the most common list
        if st in most_common:
            # Remove the special token from the most common list to prevent overlap
            del most_common[st]
        
    # Initialize the vocabulary list with the special tokens
    vocab = special_tokens.copy()
    # Append the most frequent tokens to the vocabulary list
    for token, count in most_common:
        vocab.append(token)
    
    # Generate a lookup table for converting word strings into numerical indices
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    # Generate a lookup table for converting numerical indices back into word strings
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Determine the total volume of tokens in the original processed text
    total_token_occurrences = sum(token_counts.values())
    # Calculate the frequency of tokens that successfully appear in the built vocabulary
    covered_token_occurrences = sum(token_counts[token] for token in vocab if token in token_counts)
    # Determine the percentage of the text represented by the current vocabulary
    coverage = covered_token_occurrences / total_token_occurrences
    
    # Count the occurrences of tokens that fall outside the final vocabulary
    unknown_count = sum(count for token, count in token_counts.items() if token not in word2idx)
    # Determine the ratio of unknown tokens to the total number of tokens
    unknown_rate = unknown_count / total_token_occurrences
    
    # Log the summary statistics for the generated vocabulary to the console
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Unique tokens in text: {len(token_counts)}")
    print(f"Coverage: {coverage:.1%} of token occurrences")
    print(f"Unknown token rate: {unknown_rate:.1%}")
    print(f"Most common tokens: {vocab[3:13]}")
    print(f"Least common in vocab: {vocab[-10:]}")
    
    # Provide the vocabulary and the mapping tools for model use
    return vocab, word2idx, idx2word, tokenizer


# Generates overlapping input and target sequences for training language models
def create_sequences(text, word2idx, idx2word, tokenizer=None, seq_len=150):
    """
    Generates overlapping input and target sequences for training language models.

    Args:
        text (str): The raw input string to be tokenized and processed.
        word2idx (dict): A dictionary mapping unique tokens to their integer indices.
        idx2word (dict): A dictionary mapping integer indices back to their token strings.
        tokenizer (object): An optional tokenizer instance; defaults to ShakespeareTokenizer if None.
        seq_len (int): The fixed length for each generated input sequence.

    Returns:
        inputs (list): A list of integer lists, where each inner list is an input sequence.
        targets (list): A list of integer lists, where each inner list is the corresponding target sequence.
    """
    # Instantiate a default tokenizer if a specific instance is not provided
    if tokenizer is None:
        tokenizer = ShakespeareTokenizer()
    
    # Transform the raw text into a complete list of tokens
    tokens = tokenizer(text)
    
    # Initialize containers for the processed numerical sequences
    inputs = []
    targets = []
    
    # Iterate through the tokens to create sliding window samples
    for i in range(len(tokens) - seq_len):
        # Define the current input window based on the sequence length
        window = tokens[i:i+seq_len]
        # Define the target window as the input window shifted by one token
        target = tokens[i+1:i+seq_len+1]
        
        # Map input tokens to their indices, using the unknown token as a fallback
        input_ids = [word2idx.get(w, word2idx['<unk>']) for w in window]
        # Map target tokens to their indices, using the unknown token as a fallback
        target_ids = [word2idx.get(w, word2idx['<unk>']) for w in target]
        
        # Append the encoded numerical sequences to the output lists
        inputs.append(input_ids)
        targets.append(target_ids)
    
    # Log the total count of sequences generated during the process
    print(f"Created {len(inputs)} sequences of length {seq_len}")
    
    # Validate the generated data by displaying a sample of the results
    if inputs:
        # Convert the first few indices of the first sample back to strings
        input_tokens = [idx2word[id] for id in inputs[0][:10]]
        target_tokens = [idx2word[id] for id in targets[0][:10]]
        
        # Display the human-readable tokens for verification
        print(f"Example input tokens: {input_tokens}...")
        print(f"Example target tokens: {target_tokens}...")
        
        # Check that the temporal shift between inputs and targets is correctly aligned
        if len(inputs[0]) > 5:
            print(f"\nVerifying shift:")
            for i in range(5):
                # Retrieve the strings for the current position in both sequences
                input_token = idx2word[inputs[0][i]]
                target_token = idx2word[targets[0][i]]
                # Identify the token that should theoretically be the next target
                expected = idx2word[inputs[0][i+1]] if i+1 < len(inputs[0]) else "N/A"
                print(f"  Position {i}: input='{input_token}' → target='{target_token}' (expected: '{expected}')")
    
    # Return the completed lists of input and target indices
    return inputs, targets


# Torch Dataset that serves fixed-length input and target token sequences
class ShakespeareDataset(Dataset):
    """
    A custom Dataset class for handling Shakespearean text sequences in PyTorch.
    """
    def __init__(self, inputs, targets):
        """
        Initializes the dataset with input and target sequences.

        Args:
            inputs (list): A list of integer sequences representing the input tokens.
            targets (list): A list of integer sequences representing the target tokens.
        """
        # Convert the input sequences into a PyTorch long tensor for processing
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        # Convert the target sequences into a PyTorch long tensor for processing
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        """
        Returns the total number of samples available in the dataset.

        Returns:
            length (int): The total count of sequences in the dataset.
        """
        # Calculate the size based on the number of rows in the input tensor
        return len(self.inputs)
    
    def __getitem__(self, idx):
        """
        Retrieves a specific sample and its corresponding target by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            input_sample (Tensor): The input sequence tensor at the specified index.
            target_sample (Tensor): The target sequence tensor at the specified index.
        """
        # Return the input and target tensors for the given index
        return self.inputs[idx], self.targets[idx]


# Callable word tokenizer for plain English text with punctuation
class ShakespeareTokenizer:
    """
    A specialized tokenizer for processing theatrical or poetic text.
    """
    def __call__(self, text):
        """
        Executes the tokenization process on the input text.

        Args:
            text (str): The raw text string to be processed.

        Returns:
            tokens (list): A list of strings containing words, contractions, 
                           line break markers, and punctuation marks.
        """
        # Substitute newline characters with a recognizable placeholder token
        text = text.replace('\n', ' <nl> ')
        # Apply regular expressions to extract alphanumeric words, possessives, and symbols
        return re.findall(r"\w+(?:'\w+)?|<nl>|[^\w\s]", text)


# Callable tokenizer that splits text into words and punctuation per language
class MultilingualTokenizer:
    """
    Tokenizer that can handle multiple languages
    """
    def __init__(self, language='French'):
        self.language = language
    
    def __call__(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Language-specific handling
        if self.language == 'Russian':
            # Keep Cyrillic characters
            text = re.sub(r"([.!?])", r" \1", text)
            tokens = re.findall(r"[\w]+|[.!?]", text)
        else:
            # Improved handling for Latin-based languages
            # First, add spaces around punctuation (but not apostrophes within words)
            text = re.sub(r"([.!?])", r" \1", text)
            # Find words (including contractions) and punctuation
            # This pattern captures:
            # - Words with internal apostrophes (contractions)
            # - Regular words
            # - Punctuation marks
            tokens = re.findall(r"\b\w+(?:'\w+)*\b|[.!?]", text)
        
        return tokens


# Normalize a string based on the target language
def normalize_string(s, language='French'):
    """
    Normalize a string based on the target language
    """
    # Convert to lowercase
    s = s.lower().strip()
    
    # Language-specific normalization
    if language == 'Russian':
        # Keep Cyrillic characters
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^а-яА-Яa-zA-Z.!?]+", r" ", s)
    elif language in ['French', 'Spanish', 'Portuguese', 'Italian']:
        # Keep Latin characters with accents and apostrophes for contractions
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÀ-ÿ'.!?]+", r" ", s)  # Added apostrophe
    elif language == 'German':
        # Keep German special characters and apostrophes
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÄäÖöÜüß'.!?]+", r" ", s)  # Added apostrophe
    else:
        # Default handling (including English) - keep apostrophes for contractions
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z'.!?]+", r" ", s)  # Added apostrophe
    
    # Remove extra spaces
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Prepare translation data by normalizing text and creating a tokenizer
def prepare_data(translation_pairs: List[Tuple[str, str]], 
                 target_language: str, 
                 max_pairs: int = 15000, 
                 max_length: int = 20) -> Tuple[List[Tuple[str, str]], object]:
    """
    Prepare translation data by normalizing text and creating a tokenizer.
    
    Args:
        translation_pairs: List of (English, target_language) translation pairs
        target_language: Name of the target language
        max_pairs: Maximum number of pairs to process (default: 15000)
        max_length: Maximum sentence length in words (default: 20)
        
    Returns:
        Tuple of (normalized_pairs, tokenizer)
    """
    # Apply normalization to all pairs
    normalized_pairs = []
    
    for eng, target in translation_pairs[:max_pairs]:
        eng_norm = normalize_string(eng, 'English')
        target_norm = normalize_string(target, target_language)
        
        # Filter by length for more manageable training
        if len(eng_norm.split()) <= max_length and len(target_norm.split()) <= max_length:
            normalized_pairs.append((eng_norm, target_norm))
    
    # Create tokenizer for the selected language
    tokenizer = MultilingualTokenizer(target_language)
    
    # Test the tokenizer with examples containing contractions
    print(f"\n=== Tokenizer Test for {target_language} ===")
    
    # Test with English contractions
    test_sentences = [
        "Hello, how are you?",
        "He's going to the store.",
        "I can't believe it's working!",
        "They're here, aren't they?"
    ]
    
    for test_sent in test_sentences:
        tokens = tokenizer(test_sent)
        print(f"Original: {test_sent}")
        print(f"Tokenized: {tokens}")
        print()
    
    # Test with a sample from the target language if available
    if normalized_pairs:
        sample_target = normalized_pairs[0][1]  # Get first target language sentence
        print(f"\n{target_language} sample: {sample_target}")
        print(f"{target_language} tokenized: {tokenizer(sample_target)}")
    
    print(f"\n=== Data Preparation Complete ===")
    print(f"Normalized pairs: {len(normalized_pairs)} (from {min(max_pairs, len(translation_pairs))} original pairs)")
    
    return normalized_pairs, tokenizer


# Analyzes and displays descriptive statistics for the provided sentiment dataset
def print_data_statistics(train_reviews, train_labels, test_reviews, test_labels, sample_size=100):
    """
    Analyzes and displays descriptive statistics for the provided sentiment dataset.

    Args:
        train_reviews: List of strings containing training review text.
        train_labels: List of integers representing training sentiment labels.
        test_reviews: List of strings containing test review text.
        test_labels: List of integers representing test sentiment labels.
        sample_size: Number of instances to consider for length calculations.

    Returns:
        None. This function outputs statistics directly to the console.
    """
    # Print the header and volume counts for the training portion of the dataset
    print("\n=== Dataset Statistics ===")
    print(f"Training set:")
    print(f"  Total reviews: {len(train_reviews)}")
    # Sum binary labels to determine the total count of positive samples
    print(f"  Positive reviews: {sum(train_labels)}")
    # Subtract positive count from total length to find negative sample count
    print(f"  Negative reviews: {len(train_labels) - sum(train_labels)}")
    
    # Print volume counts for the testing portion of the dataset
    print(f"\nTest set:")
    print(f"  Total reviews: {len(test_reviews)}")
    print(f"  Positive reviews: {sum(test_labels)}")
    print(f"  Negative reviews: {len(test_labels) - sum(test_labels)}")
    
    # Begin displaying representative text samples from the dataset
    print("\n=== Sample Reviews ===")
    
    # Locate and display the first available positive review from the training set
    try:
        # Find the index of the first occurrence of the positive label
        positive_idx = train_labels.index(1)
        print("\n--- Positive Review Example ---")
        print(f"Label: Positive")
        # Display a truncated version of the review text
        print(f"Text (first 400 chars): {train_reviews[positive_idx][:400]}...")
    except ValueError:
        # Handle cases where no positive labels are present in the list
        print("\nNo positive reviews found in training set")
    
    # Locate and display the first available negative review from the training set
    try:
        # Find the index of the first occurrence of the negative label
        negative_idx = train_labels.index(0)
        print("\n--- Negative Review Example ---")
        print(f"Label: Negative")
        # Display a truncated version of the review text
        print(f"Text (first 400 chars): {train_reviews[negative_idx][:400]}...")
    except ValueError:
        # Handle cases where no negative labels are present in the list
        print("\nNo negative reviews found in training set")
    
    # Ensure the requested sample size does not exceed the total available reviews
    sample_size = min(sample_size, len(train_reviews))
    # Calculate word counts for each review by splitting on whitespace
    review_lengths = [len(review.split()) for review in train_reviews[:sample_size]]
    
    # Output descriptive statistics for the review lengths
    print(f"\n=== Review Length Statistics (first {sample_size} reviews) ===")
    # Calculate the arithmetic mean of the review lengths
    print(f"Average length: {np.mean(review_lengths):.0f} words")
    # Identify the shortest review in the sample
    print(f"Min length: {min(review_lengths)} words")
    # Identify the longest review in the sample
    print(f"Max length: {max(review_lengths)} words")
    # Calculate the median length for the sample
    print(f"Median length: {np.median(review_lengths):.0f} words")
    # Calculate the standard deviation of lengths to understand variability
    print(f"Std deviation: {np.std(review_lengths):.0f} words")


# Displays the final validation metrics in a formatted manner
def print_final_metrics(results_dic):
    """
    Displays the final validation metrics in a formatted manner.

    Args:
        results_dic: A dictionary containing the final validation metrics,
                     expected to have keys 'val_accuracy', 'val_precision',
                     'val_recall', and 'val_f1'.
    """
    # Print a header for the metrics section.
    print("\nFinal Validation Metrics")
    # Print the formatted validation accuracy.
    print(f"\nAccuracy:   {results_dic['val_accuracy']:.4f}")
    # Print the formatted validation precision.
    print(f"Precision:  {results_dic['val_precision']:.4f}")
    # Print the formatted validation recall.
    print(f"Recall:     {results_dic['val_recall']:.4f}")
    # Print the formatted validation F1-score.
    print(f"F1:         {results_dic['val_f1']:.4f}\n")


# Formats the per-class accuracy data into a markdown table string
def _create_accuracy_table(class_accuracy, id2cat):
    """
    Formats the per-class accuracy data into a markdown table string.
    
    This is a helper function designed to be called by the main analysis function.

    Args:
        class_accuracy (np.array): An array of accuracy values for each class.
        id2cat (dict): A dictionary mapping class IDs to category names.

    Returns:
        str: A formatted markdown string representing the accuracy table.
    """
    # Build the markdown string for the table headers and rows.
    markdown_table = "| Category                  | Accuracy |\n"
    markdown_table += "|---------------------------|----------|\n"
    
    # Loop through each class to create a row in the table.
    for class_id, accuracy in enumerate(class_accuracy):
        category_name = id2cat[class_id]
        # Format each row and add it to the string.
        markdown_table += f"| {category_name:<25} | {accuracy:.2%}    |\n"
        
    return markdown_table


# Builds vocabulary, sequences and train/validation loaders for next-token models
def prepare_text_sequence_data(text_file_or_string, vocab_size=5000, seq_len=150, batch_size=32, train_split=0.9):
    """
    Runs the full text pipeline for next-token models: vocabulary, sequences and train/validation loaders.

    Args:
        text_file_or_string: Path to a .txt file or the text itself.
        vocab_size: Maximum vocabulary size.
        seq_len: Length of each input sequence.
        batch_size: Batch size of the loaders.
        train_split: Fraction of sequences used for training.

    Returns:
        Dict with the vocab, word2idx, idx2word, tokenizer, loaders, dataset and sizes.
    """
    if isinstance(text_file_or_string, str) and text_file_or_string.endswith('.txt'):
        with open(text_file_or_string, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = text_file_or_string
    print('Step 1: Building vocabulary...')
    vocab, word2idx, idx2word, tokenizer = build_vocabulary(text, vocab_size)
    print(f'\nStep 2: Creating sequences (length={seq_len})...')
    inputs, targets = create_sequences(text, word2idx, idx2word, tokenizer, seq_len)
    print(f'\nStep 3: Creating dataloaders (batch_size={batch_size})...')
    train_loader, val_loader, dataset = create_sequence_dataloaders(inputs, targets, batch_size, train_split)
    return {'vocab': vocab, 'word2idx': word2idx, 'idx2word': idx2word, 'vocab_size': len(vocab),
            'tokenizer': tokenizer, 'train_loader': train_loader, 'val_loader': val_loader,
            'dataset': dataset, 'seq_len': seq_len}



# =============================================================================
# DIFFUSION AND GENERATIVE IMAGE HELPERS
# =============================================================================


# Estimates the clean image from a noisy one in a single diffusion step
def remove_all_noise_at_once(noisy_image, predicted_noise, timestep, scheduler):
    """
    Apply the diffusion denoising formula to estimate the original clean image (x_0)
    directly from the current noisy state in a single step.

    Arguments:
        noisy_image: The current latent tensor with noise.
        predicted_noise: The noise component predicted by the model.
        timestep: The current diffusion timestep.
        scheduler: The noise scheduler containing alpha schedules.

    Returns:
        The estimated clean image tensor.
    """
    # Retrieve the cumulative alpha value for the specific timestep
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(noisy_image.device)
    
    # Compute the square root of the cumulative alpha
    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)

    # Compute the square root of one minus the cumulative alpha
    sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
    
    # Apply the mathematical formula to subtract the predicted noise and scale the result
    clean_image = (noisy_image - sqrt_one_minus_alpha_prod_t * predicted_noise) / sqrt_alpha_prod_t
    
    # Return the resulting estimated clean image
    return clean_image


# Perform a single iteration of the denoising process to transition from timestep t to t-1
def gradual_denoise_step(noisy_image, predicted_noise, timestep, scheduler):
    """
    Perform a single iteration of the denoising process to transition from
    timestep t to t-1.

    Arguments:
        noisy_image: The current image tensor.
        predicted_noise: The noise predicted by the model.
        timestep: The current step in the diffusion process.
        scheduler: The scheduler object handling the step logic.

    Returns:
        The slightly denoised image sample for the next step.
    """
    # Execute the scheduler step function to compute the previous sample
    step = scheduler.step(predicted_noise, timestep, noisy_image)
    
    # Extract and return the denoised sample from the scheduler output
    return step.prev_sample


# Shows gradual and one-step denoising results at several steps of a DDPM
@torch.no_grad()
def visualize_ddpm_denoising(pipe, num_inference_steps=100):
    """
    Execute the diffusion process to visualize both gradual denoising and 
    one-step noise removal approximations at various intervals.

    Arguments:
        pipe: The stable diffusion pipeline object.
        num_inference_steps: The total number of denoising steps to perform.

    Returns:
        A tuple containing two lists: gradual denoising images and full removal approximation images.
    """
    # Extract image configuration parameters from the model configuration
    image_size = pipe.unet.config.sample_size
    num_channels = pipe.unet.config.in_channels
    device = pipe.device

    # Initialize a random noise tensor to serve as the starting point
    images = torch.randn(1, num_channels, image_size, image_size, device=device)
    
    # distinct references to the scheduler and UNet model
    scheduler = pipe.scheduler
    model = pipe.unet
    
    # Configure the scheduler with the specified number of inference steps
    scheduler.set_timesteps(num_inference_steps)

    # Initialize a list to track specific timesteps for visualization
    timesteps_to_show = []
    n_vis = 7

    # Calculate indices for the steps to be visualized
    for i in range(n_vis):
        # Determine the index for the current visualization interval
        t_idx = int((num_inference_steps - 1) * i / (n_vis - 1))
        timesteps_to_show.append(t_idx)

    # Initialize storage lists for the output images
    gradual_images = []     # stores (step_index, gradual denoised image)
    full_removal_images = [] # stores (step_index, "full denoise" image)
    
    # Set up the timesteps and initial latent variables
    timesteps = scheduler.timesteps
    latents = images

    # Iterate through each timestep in the denoising process
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Generate a noise prediction using the model
        noise_pred = model(latents, t).sample

        # Determine if the current step is flagged for visualization
        if i in timesteps_to_show:
            # Estimate the fully clean image from the current state
            full_removal = remove_all_noise_at_once(latents, noise_pred, t, scheduler)
            full_removal_img = tensor_to_image(full_removal)
            full_removal_images.append((i, full_removal_img))
            
            # Convert the current noisy latent to an image for comparison
            current_img = tensor_to_image(latents)
            gradual_images.append((i, current_img))

        # Update the latents by performing a single denoising step
        latents = gradual_denoise_step(latents, noise_pred, t, scheduler)

    # Convert the final resulting tensor to an image
    final_img = tensor_to_image(latents)
    gradual_images.append((num_inference_steps, final_img))

    # Return the collected image sequences
    return gradual_images, full_removal_images


# Initialize and display an interactive UI widget for image generation
def load_widget(pipe):
    """
    Initialize and display an interactive UI widget for image generation.

    Arguments:
        pipe: The loaded Stable Diffusion pipeline.
    """
    
    # Initialize output container and image display widgets
    output = widgets.Output()
    gif_widget = widgets.Image(format='gif', width=320, height=320)
    final_image_widget = widgets.Image(format='png', width=320, height=320)
    
    # Define generic styling dictionaries for widgets
    textbox_style = {'description_width': '120px'}
    slider_style = {'description_width': '140px'}
    
    # Create the HTML header element
    heading = widgets.HTML("<h3 style='color:#0EA5E9;font-family:sans-serif'>Stable Diffusion Image Generator</h3>")
    
    # Initialize the checkbox for toggling animation mode
    mode_checkbox = widgets.Checkbox(
        value=True,
        description='Show denoising animation (GIF)',
        indent=False,
        layout=widgets.Layout(width='300px')
    )
    
    # Create an informational tooltip/label regarding the generation mode
    mode_info = widgets.HTML(
        value="""<div style='background-color:#000000;padding:8px;border-radius:6px;margin:8px 0;'>
        <small><b>💡 Tip:</b> Uncheck for faster generation (final image only)</small>
        </div>"""
    )
    
    # Create the text input field for the primary prompt
    prompt_widget = widgets.Text(
        value="A puppy dog riding a skateboard in times square", 
        description='Prompt:', 
        style=textbox_style,
        layout=widgets.Layout(width='500px')
    )
    
    # Create the text input field for the negative prompt
    negative_prompt_widget = widgets.Text(
        value="", 
        description='Negative:', 
        style=textbox_style,
        layout=widgets.Layout(width='500px')
    )
    
    # Create a slider for adjusting the number of inference steps
    steps_slider = widgets.IntSlider(
        value=50, min=10, max=100, step=1,
        description='Inference steps:', 
        style=slider_style,
        continuous_update=False,
        readout=False,
        layout=widgets.Layout(width='350px'),
    )
    
    # Create a label to display the current steps value
    steps_value = widgets.Label(f"{steps_slider.value}")
    
    # Define a callback to update the steps label when the slider moves
    def update_steps_label(*a):
        steps_value.value = f"{steps_slider.value}"
    steps_slider.observe(update_steps_label, "value")
    
    # Create a slider for adjusting the guidance scale
    gs_slider = widgets.FloatSlider(
        value=7.5, min=4.0, max=16.0, step=0.1,
        description='Guidance scale:', 
        style=slider_style,
        continuous_update=False,
        readout=False,
        layout=widgets.Layout(width='350px'),
    )
    
    # Create a label to display the current guidance scale value
    gs_value = widgets.Label(f"{gs_slider.value:.2f}")
    
    # Define a callback to update the guidance scale label when the slider moves
    def update_gs_label(*a):
        gs_value.value = f"{gs_slider.value:.2f}"
    gs_slider.observe(update_gs_label, "value")
    
    # Initialize the execution button
    run_button = widgets.Button(
        description="✨ Generate with Animation ✨", 
        button_style='info',
        layout=widgets.Layout(width='250px', height='40px')
    )
    
    # Define logic to update button text based on the selected mode
    def update_button_text(*args):
        if mode_checkbox.value:
            run_button.description = "✨ Generate with Animation ✨"
            run_button.button_style = 'info'
        else:
            run_button.description = "⚡ Quick Generate ⚡"
            run_button.button_style = 'success'
    mode_checkbox.observe(update_button_text, 'value')
    
    # Arrange the input widgets vertically
    input_form = widgets.VBox([
        heading,
        widgets.HBox([mode_checkbox, mode_info]),
        prompt_widget,
        negative_prompt_widget,
        widgets.HBox([steps_slider, steps_value]),
        widgets.HBox([gs_slider, gs_value]),
        run_button,
    ], layout=widgets.Layout(
        border='2px solid #0EA5E9', 
        box_shadow="2px 2px 10px #0EA5E9",
        padding='18px 24px 18px 24px', 
        border_radius='14px',
        width='600px',
        background='white'
    ))
    
    # Group the image widgets into a container
    image_container = widgets.VBox([gif_widget, final_image_widget])
    
    # Combine the form, output log, and images into the main UI object
    ui = widgets.VBox([
        input_form,
        output,
        image_container
    ])
    
    def generate_with_animation(prompt, negative_prompt, num_inference_steps, guidance_scale):
        """
        Execute generation process including intermediate step capture for animation.

        Arguments:
            prompt: Text prompt for generation.
            negative_prompt: Negative text prompt.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale value.
        """
        with output:
            # Clear previous output logs
            output.clear_output()
            print(f"\n🎬 Generating animation with {num_inference_steps} steps")
            print(f"Prompt: {prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            print()
            
            # Initialize storage for captured frames
            intermediate_images = []
            
            # Initialize and display the progress bar
            progress = widgets.IntProgress(
                value=0,
                min=0,
                max=num_inference_steps,
                description='Denoising:',
                bar_style='info',
                orientation='horizontal'
            )
            display(progress)
            
            # Define the callback function to run at each step
            def collect_callback(step, timestep, latents):
                with torch.no_grad():
                    # Scale latents to match the VAE expectation
                    scaled_latents = latents / 0.18215
                    # Decode latents to image space
                    image = pipe.vae.decode(scaled_latents).sample
                    # Normalize and clamp image values
                    image = (image / 2 + 0.5).clamp(0, 1)
                    # Rearrange dimensions for PIL compatibility
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    # Convert to PIL Image object
                    pil_img = Image.fromarray((image * 255).astype('uint8'))
                # Store the processed image frame
                intermediate_images.append((step, pil_img))
                # Update progress bar value
                progress.value = step + 1
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Create a deterministic generator based on a fixed seed
            generator = torch.Generator(pipe.device).manual_seed(42)
            intermediate_images.clear()
            
            # Execute the pipeline with the callback function
            _ = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=collect_callback,
                callback_steps=1,
                progress_bar=False,
            )
            
            # Update visual indicators for completion
            progress.bar_style = 'success'
            progress.description = 'Done'
            
            print("\n🎨 Creating animation...")
            # Initialize buffer for GIF data
            gif_bytes = io.BytesIO()
            # Extract image objects from the storage list
            all_imgs = [img for (step, img) in intermediate_images]
            # Save the sequence of images as a GIF
            imageio.mimsave(gif_bytes, all_imgs, format='GIF', duration=0.15)
            
            # Update widget visibility to show the animation
            final_image_widget.layout.display = 'none'
            gif_widget.layout.display = 'block'
            gif_widget.value = gif_bytes.getvalue()
            
            # Calculate and print elapsed time
            elapsed = time.time() - start_time
            print(f"✅ Animation generated in {elapsed:.1f} seconds")
    
    def generate_final_only(prompt, negative_prompt, num_inference_steps, guidance_scale):
        """
        Execute generation process returning only the final result for efficiency.

        Arguments:
            prompt: Text prompt for generation.
            negative_prompt: Negative text prompt.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale value.
        """
        with output:
            # Clear previous output logs
            output.clear_output()
            print(f"\n⚡ Quick generation with {num_inference_steps} steps")
            print(f"Prompt: {prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            print()
            
            # Initialize and display the progress bar
            progress = widgets.IntProgress(
                value=0,
                min=0,
                max=1,
                description='Generating:',
                bar_style='success',
                orientation='horizontal'
            )
            display(progress)
            
            # Record start time
            start_time = time.time()
            # Create a deterministic generator
            generator = torch.Generator(pipe.device).manual_seed(42)
            
            # Run the pipeline without callbacks
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                progress_bar=False,
            )
            
            # Update progress bar to completion
            progress.value = 1
            progress.description = 'Done'
            
            # Retrieve the resulting image
            final_img = result.images[0]
            # Save image to a byte buffer
            img_bytes = io.BytesIO()
            final_img.save(img_bytes, format='PNG')
            
            # Update widget visibility to show the static image
            gif_widget.layout.display = 'none'
            final_image_widget.layout.display = 'block'
            final_image_widget.value = img_bytes.getvalue()
            
            # Calculate and print elapsed time
            elapsed = time.time() - start_time
            print(f"✅ Image generated in {elapsed:.1f} seconds")
            
            # Display summary statistics
            print(f"\n📊 Generation Stats:")
            print(f"  • Mode: Quick (final image only)")
            print(f"  • Steps: {num_inference_steps}")
            print(f"  • Guidance Scale: {guidance_scale}")
            print(f"  • Time saved: ~{num_inference_steps * 0.15:.1f}s (animation creation)")
    
    # Define the click handler for the run button
    def run_on_click(b):
        if mode_checkbox.value:
            generate_with_animation(
                prompt_widget.value,
                negative_prompt_widget.value or None,
                steps_slider.value,
                gs_slider.value
            )
        else:
            generate_final_only(
                prompt_widget.value,
                negative_prompt_widget.value or None,
                steps_slider.value,
                gs_slider.value
            )
    
    # Attach the click handler to the button
    run_button.on_click(run_on_click)
    # Render the UI
    display(ui)



# =============================================================================
# EXPERIMENT TRACKING AND REPORTING
# =============================================================================


# Shows a per-class accuracy table and a confusion matrix heatmap from a results dict
def analyze_and_plot_results(results, id2cat, title='Confusion Matrix', confusion_matrix_key='confusion_matrix',
                             xlabel='Predicted Category', ylabel='True Category'):
    """
    Shows a per-class accuracy table and a confusion matrix heatmap from a results dict.

    Args:
        results: Dict that holds a confusion matrix tensor under confusion_matrix_key.
        id2cat: Dict mapping class id to class name.
        title, xlabel, ylabel: Heatmap texts.
        confusion_matrix_key: Key of the confusion matrix inside results.
    """
    confusion_matrix_numpy = results[confusion_matrix_key].cpu().numpy()
    class_names = [name for _, name in sorted(id2cat.items())]
    class_accuracy = confusion_matrix_numpy.diagonal() / (confusion_matrix_numpy.sum(axis=1) + 1e-09)
    display(Markdown('### **Per-Class Accuracy**'))
    display(Markdown(_create_accuracy_table(class_accuracy, id2cat)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_numpy, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Builds a styled table that compares metrics of several models
def get_results_df(results_by_name, metrics=('val_accuracy', 'val_precision', 'val_recall', 'val_f1')):
    """
    Builds a styled table that compares the metrics of several models.

    Args:
        results_by_name: Dict mapping a model name to its results dict.
        metrics: Keys to show as columns. Each key becomes a title-cased column such as 'Val Accuracy'.

    Returns:
        A pandas Styler with four decimals per value.
    """
    columns = {key.replace('_', ' ').title(): [res[key] for res in results_by_name.values()] for key in metrics}
    results_df = pd.DataFrame({'Model': list(results_by_name.keys()), **columns}).set_index('Model')
    return results_df.style.format({col: '{:.4f}' for col in columns})


# Starts the MLflow UI server in the background and prints its address
def start_mlflow_ui(base_path='/mlflow', port=5000, access_url=None):
    """
    Starts the MLflow UI server in the background and prints how to reach it.

    Args:
        base_path: URL prefix the UI is served under, for use behind a reverse proxy.
        port: Port the server listens on.
        access_url: Full URL to print for the user. Defaults to http://localhost:<port><base_path>.

    Returns:
        The subprocess handle. Call .terminate() on it to stop the server.
    """
    print('Starting MLflow UI...')
    process = subprocess.Popen(['mlflow', 'ui', '--host', '0.0.0.0', '--port', str(port), '--static-prefix', base_path],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('Waiting for server to start...')
    time.sleep(3)
    print('\n' + '=' * 60)
    print('MLflow UI is running in the background!')
    print(f"Access it at: {access_url or f'http://localhost:{port}{base_path}'}")
    print('To stop the server later, call .terminate() on the returned process.')
    print('=' * 60)
    return process


# Prints the experiment, params, final metrics and artifacts of an MLflow run
def display_mlflow_run_details(run_id, param_keys=None, metric_keys=None):
    """
    Prints the experiment, parameters, final metrics and artifacts of an MLflow run.

    Args:
        run_id: ID of the run to inspect.
        param_keys: Parameter names to show. All parameters when None.
        metric_keys: Metric names to show. All metrics when None.
    """
    if not run_id or not run_id.strip():
        print('Error: A valid Run ID must be provided. It cannot be empty or just whitespace.')
        return
    client = mlflow.tracking.MlflowClient()
    print(f'MLflow Client Initialized. Fetching details for Run ID: {run_id}\n')
    try:
        run_info = client.get_run(run_id)
        print(f'Details for Run ID: {run_info.info.run_id}')
        experiment_id = run_info.info.experiment_id
        try:
            print(f'Belongs to Experiment: {client.get_experiment(experiment_id).name} (ID: {experiment_id})')
        except Exception as e:
            print(f'Belongs to Experiment ID: {experiment_id} (Could not fetch details: {e})')
        params = run_info.data.params
        metrics = run_info.data.metrics
        print('\n  Parameters:')
        shown = [k for k in (param_keys if param_keys is not None else params) if k in params]
        for key in shown:
            print(f'    - {key}: {params[key]}')
        if not shown:
            print('    No matching parameters found for this run.')
        print('\n  Metrics (final values):')
        shown = [k for k in (metric_keys if metric_keys is not None else metrics) if k in metrics]
        for key in shown:
            try:
                print(f'    - {key}: {float(metrics[key]):.4f}')
            except ValueError:
                print(f'    - {key}: {metrics[key]}')
        if not shown:
            print('    No matching metrics found for this run.')
        print('\n  Artifacts:')
        artifacts = client.list_artifacts(run_info.info.run_id)
        if artifacts:
            for artifact in artifacts:
                print(f"    - '{artifact.path}' ({'Directory' if artifact.is_dir else 'File'}).")
        else:
            print('    No artifacts found for this run.')
        print('  ---')
    except mlflow.exceptions.MlflowException as e:
        print(f"Error fetching run details for Run ID '{run_id}': {e}")
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


# Prints model inputs and the predicted result as a small text table
def print_prediction_table(rows, result, title=' Model Prediction ', label_width=40, value_width=21):
    """
    Prints inputs and a model result as a small text table.

    Args:
        rows: List of (label, value) pairs describing the inputs.
        result: (label, value) pair printed as the last row.
        title: Centered table title.
        label_width: Width of the label column.
        value_width: Width of the value column.
    """
    border = '+' + '-' * (label_width + 2) + '+' + '-' * (value_width + 2) + '+'
    print(border)
    print('|' + title.center(label_width + value_width + 5) + '|')
    print(border)
    for label, value in rows:
        print(f'| {label:<{label_width}} | {str(value):<{value_width}} |')
    print(border)
    print(f'| {result[0]:<{label_width}} | {str(result[1]):<{value_width}} |')
    print(border)


# Prints parameter counts, file size and metrics of two saved models
def compare_saved_models(state_dict_path_a, metrics_path_a, state_dict_path_b, metrics_path_b, device='cpu',
                         names=('Baseline Model', 'Compared Model'),
                         metric_labels=(('accuracy', 'Final Accuracy', '{:.2f}%'),
                                        ('precision', 'Final Precision (Macro)', '{:.4f}'),
                                        ('recall', 'Final Recall (Macro)', '{:.4f}'),
                                        ('f1_score', 'Final F1-Score (Macro)', '{:.4f}'))):
    """
    Prints parameter counts, file size and metrics of two saved models side by side in two blocks.

    Args:
        state_dict_path_a: Saved state dict of the first model.
        metrics_path_a: Pickle file with the first model's metrics dict.
        state_dict_path_b: Saved state dict of the second model.
        metrics_path_b: Pickle file with the second model's metrics dict.
        device: Device used to load the state dicts.
        names: Headings of the two blocks.
        metric_labels: Tuples of (metric key, label, format string) that are printed.
    """
    def describe(name, state_dict_path, metrics_path):
        state_dict = torch.load(state_dict_path, map_location=device)
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        total = sum(p.numel() for p in state_dict.values())
        nonzero = sum(torch.count_nonzero(p).item() for p in state_dict.values())
        size_mb = os.path.getsize(state_dict_path) / (1024 * 1024)
        lines = [f'--- {name} ---', f'Total Parameters:         {total:,}',
                 f'Non-Zero Parameters:      {nonzero:,}', f'Saved model size:         {size_mb:.2f} MB']
        for key, label, fmt in metric_labels:
            lines.append(f'{label + ":":<26}{fmt.format(metrics[key])}')
        return '\n'.join(lines)
    print('\n' + describe(names[0], state_dict_path_a, metrics_path_a))
    print('\n' + describe(names[1], state_dict_path_b, metrics_path_b) + '\n')

