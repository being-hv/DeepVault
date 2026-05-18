"""
USE CASE: Aspect-Preserving Image Transformation & Normalization Pipeline
This file handles opening the image, padding it to a perfect square to preserve
aspect ratio and facial proportions, resizing, converting to tensor, and normalizing.
This ensures the entire image is scanned without stretching or cropping.
"""

import io
from PIL import Image
import torch
import torchvision.transforms as transforms

def pad_to_square(image: Image.Image, background_color=(0, 0, 0)) -> Image.Image:
    """
    Pads a rectangular image to a perfect square using neutral borders,
    preventing any stretching, squishing, or loss of visual data.
    """
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        result = Image.new(image.mode, (width, width), background_color)
        # Paste original image centered vertically
        result.paste(image, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(image.mode, (height, height), background_color)
        # Paste original image centered horizontally
        result.paste(image, ((height - width) // 2, 0))
        return result

def get_base_transforms():
    """
    Standard torchvision sequence applied to normalized square tensors.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def process_uploaded_image(image_bytes: bytes) -> torch.Tensor:
    """
    Converts raw binary uploaded files into the exact tensor shape (1, 3, 224, 224) 
    that our PyTorch neural network expects, fully preserving aspect ratios.
    """
    try:
        # Load the raw bytes into a PIL Image and force it to RGB (removes alpha channels)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 1. Pad to perfect square to preserve facial structures
        squared_image = pad_to_square(image)
        
        # 2. Apply the pre-defined normalization transformations
        transform = get_base_transforms()
        tensor = transform(squared_image)
        
        # 3. Add batch dimension: turns shape from [3, 224, 224] to [1, 3, 224, 224]
        return tensor.unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")
