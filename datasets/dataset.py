"""
USE CASE: Training Dataset Pipe
To train a model in PyTorch, we need a robust loader that can search files, open them, 
apply complex data augmentations, and serve batches to the GPU.
This file implements a PyTorch Dataset that scans a parent folder containing 'real' and 'fake' subfolders,
and applies custom training or evaluation image transforms.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeepfakeDataset(Dataset):
    """
    USE CASE: Custom Dataset Loader
    Inherits from PyTorch's native Dataset class. It maps out image files and labels:
    - 0 -> Real
    - 1 -> Fake
    """
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Root directory path of the dataset.
            transform (callable, optional): Optional transforms to apply to the images.
            mode (string): Directory scope to run in ('train', 'val', or 'test').
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        self.samples = []
        
        # Build path to specific run split: e.g., 'datasets/data/train'
        mode_dir = os.path.join(root_dir, mode)
        
        # Read files if the directory exists
        if os.path.exists(mode_dir):
            # Class mapping: 'real' folder gets label 0, 'fake' folder gets label 1
            for label, class_name in enumerate(['real', 'fake']):
                class_dir = os.path.join(mode_dir, class_name)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        # Filter to only parse standard image files
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(class_dir, img_name), label))
                            
    def __len__(self):
        """
        Returns the total number of images in this dataset split.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        USE CASE: Batch Fetching & Processing
        Triggered when training loops load an item at a specific index.
        Opens the image, processes it, and returns the (ImageTensor, Label) pair.
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(is_train=True):
    """
    USE CASE: Augmentation & Regularization
    For training:
    We apply randomized transformations (cropping, horizontal flips, color fluctuations).
    This simulates changing environments, shadows, and face angles, preventing the model 
    from overfitting or memorizing exact pixels.
    
    For evaluation:
    We do NOT apply random changes; we only resize and normalize to get deterministic predictions.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224), # Randomly crops a 224x224 chunk from the 256x256 image
            transforms.RandomHorizontalFlip(), # Flips image with a 50% probability
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Slight color jittering
            transforms.ToTensor(),
            # Normalize with ImageNet standard values
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
