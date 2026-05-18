"""
USE CASE: Semi-Supervised GAN (SGAN) Architectures
In a standard GAN, a Generator creates fake images and a Discriminator tries to classify them as Real vs Fake.
In a Semi-Supervised GAN (SGAN), the Discriminator performs a dual role:
1. It tries to distinguish between Real and Fake images (unsupervised adversarial loss).
2. It tries to classify the exact class of real images (supervised classification loss - e.g., identifying fine-grained visual classes).
This architecture is extremely powerful when labeled data is scarce but unlabeled data is abundant.
"""

import torch
import torch.nn as nn
from .resnet_backbone import ResNet50FeatureExtractor

class Generator(nn.Module):
    """
    USE CASE: Fake Image Synthesizer
    Takes a 1D vector of random noise (latent space 'z') and upsamples/deconvolves it 
    into a synthetic 3-channel (RGB) image matching our desired 224x224 input shape.
    During training, the generator continuously refines this process to trick the discriminator.
    """
    def __init__(self, latent_dim=100, img_channels=3, img_size=224):
        super(Generator, self).__init__()
        # Initial starting resolution for the dense linear layer (7x7)
        self.init_size = img_size // 32 # 224 / 32 = 7
        
        # Dense input mapping: maps random latent vector to a large block of features
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))
        
        # Deconvolutional sequence: Upsamples 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2), # Upsample to 14x14
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2), # Upsample to 28x28
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2), # Upsample to 56x56
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2), # Upsample to 112x112
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2), # Upsample to 224x224
            nn.Conv2d(32, img_channels, 3, stride=1, padding=1),
            nn.Tanh() # Tanh restricts output pixels to range [-1, 1] for stable GAN training
        )

    def forward(self, z):
        out = self.l1(z)
        # Reshape flat vector back into a 4D feature map: (Batch, Channels, Height, Width)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class SGANDiscriminator(nn.Module):
    """
    USE CASE: Dual Classifier / Detector
    Uses a high-performance pre-trained ResNet50 encoder to pull deep spatial features from the image,
    and then passes them to a classification head which produces logits representing the target classes.
    For standard deepfake detection:
    - Logit 0: Real Image
    - Logit 1: Fake/Manipulated Image
    """
    def __init__(self, num_classes=2):
        super(SGANDiscriminator, self).__init__()
        
        # Load the custom ResNet50 feature extractor backbone
        self.feature_extractor = ResNet50FeatureExtractor(pretrained=True)
        
        # Final classification neural network head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.output_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), # Regularization to prevent overfitting on specific patterns
            nn.Linear(512, num_classes) # Outputs raw logits for each class
        )
        
    def forward(self, img):
        # 1. Feed input image through ResNet convolutional feature layers
        features_2d = self.feature_extractor.features(img)
        
        # 2. Flatten spatial dimensions into a single feature vector
        features_flat = features_2d.view(features_2d.size(0), -1)
        
        # 3. Classify features
        validity_and_class = self.classifier(features_flat)
        
        # We return both the final prediction logits AND the raw 2D feature activations.
        # Returning the 2D feature map is critical for calculating Grad-CAM explainability heatmaps!
        return validity_and_class, features_2d
