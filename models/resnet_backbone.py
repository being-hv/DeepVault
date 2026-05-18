"""
USE CASE: Feature Extraction Backbone
Instead of training a model from scratch to recognize shapes, edges, and textures, 
we use a pre-trained ResNet50 model (trained on millions of ImageNet images) as a "backbone".
We chop off the final classification head, leaving a high-powered encoder that turns an image 
into a highly dense vector representing visual features.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50FeatureExtractor, self).__init__()
        # Load the default pre-trained weights for ResNet50 from torchvision
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Remove the final fully-connected (classification) layer.
        # We only want the convolutional features, not the generic 1000-class predictions.
        # resnet.children() gives all top-level layers; we slice out everything up to the final FC layer.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet50's final pooling output size is 2048 dimensions.
        self.output_dim = 2048 
        
    def forward(self, x):
        """
        Forward pass:
        Takes an image tensor (Batch, 3, 224, 224) and outputs dense feature features (Batch, 2048).
        """
        x = self.features(x)
        # Flatten the features from shape (Batch, 2048, 1, 1) into (Batch, 2048)
        x = x.view(x.size(0), -1) 
        return x
