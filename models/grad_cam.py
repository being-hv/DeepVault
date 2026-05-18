"""
USE CASE: Model Explainability (Grad-CAM)
Deep learning models are often criticised as "black boxes". Gradient-weighted Class Activation Mapping (Grad-CAM) 
solves this by calculating the gradients of the model's prediction with respect to the final convolutional feature layer.
This maps exactly where the model was looking (e.g. around the eyes, nose, mouth) when it made its "Real" or "Fake" decision,
and lets us overlay a glowing heatmap onto the original image.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network model (SGAN Discriminator).
            target_layer: The specific layer to monitor (usually the last Conv2d layer of the backbone).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Set up PyTorch hooks. Hooks are functions triggered automatically 
        # during forward passes (to save layer outputs) and backward passes (to save gradients).
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # Save output feature maps of target layer during forward pass
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Save output gradients during backward pass (backpropagation)
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx):
        """
        USE CASE: Target Layer Gradient Tracking
        Triggers a forward and backward pass, computes weights from the gradients,
        and uses them to construct an attention map.
        """
        # 1. Perform inference on the input image
        model_output, _ = self.model(input_image)
        
        # 2. Reset gradients
        self.model.zero_grad()
        
        # 3. Focus on the chosen prediction category (e.g., Fake) and perform backpropagation
        target = model_output[0][class_idx]
        target.backward()
        
        # 4. Extract saved gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # 5. Global Average Pooling (GAP): Average the gradients for each channel
        # This tells us how important each individual filter channel was in making the final decision.
        weights = np.mean(gradients, axis=(1, 2))
        
        # 6. Compute the weighted combination of all feature channel activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # 7. Apply Rectified Linear Unit (ReLU) activation:
        # We only care about features that positively contribute to the target class decision.
        heatmap = np.maximum(heatmap, 0)
        
        # 8. Normalize heatmap between 0.0 and 1.0
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else heatmap
        
        # 9. Resize the small heatmap back to match the resolution of the original input image
        heatmap = cv2.resize(heatmap, (input_image.shape[3], input_image.shape[2]))
        
        return heatmap
        
    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        USE CASE: Visualization Overlay
        Takes a raw gray-scale normalized heatmap, colors it using a rainbow color-map (JET),
        and blends it directly onto the original image using weighted blending.
        """
        # Convert grayscale heatmap to RGB Jet colormap (red is high focus, blue is no focus)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend the heatmap and original image: Output = (Image * 0.5) + (Heatmap * 0.5)
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
