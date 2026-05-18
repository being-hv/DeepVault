"""
USE CASE: Neural Network Adversarial Training Loop
This script coordinates SGAN training. In GAN loops:
1. The Generator takes random noise and attempts to output fake images.
2. The Discriminator takes both real and fake images and classifies them.
3. The losses are computed, backpropagated, and optimizers update model weights.
This script also handles GPU utilization, automated mixed-precision training, and model checkpoint saving.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.sgan import Generator, SGANDiscriminator
from datasets.dataset import DeepfakeDataset, get_transforms
import time

def train_sgan(data_dir, epochs=50, batch_size=32, lr=0.0002, latent_dim=100, save_dir='checkpoints'):
    """
    Core SGAN training sequence.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Enable hardware acceleration: automatically uses CUDA (NVIDIA GPUs) if available, otherwise defaults to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize custom datasets and loaders
    train_dataset = DeepfakeDataset(data_dir, transform=get_transforms(is_train=True), mode='train')
    
    if len(train_dataset) == 0:
        print("Warning: No data found. Running mock training loop.")
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Instantiate Generator and SGAN Discriminator models and push them to memory (GPU/CPU)
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = SGANDiscriminator(num_classes=2).to(device)
    
    # 3. Optimizers: Adam is optimal for training GAN architectures
    # We use lower beta1 momentum value (0.5) to stabilize adversarial oscillations.
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 4. Define mathematical Loss criteria
    # CrossEntropyLoss computes logits distance for classification (Real vs Fake classification)
    classification_loss = nn.CrossEntropyLoss()
    
    # 5. Mixed Precision Training Scaler
    # Mixed precision allows using half-precision floats (float16) on supported GPUs,
    # decreasing training time and memory footprint without sacrificing accuracy.
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            
            # Send batch data to active device (GPU/CPU)
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # ============================================================
            # 1. TRAIN DISCRIMINATOR
            # ============================================================
            # The Discriminator wants to learn to identify:
            # - Real Images -> Class 0
            # - Fake/Generated Images -> Class 1
            opt_d.zero_grad()
            
            # Run operations in autocast block for automatic mixed-precision conversion
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # A. Real Image Evaluation
                validity_real, _ = discriminator(real_imgs)
                real_loss = classification_loss(validity_real, labels) # Target: real labels (0)
                
                # B. Fake Image Generation & Evaluation
                # Generate random noise vector from standard normal distribution
                z = torch.randn(batch_size, latent_dim, device=device)
                # Synthesize fake images (detach from graph so we don't calculate generator gradients here)
                gen_imgs = generator(z)
                
                validity_fake, _ = discriminator(gen_imgs.detach())
                # Create fake target classification labels (Class 1)
                fake_labels = torch.ones(batch_size, dtype=torch.long, device=device)
                fake_loss = classification_loss(validity_fake, fake_labels) # Target: fake labels (1)
                
                # Total Discriminator loss: average of real and fake detection performance
                d_loss = (real_loss + fake_loss) / 2
                
            # Perform backpropagation with gradient scaling
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)
            
            # ============================================================
            # 2. TRAIN GENERATOR
            # ============================================================
            # The Generator wants the discriminator to misclassify fake images as Real (Class 0)
            opt_g.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                validity_gen, _ = discriminator(gen_imgs)
                # Generator loss: Target is Class 0 (Real label), forcing it to create more realistic images
                g_loss = classification_loss(validity_gen, torch.zeros(batch_size, dtype=torch.long, device=device))
                
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update() # Update gradient scaler steps
            
            if i % 50 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                
        # 6. Save Model Checkpoints
        # This records state weights at every epoch so training can be resumed if interrupted.
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'opt_g_state_dict': opt_g.state_dict(),
            'opt_d_state_dict': opt_d.state_dict(),
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./datasets/data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    train_sgan(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
