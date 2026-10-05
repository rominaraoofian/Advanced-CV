import torch
import torch.nn as nn


class InpaintingGenerator(nn.Module):
    """Compact UNet Generator for 64x64 Image Inpainting.
    
    Input shape: (B, 4, 64, 64) - Concatenated masked_image (3ch) and mask (1ch).
    Output shape: (B, 3, 64, 64) - Reconstructed image.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder: Downsampling
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder: Upsampling with Skip Connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.Tanh()  # Output values in [-1, 1]
        )

    def forward(self, x):
        # Encoder forward pass
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder forward pass with skips
        d1 = self.dec1(e4)
        d1_cat = torch.cat([d1, e3], dim=1)  # 256 + 256 = 512 channels
        
        d2 = self.dec2(d1_cat)
        d2_cat = torch.cat([d2, e2], dim=1)  # 128 + 128 = 256 channels
        
        d3 = self.dec3(d2_cat)
        d3_cat = torch.cat([d3, e1], dim=1)  # 64 + 64 = 128 channels
        
        out = self.dec4(d3_cat)
        return out


class InpaintingDiscriminator(nn.Module):
    """PatchGAN Discriminator for Pix2Pix-style Conditional GAN.
    
    Input: Concatenated condition (masked_image + mask) and candidate (real or fake image).
    Input shape: (B, 7, 64, 64)
    Output shape: (B, 1, 7, 7) - Patch validity scores.
    """
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            # layer 1: 7 -> 64
            nn.Conv2d(7, 64, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            
            # layer 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # layer 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # layer 4: 256 -> 1
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)    # -> 7x7 (Patch classification)
        )

    def forward(self, x):
        return self.model(x)
