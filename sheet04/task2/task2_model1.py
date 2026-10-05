import torch
import torch.nn as nn

N_ATTR = 5
Z_DIM = 128

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, n_attr=N_ATTR, base_ch=64):
        super().__init__()
        in_dim = z_dim + n_attr
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim, base_ch * 8 * 4 * 4),
            nn.BatchNorm1d(base_ch * 8 * 4 * 4),
            nn.ReLU(True)
        )
        
        self.main = nn.Sequential(
            # Input: (base_ch * 8) x 4 x 4
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(True),
            # State: (base_ch * 4) x 8 x 8
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),
            # State: (base_ch * 2) x 16 x 16
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),
            # State: base_ch x 32 x 32
            nn.ConvTranspose2d(base_ch, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # State: 3 x 64 x 64
        )
        self.base_ch = base_ch

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), self.base_ch * 8, 4, 4)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, n_attr=N_ATTR, base_ch=64):
        super().__init__()
        # Input: 3 x 64 x 64
        self.main = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, base_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc_source = nn.Linear(base_ch * 8 * 4 * 4, 1)
        self.fc_classes = nn.Linear(base_ch * 8 * 4 * 4, n_attr)

    def forward(self, x, labels=None):

        features = self.main(x)
        features = features.view(features.size(0), -1)
        
        source_logit = self.fc_source(features).squeeze(-1)
        class_logits = self.fc_classes(features)
        
        return source_logit, class_logits
