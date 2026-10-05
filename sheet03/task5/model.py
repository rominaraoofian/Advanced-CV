import torch
import torch.nn as nn
import torch.nn.functional as F

# paste any helper classes here


class YourNet(nn.Module):
    def __init__(self, num_classes=21, f1=32, f2=64, f3=128, f4=128, f5=144):
        super().__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3,  f1, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(f3)

        self.conv4 = nn.Conv2d(f3, f4, kernel_size=3, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(f4)

        self.conv5 = nn.Conv2d(f4, f5, kernel_size=3, padding=1, bias=False)
        self.bn5   = nn.BatchNorm2d(f5)

        self.pool  = nn.AdaptiveAvgPool2d((2, 2))

        self.fc1   = nn.Linear(f5 * 2 * 2, 128)
        self.drop1 = nn.Dropout(0.15)

        self.fc2   = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.15)

        self.fc3   = nn.Linear(64, num_classes)

    def forward(self, x):
        # YOUR FORWARD
        x = self.maxpool(self.lrelu(self.bn1(self.conv1(x))))  # 128x128 -> 64x64
        x = self.maxpool(self.lrelu(self.bn2(self.conv2(x))))  # 64x64   -> 32x32
        x = self.maxpool(self.lrelu(self.bn3(self.conv3(x))))  # 32x32   -> 16x16
        x = self.maxpool(self.lrelu(self.bn4(self.conv4(x))))  # 16x16   -> 8x8
        x = self.maxpool(self.lrelu(self.bn5(self.conv5(x))))  # 8x8     -> 4x4

        # ── Feature Flattening ──
        x = self.pool(x)                                       # Condenses to 2x2
        x = x.view(x.size(0), -1)                              # Flatten to [B, 576]

        # ── Regularized Classifier Head ──
        x = self.drop1(self.lrelu(self.fc1(x)))
        x = self.drop2(self.lrelu(self.fc2(x)))

        return self.fc3(x)
