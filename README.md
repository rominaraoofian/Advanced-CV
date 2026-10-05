# Computer Vision Projects


| # | Project | Dataset | Approach | Key Result |
|---|---------|---------|----------|------------|
| 1 | [Classical Vision & Regression](#1-classical-vision--regression) | Oxford-IIIT Pet | NumPy, HOG + PCA, GPR | RBF kernel best on IoU |
| 2 | [Decision Trees & Random Forests](#2-decision-trees--random-forests) | Airlines (OpenML 42493) | NumPy | 64.5% test accuracy |
| 3 | [Bird Species Classification](#3-bird-species-classification) | CUB-200-2011 (25 classes) | CNN ablations, ≤ 500k params | 43.00% val accuracy |
| 4 | [Generative Models](#4-generative-models) | CelebA | Pix2Pix, AC-GAN | 13.08 dB PSNR, FID 23.85 |

---

## 1. Classical Vision & Regression

Bounding box regression and cat-vs-dog classification on the Oxford-IIIT Pet dataset. Models, kernels, and metrics are implemented from scratch in NumPy.

**Pipeline**
- **Features:** HOG descriptors (9 orientations, 8×8 cells, 2×2 blocks) on normalized grayscale images
- **Dimensionality reduction:** PCA, keeping components that explain more than 75% of variance
- **Normalization:** Z-score using training-set statistics

**Models**
- **Linear Regression:** closed-form OLS with a bias term, `W* = (XᵀX)⁻¹XᵀY`
- **Gaussian Process Regression:** one GP per box coordinate, with Linear, Polynomial (p=2), and RBF kernels. Hyperparameters (length-scale, degree, noise) are tuned by minimizing the negative log marginal likelihood with L-BFGS-B in log-space
- **Kernel Logistic Regression:** RBF-kernel binary classifier trained with batch gradient descent and early stopping (patience of 200 epochs)

**Findings**
- Linear Regression and Linear GPR performed comparably and stably, so the PCA-reduced HOG space maps roughly linearly to box coordinates
- The Polynomial kernel overfit in the high-dimensional space
- The RBF kernel achieved the best accuracy and IoU by capturing local non-linear structure

---

## 2. Decision Trees & Random Forests

Binary flight-delay prediction on ~27,000 samples from the OpenML Airlines dataset. Both models are implemented from scratch in NumPy.

**Preprocessing**
- 8 features (6 categorical, 2 numerical)
- High-cardinality categoricals (`Airline`, `AirportFrom`, `AirportTo`) are ordinal-encoded to avoid the dimensionality blow-up of one-hot encoding
- Numerical features are normalized

**Models**
- **Decision Tree:** entropy and information-gain splitting, tuned over `max_depth ∈ {5, 10, 15, 20}` and `min_samples_split ∈ {2, 10, 20, 50, 100}`
- **Random Forest:** bagging with random feature subsampling at each split

| Model | Configuration | Test Accuracy |
|-------|---------------|:-------------:|
| Decision Tree | `max_depth=5`, `min_samples_split=2` | 63.6% |
| Random Forest | 50 trees, `max_depth=10`, `max_features=4` | **64.5%** |

Deeper single trees overfit badly. Bagging and feature subsampling decorrelate the trees, which lets the forest use deeper trees without the same penalty.

---

## 3. Bird Species Classification

A controlled ablation study on a 25-class subset of CUB-200-2011 under a 500k parameter budget. The baseline is a 3-stage LeNet-style CNN (365,849 params, 33.08% val accuracy).

| Variant | Params | Best Epoch | Val Acc. |
|---------|-------:|:----------:|---------:|
| Baseline (ReLU + BatchNorm + MaxPool) | 365,849 | 24 | 33.08% |
| AvgPool | 365,849 | 49 | 33.85% |
| Strided Convolutions | 365,849 | 15 | 33.08% |
| LeakyReLU (α=0.1) | 365,849 | 21 | 38.46% |
| GELU | 365,849 | 10 | 35.38% |
| No Normalization | 365,401 | 15 | 31.54% |
| Group Normalization (G=8) | 365,849 | 20 | 35.38% |
| Depthwise Separable Convolutions | 284,026 | 30 | 31.54% |
| **YourNet (final)** | **490,837** | n/a | **43.00%** |

**Takeaways**
- **Downsampling:** strided convolutions converged fastest (epoch 15 vs. 24) at similar accuracy
- **Activation:** LeakyReLU gave the largest single gain (+5.38%), likely by avoiding dead neurons
- **Normalization:** GroupNorm beat BatchNorm, which suffers from noisy statistics with small batches
- **Efficiency:** depthwise separable convolutions cut parameters by ~1.3× (365k → 284k) for a small accuracy drop

**YourNet**
- Aspect-ratio-preserving augmentation: pad to 500×500, then RandomResizedCrop to 128×128, plus ColorJitter and horizontal flips
- 5 convolutional stages with LeakyReLU
- Dropout in the classifier head and 2×2 adaptive average pooling to limit dense-layer parameters
- **Result:** 43.00% validation accuracy, nearly +10% over the baseline

---

## 4. Generative Models

Two generative tasks on CelebA.

### Image Inpainting (Pix2Pix)
- **Architecture:** 4-stage U-Net generator (64 → 512 channels) with a 7×7 PatchGAN discriminator
- **Masks:** random rectangles covering 10–30% of the image, generated on the fly
- **Loss:** `L_G = L_cGAN + 10·L1_global + 100·L1_masked`, where the masked term is normalized over the masked region only
- **Results** (19,962 test images): **13.08 dB PSNR**, **0.5088 mean L1**

### Attribute-Conditioned Synthesis (AC-GAN)
- **Task:** generate 64×64 faces conditioned on 5 attributes: `Smiling`, `Eyeglasses`, `Male`, `BlondHair`, `Young`
- **Conditioning:** the latent vector and attribute vector are concatenated at the input (`h₀ = [z; y] ∈ ℝ¹³³`) and fed to a 4-stage transposed-convolution generator. This avoided the mode collapse seen with Conditional BatchNorm
- **Discriminator:** dual-headed, predicting real/fake and multi-label attributes
- **Results:** **FID 23.85**; strong controllability (Smiling 94.50% accuracy / 0.9413 F1, Male 90.58% / 0.9153). Compositional generation was evaluated across all 32 attribute combinations

---

## Authors

- Romina Raoofian
- Shive Sinaei
