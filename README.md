## Project 1: NumPy Classical Vision & Regression (Oxford-IIIT Pet)

*Implemented strictly from scratch using NumPy for model architectures, kernel functions, and evaluation metrics.*

### Feature Extraction & Dimensionality Reduction
- **HOG Features:** Extracted Histogram-of-Oriented-Gradients descriptors (`orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)`) over grayscale normalized animal images.
- **PCA Dimensionality Reduction:** Analyzed variance curves across candidate components ($n \in [50, 300]$). Retained principal components reaching $>75\%$ explained variance to project raw HOG vectors prior to regression modeling.
- **Z-Score Normalization:** Normalized feature distributions via train split mean ($\mu$) and standard deviation ($\sigma$) to stabilize non-linear kernel classification.

### Model Architectures & Optimization
- **Linear Regression (Closed-Form):** Solved bounding box prediction via Ordinary Least Squares (OLS) with an explicit bias column:
  $$\mathbf{W}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}$$
- **Gaussian Process Regression (GPR):**
  - Fitted independent scalar outputs per bounding box coordinate.
  - Implemented Linear, Polynomial ($p=2$), and Radial Basis Function (RBF) kernels:
    $$K_{\text{linear}}(\mathbf{x}, \mathbf{x}') = \mathbf{x} \mathbf{x}'^T + 1$$
    $$K_{\text{poly}}(\mathbf{x}, \mathbf{x}') = (\max(\mathbf{x} \mathbf{x}'^T + 1, 10^{-9}))^p$$
    $$K_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{\Vert{}\mathbf{x} - \mathbf{x}'\Vert{}^2}{2\ell^2}\right)$$
  - Tuned length-scale $\ell$, degree $p$, and observation noise variance via negative log marginal likelihood (LML) minimization using L-BFGS-B in log-space:
    $$\log p(\mathbf{y}\vert{}\mathbf{X}) = -\frac{1}{2} \mathbf{y}^T \mathbf{K}_y^{-1} \mathbf{y} - \frac{1}{2}\log\vert{}\mathbf{K}_y\vert{} - \frac{n}{2}\log(2\pi)$$
- **Kernel Logistic Regression:** Binary cat vs. dog classifier using an RBF kernel, trained with batch gradient descent and an early stopping patience threshold of 200 epochs on validation loss.

### Empirical Findings
- Linear Regression and Linear Kernel GPR produced comparable, stable results, showing that the principal HOG feature space maps linearly to bounding box coordinates.
- The Polynomial Kernel exhibited signs of overfitting in the high-dimensional space.
- The RBF Kernel achieved the highest predictive accuracy and IoU on test data by capturing localized non-linear boundaries.

---

## Project 2: Decision Trees & Random Forests from Scratch (Airlines Dataset)

*Implemented from scratch using only NumPy to predict binary flight delays on ~27,000 tabular instances from OpenML (dataset ID: 42493).*

### Methodology & Preprocessing
- **Feature Preprocessing:** Evaluated 8 features (6 categorical, 2 numerical). High-cardinality nominal features (`Airline`, `AirportFrom`, `AirportTo`) were ordinal integer encoded to avoid dimensionality explosions associated with one-hot encoding. Numerical and discrete features (`Flight`, `DayOfWeek`, `Time`, `Length`) were normalized and cast to floating-point tensors.
- **Decision Tree Classifier:**
  - Built an Information Gain / Shannon Entropy split-criterion algorithm.
  - Hyperparameter optimization evaluated across $\text{max\_depth} \in \{5, 10, 15, 20\}$ and $\text{min\_samples\_split} \in \{2, 10, 20, 50, 100\}$.
  - **Result:** Optimal test accuracy reached **63.6%** with $\text{max\_depth}=5$ and $\text{min\_samples\_split}=2$. Deeper trees showed severe overfitting.
- **Random Forest Ensemble:**
  - Utilized bootstrap aggregation (bagging) of instances and random feature sub-sampling ($\text{max\_features} = 4$ out of 7) at each candidate split.
  - **Result:** Ensemble test accuracy reached **64.5%** ($n_{\text{estimators}}=50, \text{max\_depth}=10, \text{max\_features}=4$). Bootstrapping and feature subsampling effectively decorrelated tree variance, allowing deeper individual trees without overfitting.

---

## Project 3: Fine-Grained Bird Species Classification (CUB-200-2011)

*Systematic architectural ablation on a 25-class subset of the Caltech-UCSD Birds dataset under a strict $\le 500\text{k}$ parameter budget.*

### Controlled Ablation Studies
Starting from a 3-stage LeNet baseline (365,849 params, 33.08% validation accuracy):
1. **Downsampling Strategies (Task 1):** Compared MaxPool (33.08%), AvgPool (33.85%), and Strided Convolutions (33.08%). Strided convolutions converged fastest (epoch 15 vs. 24) with lower wall-clock overhead (142.8s).
2. **Activation Functions (Task 2):** Replacing standard ReLU with LeakyReLU ($\alpha = 0.1$) provided the largest single-component improvement (+5.38%, reaching **38.46%**) by mitigating dying neurons during fine-grained gradient propagation. GELU converged rapidly (epoch 10) at 35.38%.
3. **Normalization Techniques (Task 3):** Group Normalization ($G=8$, 35.38%) outperformed Batch Normalization (33.08%) and No-Norm (31.54%). GroupNorm avoided the noisy mini-batch statistics typical of small fine-grained sample sizes.
4. **Depthwise Separable Convolutions (Task 4):** Replaced standard 2D convolutions with factorized spatial/pointwise convolutions. Slashed parameters by $1.3\times$ (365k $\rightarrow$ 284k) with only a minor performance tradeoff (31.54% val accuracy).

| Architecture / Ablation Variant | Parameters | Best Val Epoch | Val Accuracy (%) |
| :--- | :---: | :---: | :---: |
| Baseline (ReLU + BatchNorm + MaxPool) | 365,849 | 24 | 33.08% |
| AvgPool | 365,849 | 49 | 33.85% |
| Strided Convolutions | 365,849 | 15 | 33.08% |
| LeakyReLU ($\alpha=0.1$) | 365,849 | 21 | 38.46% |
| GELU | 365,849 | 10 | 35.38% |
| No Normalization | 365,401 | 15 | 31.54% |
| Group Normalization ($G=8$) | 365,849 | 20 | 35.38% |
| Depthwise Separable Convolutions | 284,026 | 30 | 31.54% |
| **YourNet (Proposed Final Architecture)** | **490,837** | — | **43.00%** |

### The Custom `YourNet` Architecture
- **Aspect-Ratio Preserving Augmentation:** Padded inputs to $500\times500$ before RandomResizedCrop to $128\times128$ to avoid distortion, combined with ColorJitter and random horizontal flips.
- **Structural Upgrades:** Scaled network depth to 5 convolutional stages with LeakyReLU activations, inserted Dropout regularization in the classifier head, and constrained dense parameters via $2\times2$ Adaptive Average Pooling.
- **Final Metrics:** Achieved **43.00% validation accuracy** (nearly a +10% improvement over baseline) using 490,837 parameters.

---

## Project 4: Generative Neural Networks (CelebA)

*Deep generative models applied to high-resolution facial datasets conditioned on geometric masks and attribute vectors.*

### Task 1: Context-Aware Image Inpainting
- **Framework:** Pix2Pix-style conditional GAN containing a 4-stage symmetric U-Net Generator ($64 \rightarrow 128 \rightarrow 256 \rightarrow 512$ channels) and a $7\times7$ patch-based PatchGAN discriminator.
- **Masking:** On-the-fly random rectangular masks covering 10%–30% of the image area.
- **Compound Objective Function:**
  $$\mathcal{L}_G = \mathcal{L}_{\text{cGAN}} + 10 \cdot \mathcal{L}_{1, \text{global}} + 100 \cdot \mathcal{L}_{1, \text{masked}}$$
  where $\mathcal{L}_{1, \text{masked}}$ normalizes absolute reconstruction error strictly over the variable masked region.
- **Results:** Evaluated on 19,962 unseen CelebA test images, reaching **13.08 dB PSNR** and **0.5088 mean L1** error. Skip connections preserved unmasked global features while the PatchGAN prevented seam artifacts.

### Task 2: Attribute-Conditioned Face Synthesis
- **Framework:** Auxiliary Classifier GAN (AC-GAN) tailored to synthesize $64\times64$ facial images conditioned on 5 selected attributes: `Smiling`, `Eyeglasses`, `Male`, `BlondHair`, and `Young`.
- **Conditioning Paradigm:** Input-level concatenation ($h_0 = [z; y] \in \mathbb{R}^{133}$) into a 4-stage transposed-convolution generator, preventing the mode collapse observed when using Conditional Batch Normalization (CBN).
- **Discriminator:** Dual-headed CNN predicting both adversarial authenticity ($D_{\text{source}}$) and multi-label attribute presence ($D_{\text{classes}}$).
- **Results:**
  - Realism: Achieved an **FID score of 23.85**.
  - Controllability: Balanced attributes showed strong controllability (`Smiling`: 94.50% accuracy, 0.9413 F1; `Male`: 90.58% accuracy, 0.9153 F1). Evaluated multi-attribute compositional generations across all 32 combinations.

---
**Authors:**
- **Romina Raoofian**
- **Shive Sinaei**
