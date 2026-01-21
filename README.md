# MIRA-U: Confidence-Weighted Semi-Supervised Learning for Skin Lesion Segmentation

PyTorch implementation of the MIRA-U framework for semi-supervised skin lesion segmentation with uncertainty-aware pseudo-labeling and hybrid CNN-Transformer architecture.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Prediction](#prediction)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Two-Stage Training Pipeline** (Algorithms 1 & 2)
  - Stage 1: Masked Image Modeling (MIM) pretraining with **80% masking ratio**
  - Stage 2: Semi-supervised learning with confidence-weighted pseudo-labels
- **Uncertainty-Aware Pseudo-Labeling** using Monte Carlo dropout (M=20 passes)
- **Hybrid CNN-Transformer Student** (21.3M parameters) with bidirectional cross-attention
- **Lightweight ViT Teacher** (2.1M parameters) for efficient pseudo-label generation
- **Confidence Weighting** with uncertainty-based filtering (τᵤ=0.15, κ=0.5)
- **EMA Teacher Updates** (α=0.999) for stable pseudo-labels
- **Comprehensive Metrics** (DSC, IoU, Accuracy, Precision, Recall, Sensitivity, Specificity)
- Test Time Augmentation (TTA) support

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- PyTorch 1.12+

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install albumentations opencv-python numpy tqdm einops
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## 📁 Dataset Preparation

### Supported Datasets
- **ISIC-2016** (primary training/testing)
- **PH2** (external validation)


### Directory Structure

Organize your dataset as follows:

```
data/
└── ISIC2016/
    ├── images/
    │   ├── ISIC_0000001.jpg
    │   ├── ISIC_0000002.jpg
    │   └── ...
    └── masks/
        ├── ISIC_0000001_segmentation.png
        ├── ISIC_0000002_segmentation.png
        └── ...
```

**Important Notes:**
- Images should be RGB color images (`.jpg` format)
- Masks should be binary grayscale images (`.png` format)
- Mask filename format: `{image_name}_segmentation.png`
- Dataset will be automatically split into 80% train / 20% test
- Labeled ratio (10%, 20%, 30%, 50%) is controlled in configuration

### Download ISIC-2016

```bash
# Download from official ISIC Archive
# https://challenge.isic-archive.com/data/#2016

# Expected dataset size:
# - Training: 900 images
# - Testing: 379 images (used for final evaluation)
```

## 🚀 Training

### Quick Start

```bash
# Train with default settings (50% labeled data)
python training.py
```

### Custom Configuration Using config.py

Choose from pre-configured settings:

```python
from config import Config, ISIC2016_50_Config, ISIC2016_10_Config

# Default: 50% labeled data
config = Config()

# Or use specific configuration
config = ISIC2016_10_Config()  # 10% labeled (90 images)
config = ISIC2016_20_Config()  # 20% labeled (180 images)
config = ISIC2016_30_Config()  # 30% labeled (270 images)
config = ISIC2016_50_Config()  # 50% labeled (360 images)

# Print configuration
print(config)
```

### Key Parameters 

```python
from config import Config

config = Config()

# ========== STAGE 1: MIM PRETRAINING (Algorithm 1) ==========
config.mim_epochs = 50              # E_pretrain = 50
config.mask_ratio = 0.80            # p = 0.80 (80% masking)
config.patch_size = 8               # 8×8 patches
config.teacher_lr = 0.001           # η = 0.001
config.mim_batch_size = 16          # Batch size for MIM

# ========== STAGE 2: SSL TRAINING (Algorithm 2) ==========
config.train_epochs = 150           # E_train = 150
config.mc_samples = 20              # M = 20 (MC dropout passes)
config.tau_u = 0.15                 # τᵤ = 0.15 (uncertainty threshold)
config.kappa = 0.5                  # κ = 0.5 (confidence scale)
config.ema_decay = 0.999            # α = 0.999 (EMA decay)
config.ramp_up_epochs = 80          # β(t) ramp-up over 80 epochs
config.batch_size = 4               # Batch size for SSL
config.student_lr = 0.001           # η = 0.001

# ========== LOSS WEIGHTS (Algorithm 2) ==========
config.lambda_D = 0.5               # Dice loss weight
config.lambda_B = 0.5               # BCE loss weight
config.lambda_U = 1.0               # Unsupervised CE weight
config.lambda_C = 0.1               # Consistency loss weight
config.gamma = 0.01                 # Entropy regularization weight
```

### Training Process

The training consists of **two stages**:

#### **Stage 1: MIM Pretraining (Algorithm 1)**
- **Duration**: 50 epochs
- **Objective**: Teacher network learns context-rich representations
- **Method**: Masked Image Modeling with **80% masking ratio**
- **Loss**: L_MIM = (1/|M|) Σ ||Î_m - I_m||₁ (L1 reconstruction loss)
- **Data**: Uses ALL available images (no labels required)
- **Output**: Pretrained teacher parameters φ*

```
Stage 1 Pipeline:
Input → Patch Embedding (8×8) → Random Masking (80%) → 
ViT Encoder (4 layers) → Decoder (2 layers) → RGB Reconstruction
```

#### **Stage 2: Semi-Supervised Training (Algorithm 2)**
- **Duration**: 150 epochs
- **Components**: Teacher (frozen encoder, active segmentation head) + Student
- **Pseudo-Labeling**: 
  - M=20 MC dropout passes on weakly augmented images
  - Uncertainty estimation: σᵢ = √Var(predictions)
  - Confidence weights: wᵢ = exp(-σᵢ/κ)
  - Filtering: Keep only wᵢ ≥ τᵤ
  - Soft pseudo-labels: ỹᵢ = wᵢ · μ̂ᵢ
- **Student Training**: Hybrid CNN-Transformer with strong augmentation
- **EMA Updates**: φ* ← α·φ* + (1-α)·θ every iteration
- **Loss Ramp-Up**: β(t) = exp(-5(1 - t/80)²) for t < 80, else β(t) = 1.0

```
Stage 2 Pipeline:
Labeled: Strong Aug → Student → L_sup (Dice + BCE)
Unlabeled: 
  - Weak Aug → Teacher (MC dropout ×20) → Pseudo-labels ỹ
  - Strong Aug → Student → L_unsup (Confidence-weighted CE)
  - Consistency loss between teacher and student predictions
Total Loss: L = L_sup + β(t)·L_unsup + γ·L_ent
```

### Monitor Training

Training displays:
- **Stage 1**: L_MIM (reconstruction loss)
- **Stage 2**: L_sup, L_unsup, L_ent, β(t), learning rate
- Automatic best model saving
- Periodic checkpoints every 20 epochs

Example output:
```
Stage 1: MIM Pretraining
Epoch 50/50 | L_MIM: 0.0234 | LR: 0.000100

Stage 2: Semi-Supervised Training
Epoch 150/150 Summary:
  L_sup: 0.1234 | L_unsup: 0.0567 | L_ent: 0.0012
  Total Loss: 0.1813 | β(t): 1.000
  Val Dice: 0.9153
  
```

### Fast Debug Mode

For quick testing:

```python
from config import FastDebugConfig

config = FastDebugConfig()  # 2 MIM epochs, 5 SSL epochs
```

## 🔮 Prediction

### Evaluate on Test Set

Evaluate the trained model and generate comprehensive metrics:

```bash
python predict.py
```

This will:
1. Load the best trained model (`./checkpoints/student_best_dice.pth`)
2. Evaluate on the test set
3. Calculate all metrics (DSC, IoU, ACC, PRE, REC, SEN, SPE)
4. Save prediction visualizations (8 samples)
5. Generate metrics report (`./results/evaluation_metrics.txt`)



### Predict Single Image

Predict segmentation mask for a single image:

```python
from predict import MIRAUPredictor

config = {
    'checkpoint_path': './checkpoints/student_best_dice.pth',
    'image_size': 256,
    'base_channels': 32
}

predictor = MIRAUPredictor(config['checkpoint_path'], config)

# Predict with visualization
pred_mask = predictor.predict_single_image(
    image_path='path/to/image.jpg',
    save_path='path/to/output_mask.png',
    visualize=True  # Creates a 3-panel comparison image
)

print(f"Prediction shape: {pred_mask.shape}")
print(f"Value range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
```

### Batch Prediction

Predict segmentation masks for all images in a directory:

```python
from predict import MIRAUPredictor

config = {
    'checkpoint_path': './checkpoints/student_best_dice.pth',
    'image_size': 256,
    'base_channels': 32
}

predictor = MIRAUPredictor(config['checkpoint_path'], config)

# Process entire directory
predictor.batch_predict(
    image_dir='./data/ISIC2016/images/',
    output_dir='./predictions/',
    use_tta=False  # Set True for Test Time Augmentation
)
```

### Test Time Augmentation (TTA)

Improve prediction quality using TTA (averages predictions over multiple augmentations):

```bash
# Edit predict.py and set:
use_tta = True
```

Or in code:

```python
# Evaluate with TTA
metrics = predictor.evaluate(
    test_loader,
    use_tta=True,  # Enables 4 augmentations (original, hflip, vflip, both)
    save_results=True,
    output_dir='./results_tta'
)
```

**TTA Augmentations:**
- Original image
- Horizontal flip
- Vertical flip  
- Both flips

Final prediction = Average of all 4 predictions

### Custom Prediction Pipeline

For advanced use cases:

```python
from predict import MIRAUPredictor
from data_loader import get_test_augmentation
import cv2
import torch

# Initialize predictor
predictor = MIRAUPredictor('./checkpoints/student_best_dice.pth', config)

# Load and preprocess image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = get_test_augmentation(image_size=256)
transformed = transform(image=image)
image_tensor = transformed['image'].unsqueeze(0)

# Predict (choose one method)
pred = predictor.predict(image_tensor)           # Standard prediction
pred_tta = predictor.predict_with_tta(image_tensor)  # With TTA

# Post-process
pred_binary = (pred.cpu().numpy() > 0.5).astype(np.uint8) * 255
cv2.imwrite('output_mask.png', pred_binary.squeeze())
```

### Prediction Output Structure

After running predictions, you'll have:

```
results/
├── evaluation_metrics.txt          # All metrics (DSC, IoU, ACC, etc.)
├── predictions/
│   ├── sample_000.png             # Visualization: input | GT | soft | binary
│   ├── sample_001.png
│   ├── ...
│   └── sample_007.png             # 8 sample predictions
```

Each visualization shows:
1. **Input Image**: Original RGB image
2. **Ground Truth**: True segmentation mask
3. **Prediction (Soft)**: Probability map [0, 1]
4. **Prediction (Binary)**: Thresholded mask (threshold=0.5)

## 📂 Project Structure

```
mira-u/
├── config.py                  # Configuration file (revised parameters)
├── model.py                   # Model architectures (MIMTeacher + HybridStudent)
├── data_loader.py             # Data loading and augmentation
├── training.py                # Two-stage training pipeline
├── predict.py                 # Prediction and evaluation (create this)
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── data/                      # Dataset directory
│   └── ISIC2016/
│       ├── images/            # Raw images (*.jpg)
│       └── masks/             # Ground truth masks (*_segmentation.png)
│
├── checkpoints/               # Saved models
│   ├── teacher_pretrained_best.pth        # Stage 1 best
│   ├── teacher_pretrained_final.pth       # Stage 1 final
│   ├── student_best_dice.pth              # Stage 2 best (by Dice)
│   ├── student_best_loss.pth              # Stage 2 best (by loss)
│   ├── student_final.pth                  # Stage 2 final
│   └── student_epoch_*.pth                # Periodic checkpoints
│
└── results/                   # Prediction outputs
    ├── predictions/
    └── metrics.txt
```

## ⚙️ Configuration Details

### Key Hyperparameters 

| Parameter | Description | Value | Reference |
|-----------|-------------|-------|-----------|
| **Stage 1 (Algorithm 1)** |
| `mim_epochs` | Pretraining epochs | 50 | Algorithm 1 |
| `mask_ratio` | Masking ratio | 0.80 | 80% |
| `patch_size` | Patch size | 8 | 8×8 patches |
| `mim_batch_size` | Batch size | 16 | Section III-E |
| **Stage 2 (Algorithm 2)** |
| `train_epochs` | SSL training epochs | 150 | Algorithm 2 |
| `mc_samples` (M) | MC dropout passes | 20 | Algorithm 2, line 15 |
| `tau_u` (τᵤ) | Uncertainty threshold | 0.15 | Algorithm 2, line 21 |
| `kappa` (κ) | Confidence scale | 0.5 | Algorithm 2, line 19 |
| `ema_decay` (α) | EMA decay factor | 0.999 | Algorithm 2, line 42 |
| `ramp_up_epochs` | β(t) ramp-up duration | 80 | Algorithm 2, lines 28-32 |
| `batch_size` | Batch size | 4 | Section III-E |
| **Model Architecture** |
| `embed_dim` | ViT embedding dim | 256 | Teacher: 2.1M params |
| `depth` | Transformer layers | 4 | Lightweight ViT |
| `num_heads` | Attention heads | 4 | Multi-head attention |
| `base_channels` | CNN base channels | 32 | Student: 21.3M params |
| **Loss Weights** |
| `lambda_D` | Dice loss weight | 0.5 | Algorithm 2 |
| `lambda_B` | BCE loss weight | 0.5 | Algorithm 2 |
| `lambda_U` | Unsupervised CE weight | 1.0 | Algorithm 2 |
| `lambda_C` | Consistency weight | 0.1 | Algorithm 2 |
| `gamma` (γ) | Entropy weight | 0.01 | Algorithm 2 |

### Loss Function Breakdown

#### Stage 1: MIM Pretraining (Algorithm 1)
```
L_MIM = (1/|M|) Σ_{m∈M} ||Î_m - I_m||₁
```
Only reconstruction loss on masked patches.

#### Stage 2: Semi-Supervised Learning (Algorithm 2)

**Supervised Loss (Labeled Data):**
```
L_sup = λ_D · L_Dice + λ_B · L_BCE
```

**Unsupervised Loss (Unlabeled Data):**
```
L_unsup = (λ_U · L_CE_conf + λ_C · L_cons) · β(t)

where:
  L_CE_conf = (1/|U|) Σ CE(ŷᵢ, ỹᵢ)  [Confidence-weighted]
  L_cons = ||ŷ_weak - ŷ_strong||²    [Consistency]
  ỹᵢ = wᵢ · μ̂ᵢ                      [Soft pseudo-labels]
  wᵢ = exp(-σᵢ/κ)                   [Confidence weights]
```

**Entropy Regularization:**
```
L_ent = -Σ ŷ · log(ŷ)
```

**Total Loss:**
```
L_total = L_sup + β(t) · L_unsup + γ · L_ent
```

**Ramp-Up Function β(t):**
```python
if t < 80:
    β(t) = exp(-5 * (1 - t/80)²)
else:
    β(t) = 1.0
```

## 📊 Results

### ISIC-2016 Dataset

#### Table 1: Performance with Different Label Fractions

| Label Fraction | DSC ↑ | IoU ↑ | ACC ↑ | PRE ↑ | REC ↑ | SEN ↑ | SPE ↑ |
|----------------|-------|-------|-------|-------|-------|-------|-------|
| 10% (90 imgs)  | 0.8934 | 0.8156 | 0.9612 | 0.8856 | 0.9024 | 0.9024 | 0.9734 |
| 20% (180 imgs) | 0.9045 | 0.8342 | 0.9658 | 0.8934 | 0.9167 | 0.9167 | 0.9768 |
| 30% (270 imgs) | 0.9089 | 0.8423 | 0.9671 | 0.8989 | 0.9201 | 0.9201 | 0.9781 |
| **50% (360 imgs)** | **0.9153** | **0.8552** | **0.9703** | **0.9013** | **0.9243** | **0.9243** | **0.9812** |

#### Table 2: Cross-Dataset Generalization (Train: ISIC-2016, Test: PH2)

| Method | DSC ↑ | IoU ↑ | ACC ↑ | PRE ↑ | REC ↑ |
|--------|-------|-------|-------|-------|-------|
| MIRA-U | **0.9130** | **0.8632** | **0.9384** | **0.9208** | **0.8691** |

### Model Complexity

| Component | Parameters | Description |
|-----------|------------|-------------|
| MIMTeacher (φ*) | 2.1M | Lightweight ViT (4 layers, 4 heads, 8×8 patches) |
| HybridStudent (θ) | 21.3M | U-Net with CNN + Swin Transformer + Cross-Attention |
| **Total** | **23.4M** | Teacher + Student |

### Training Time (NVIDIA RTX 3090)

| Stage | Duration | Notes |
|-------|----------|-------|
| Stage 1 (MIM) | ~2 hours | 50 epochs, batch_size=16 |
| Stage 2 (SSL) | ~8 hours | 150 epochs, batch_size=4 |
| **Total** | **~10 hours** | For 50% labeled setting |

## 🔬 Algorithm Details

### Algorithm 1: MIM Pretraining

```
Input: Dataset D (all images), masking ratio p=0.80, epochs E=50
Output: Pretrained teacher parameters φ*

1: Initialize teacher φ with random weights
2: for epoch = 1 to E do
3:    for each batch {x} ~ D do
4:        # Patch embedding
5:        P = PatchEmbed(x)              // 8×8 patches
6:        
7:        # Random masking with ratio p
8:        M, ids = RandomMask(P, p=0.80) // 80% masking
9:        
10:       # Forward pass through teacher
11:       Î = Teacher_φ(M)                // Reconstruct RGB
12:       
13:       # Compute MIM loss
14:       L_MIM = (1/|M|) Σ ||Î_m - I_m||₁
15:       
16:       # Update teacher
17:       φ ← φ - η∇_φ L_MIM
18:    end for
19: end for
20: Return φ*
```

### Algorithm 2: Semi-Supervised Student Training

```
Input: Labeled D_L, Unlabeled D_U, pretrained φ*, epochs E=150
Output: Trained student parameters θ*

1: Initialize student θ with random weights
2: for epoch = 1 to E do
3:    # Compute ramp-up coefficient
4:    if epoch < 80 then
5:        β(t) = exp(-5(1 - t/80)²)
6:    else
7:        β(t) = 1.0
8:    end if
9:    
10:   for each batch do
11:       # ===== LABELED DATA =====
12:       (x_i, y_i) ~ D_L  [with strong augmentation]
13:       ŷ_i = Student_θ(x_i)
14:       L_sup = λ_D·Dice(ŷ_i, y_i) + λ_B·BCE(ŷ_i, y_i)
15:       
16:       # ===== UNLABELED DATA =====
17:       x_j ~ D_U  [with strong augmentation]
18:       x̂_j ~ D_U  [with weak augmentation]
19:       
20:       # Generate pseudo-labels with MC dropout (M=20)
21:       μ̂_j, σ_j = MC_Dropout(Teacher_φ*, x̂_j, M=20)
22:       
23:       # Compute confidence weights
24:       w_j = exp(-σ_j / κ)  [κ=0.5]
25:       
26:       # Filter by uncertainty threshold
27:       if w_j ≥ τ_u then  [τ_u=0.15]
28:           ỹ_j = w_j · μ̂_j  [Soft pseudo-labels]
29:           ŷ_j = Student_θ(x_j)
30:           
31:           # Confidence-weighted CE loss
32:           L_unsup = λ_U · CE(ŷ_j, ỹ_j)
33:           
34:           # Consistency loss
35:           ŷ_j_weak = Student_θ(x̂_j)
36:           L_cons = λ_C · ||ŷ_j - ŷ_j_weak||²
37:       end if
38:       
39:       # Entropy regularization
40:       L_ent = γ · (-Σ ŷ_j · log(ŷ_j))
41:       
42:       # Total loss
43:       L = L_sup + β(t)·(L_unsup + L_cons) + L_ent
44:       
45:       # Update student
46:       θ ← θ - η∇_θ L
47:       
48:       # EMA update teacher (every iteration)
49:       φ* ← α·φ* + (1-α)·θ  [α=0.999]
50:   end for
51: end for
52: Return θ*
```

## 🛠 Troubleshooting

### Out of Memory Errors

**Problem**: CUDA out of memory during training

**Solutions**:
```python
# Option 1: Reduce batch size
config.batch_size = 2
config.mim_batch_size = 8

# Option 2: Reduce image size
config.image_size = 224

# Option 3: Reduce MC samples during training
config.mc_samples = 10  # Instead of 20

# Option 4: Enable mixed precision (if supported)
config.use_amp = True
```

### Poor Performance / Not Converging

**Problem**: Model achieves low Dice score (<0.80)

**Checklist**:
1. ✓ Verify data split is correct
   ```bash
   python data_loader.py  # Run test function
   ```

2. ✓ Check mask format (binary: 0 or 255)
   ```python
   import cv2
   mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
   print(f"Unique values: {np.unique(mask)}")  # Should be [0, 255]
   ```

3. ✓ Ensure Stage 1 completed successfully
   ```bash
   ls checkpoints/teacher_pretrained_*.pth  # Should exist
   ```

4. ✓ Verify loss weights are balanced
   ```python
   # Try adjusting if one loss dominates
   config.lambda_D = 0.5
   config.lambda_B = 0.5
   config.lambda_U = 1.0
   ```

5. ✓ Check augmentation is working
   ```python
   # Visualize augmented samples
   from data_loader import get_strong_augmentation
   ```

### Slow Training Speed

**Problem**: Training takes too long

**Solutions**:
```python
# Option 1: Increase num_workers
config.num_workers = 8  # Match CPU cores

# Option 2: Enable pin_memory (already enabled)
# Already set in create_dataloaders()

# Option 3: Reduce MC samples (trade accuracy for speed)
config.mc_samples = 10  # During training only

# Option 4: Use smaller model (not recommended)
config.base_channels = 16  # Reduces student params
```

### Installation Issues

**Problem**: Package installation failures

**Solutions**:
```bash
# For albumentations
pip install albumentations==1.3.1

# For einops
pip install einops

# For CUDA/PyTorch issues
# Verify CUDA version
nvidia-smi
# Install matching PyTorch
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Checkpoint Loading Errors

**Problem**: Cannot load saved checkpoint

**Solutions**:
```python
# Check checkpoint contents
checkpoint = torch.load('checkpoints/student_best_dice.pth')
print(checkpoint.keys())  # Should include 'student_state_dict'

# Load with map_location for CPU/GPU compatibility
checkpoint = torch.load(
    'checkpoints/student_best_dice.pth',
    map_location='cpu'  # or 'cuda:0'
)
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{mira-u2024,
  title={MIRA-U: Confidence-Weighted Semi-Supervised Learning for Skin Lesion Segmentation},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 📄 License

This implementation is for research purposes. Please refer to the original paper for usage guidelines.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Contact

For questions about the implementation:
- Review the code documentation (inline comments)
- Check the configuration file (`config.py`) for parameter explanations
- Refer to Algorithms 1 & 2 in the manuscript
- Open an issue for bugs or feature requests

## 🔗 Related Resources

- **ISIC Archive**: https://challenge.isic-archive.com/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Albumentations**: https://albumentations.ai/docs/


---

**Last Updated**: January 2025  
**Version**: 1.0 
