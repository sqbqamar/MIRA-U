# MIRA-U: Semi-Supervised Skin Lesion Segmentation

PyTorch implementation of the MIRA-U framework for semi-supervised skin lesion segmentation with uncertainty-aware learning and hybrid CNN-Transformer architecture.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Prediction](#prediction)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

## ✨ Features

- **Masked Image Modeling (MIM)** pretraining for teacher network
- **Uncertainty-aware pseudo-labeling** using Monte Carlo dropout
- **Hybrid CNN-Transformer architecture** with cross-attention skip connections
- **Semi-supervised learning** with consistency and entropy regularization
- **Exponential Moving Average (EMA)** teacher updates
- Comprehensive evaluation metrics (DSC, IoU, Accuracy, Precision, Recall)
- Test Time Augmentation (TTA) support

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)

### Install Dependencies

```bash
pip install -r Requirements.txt
```

## 📁 Dataset Preparation

Organize your dataset in the following structure:

```
data/
└── ISIC2016/
    ├── images/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    └── masks/
        ├── image_001_segmentation.png
        ├── image_002_segmentation.png
        └── ...
```

**Important Notes:**
- Images should be RGB (`.jpg`, `.png`)
- Masks should be binary grayscale images
- Mask filename format: `{image_name}_segmentation.png`
- The dataset will be automatically split into train/test (80/20)
- Labeled ratio (10%, 25%, 50%) is controlled in configuration

## 🚀 Training

### Basic Training

```bash
python training.py
```

### Custom Configuration

Edit the `config` dictionary in `train.py`:

```python
config = {
    # Data settings
    'data_dir': './data/ISIC2016',
    'labeled_ratio': 0.5,  # 10%, 25%, or 50%
    'batch_size': 4,
    'image_size': 256,
    
    # Model settings
    'patch_size': 8,
    'embed_dim': 256,
    'depth': 4,
    'num_heads': 4,
    'base_channels': 32,
    
    # Training settings
    'mim_epochs': 50,
    'train_epochs': 200,
    'teacher_lr': 0.001,
    'student_lr': 0.001,
    
    # Pseudo-labeling settings
    'mc_samples': 8,  # Monte Carlo dropout passes
    'kappa': 1.0,     # Uncertainty scale
    
    # Loss weights
    'lambda_D': 1.0,   # Dice loss
    'lambda_B': 1.0,   # BCE loss
    'lambda_U': 1.0,   # Unsupervised CE
    'lambda_C': 1.0,   # Consistency
    'gamma': 0.1,      # Entropy
    
    # EMA settings
    'ema_decay': 0.99,
}
```

### Training Process

The training consists of two stages:

**Stage 1: MIM Pretraining (50 epochs)**
- Teacher network learns representations using Masked Image Modeling
- Reconstructs masked patches to learn context-rich features
- Preserves color information (RGB)

**Stage 2: Semi-Supervised Training (200 epochs)**
- Student network trained with labeled and unlabeled data
- Teacher generates uncertainty-aware pseudo-labels
- EMA updates for stable teacher predictions
- Combines supervised loss + unsupervised consistency + entropy regularization

### Monitor Training

Training progress is displayed with:
- Epoch progress bars
- Loss values (supervised, unsupervised, entropy)
- Best model automatic saving
- Periodic checkpoints every 20 epochs

## 🔮 Prediction

### Evaluate on Test Set

```bash
python predict.py
```

This will:
1. Load the best trained model
2. Evaluate on test set
3. Calculate metrics (DSC, IoU, Accuracy, Precision, Recall)
4. Save prediction visualizations
5. Generate metrics report

### Predict Single Image

```python
from predict import MIRAUPredictor

config = {
    'checkpoint_path': './checkpoints/best_model.pth',
    'image_size': 256,
    'base_channels': 32
}

predictor = MIRAUPredictor(config['checkpoint_path'], config)
pred_mask = predictor.predict_single_image(
    'path/to/image.jpg',
    save_path='path/to/output_mask.png'
)
```

### Batch Prediction

```python
predictor.batch_predict(
    image_dir='path/to/images/',
    output_dir='path/to/outputs/',
    use_tta=False  # Set True for Test Time Augmentation
)
```

### Test Time Augmentation

For improved predictions, enable TTA in configuration:

```python
config['use_tta'] = True
```

TTA performs predictions with:
- Original image
- Horizontal flip
- Vertical flip
- Both flips
- Averages all predictions

## 📂 Project Structure

```
.
├── data_loader.py          # Data loading and augmentation
├── model.py                # MIRA-U architecture
├── training.py                # Training pipeline
├── predict.py              # Prediction and evaluation
├── Requirements.txt        # Dependencies
├── README.md              # This file
│
├── data/                   # Dataset directory
│   └── ISIC2016/
│       ├── images/
│       └── masks/
│
├── checkpoints/           # Saved models
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
│
└── results/               # Prediction outputs
    ├── predictions/
    └── metrics.txt
```

## ⚙️ Configuration Details

### Key Hyperparameters

| Parameter | Description | Default | Paper Value |
|-----------|-------------|---------|-------------|
| `mask_ratio` | MIM masking ratio | 0.75 | 75% |
| `mc_samples` | Monte Carlo dropout passes | 8 | 8 |
| `patch_size` | ViT patch size | 8 | 8×8 |
| `embed_dim` | Embedding dimension | 256 | 256 |
| `depth` | Transformer layers | 4 | 4 |
| `ema_decay` | EMA decay factor (α) | 0.99 | 0.99 |
| `kappa` | Uncertainty scale (κ) | 1.0 | 1.0 |

### Loss Function Weights

According to Algorithm 1:
- **L_MIM**: Reconstruction loss (Stage 1)
- **L_cons**: Consistency loss between teacher/student
- **L_ent**: Entropy regularization
- **L_sup**: Supervised loss (Dice + BCE)
- **L_unsup**: Confidence-weighted pseudo-label loss

Total loss: `L = L_sup + β(t)L_unsup + γL_ent`

Where β(t) ramps up from 0 to 1 over first 80 epochs.

## 📊 Results

### ISIC-2016 Dataset (50% labeled data)

| Metric | Expected Value |
|--------|----------------|
| DSC | 0.9153 |
| IoU | 0.8552 |
| Accuracy | 0.9703 |
| Precision | 0.9013 |
| Recall | 0.9243 |

### Cross-Dataset (PH2, trained on ISIC-2016)

| Metric | Expected Value |
|--------|----------------|
| DSC | 0.9130 |
| IoU | 0.8632 |
| Accuracy | 0.9384 |
| Precision | 0.9208 |
| Recall | 0.8691 |

## 🔬 Key Components

### 1. MIM Teacher Network
- Lightweight ViT encoder (4 layers, 4 heads)
- 8×8 patch embeddings
- Shallow decoder for reconstruction
- Segmentation head for pseudo-label generation

### 2. Monte Carlo Dropout Uncertainty
```python
# M stochastic forward passes
for m in range(M):
    pred_m = teacher(x_weak, dropout=True)
    predictions.append(pred_m)

# Compute mean and variance
μ = mean(predictions)
σ² = variance(predictions)

# Confidence weights
w = exp(-σ/κ)

# Confidence-weighted pseudo-labels
ỹ = w · μ
```

### 3. Hybrid CNN-Transformer Student
- CNN encoder: Captures local textures
- Swin Transformer blocks: Global context with windowed attention
- Cross-attention skip connections: Fuses encoder features
- U-shaped architecture: Progressive upsampling decoder

### 4. Training Strategy
- Weak augmentation (teacher): Flip, resize, color jitter
- Strong augmentation (student): RandAugment + CutOut
- EMA teacher updates: `φ ← αφ + (1-α)θ`
- Ramp-up schedule for unsupervised loss

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` to 2 or 1
- Reduce `image_size` to 224 or 128
- Use gradient accumulation
- Enable mixed precision training

### Poor Performance
- Check data augmentation is working
- Verify mask filename format matches images
- Increase MIM pretraining epochs
- Adjust loss weights (lambda values)
- Try different labeled_ratio

### Slow Training
- Enable `pin_memory=True` in dataloaders
- Increase `num_workers` (default: 4)
- Use GPU if available
- Reduce MC dropout samples during training
- 

## 📄 License

This implementation is for research purposes. Please refer to the original paper for usage guidelines.

## 🤝 Contributing

For issues, questions, or improvements, please open an issue or submit a pull request.

## 📞 Contact

For questions about the implementation:
- Check the paper for theoretical details
- Review the code comments for implementation details
- Open an issue for bugs or improvements

