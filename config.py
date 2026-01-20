"""
MIRA-U Configuration File 

This file contains all hyperparameters and settings for training and evaluation.
Aligned with the revised manuscript and Algorithms 1 & 2.

Key Parameters from Manuscript:
- Stage 1 (MIM Pretraining): 50 epochs, mask_ratio=0.80, patch_size=8×8
- Stage 2 (SSL Training): 150 epochs, M=20, τᵤ=0.15, κ=0.5, α=0.999
- Loss weights: λ_D=0.5, λ_B=0.5, λ_U=1.0, λ_C=0.1, γ=0.01
- Ramp-up: β(t) with 80 epoch warm-up
"""

import os


class Config:
    """MIRA-U Configuration - Aligned with Revised Manuscript"""
    
    # ========== DATA SETTINGS ==========
    # Path to dataset directory containing 'images' and 'masks' subdirectories
    data_dir = './data/ISIC2016'
    
    # Ratio of labeled data to use (0.1 = 10%, 0.25 = 25%, 0.5 = 50%)
    # Paper experiments: 10%, 20%, 30%, 50%
    labeled_ratio = 0.5
    
    # Batch size for training
    # Paper: batch_size=4 for Stage 2 (Section III-E)
    batch_size = 4
    
    # Batch size for MIM pretraining (can be larger)
    # Paper: batch_size=16 for Stage 1
    mim_batch_size = 16
    
    # Image size (height and width)
    # Paper uses: 256×256
    image_size = 256
    
    # Number of worker threads for data loading
    num_workers = 4
    
    # Train/test split ratio (using official ISIC-2016 split)
    train_test_split = 0.8  # 80% train, 20% test
    
    
    # ========== MODEL ARCHITECTURE SETTINGS ==========
    # Teacher (MIMTeacher) settings - Lightweight ViT
    # Paper Section III-B: "lightweight Vision Transformer with 2.1M parameters"
    patch_size = 8           # 8×8 patches (Algorithm 1, line 7)
    embed_dim = 256          # Embedding dimension
    depth = 4                # Number of transformer layers
    num_heads = 4            # Number of attention heads
    decoder_embed_dim = 128  # Decoder embedding dimension
    mlp_ratio = 2.0          # MLP expansion ratio
    dropout = 0.1            # Dropout rate (for MC dropout)
    
    # Student (HybridCNNTransformerStudent) settings
    # Paper: "21.3M parameters"
    base_channels = 32       # Base number of channels (32→64→128→256)
    
    
    # ========== TRAINING SETTINGS ==========
    # Number of epochs for Stage 1 (MIM pretraining)
    # Algorithm 1: E_pretrain = 50
    mim_epochs = 50
    
    # Number of epochs for Stage 2 (semi-supervised training)
    # Algorithm 2: E_train = 150
    train_epochs = 150
    
    # Learning rates
    # Algorithm 1 & 2: η = 0.001
    teacher_lr = 0.001       # Teacher learning rate (Stage 1)
    student_lr = 0.001       # Student learning rate (Stage 2)
    
    # Weight decay for AdamW optimizer
    # Paper Section III-E: weight_decay = 0.05
    weight_decay = 0.05
    
    # Learning rate scheduler patience
    lr_patience = 10
    
    # Gradient clipping max norm
    max_grad_norm = 1.0
    
    
    # ========== MASKED IMAGE MODELING (MIM) SETTINGS ==========
    # Masking ratio for MIM pretraining
    # Algorithm 1: p = 0.80 (80% mask ratio)
    # Paper: "modified masking strategy (80% mask ratio vs. standard 75%)"
    mask_ratio = 0.80
    
    
    # ========== PSEUDO-LABELING SETTINGS ==========
    # Number of Monte Carlo dropout samples for uncertainty estimation
    # Algorithm 2: M = 20
    mc_samples = 20
    
    # Uncertainty scale parameter κ (kappa)
    # Used in confidence weight: w_i = exp(-σ_i/κ)
    # Algorithm 2: κ = 0.5
    kappa = 0.5
    
    # Uncertainty threshold τ_u for filtering
    # Algorithm 2: τ_u = 0.15
    tau_u = 0.15
    
    
    # ========== LOSS FUNCTION WEIGHTS ==========
    # From Algorithm 2 input parameters
    
    # Supervised losses (Section III-C)
    # L_sup = λ_D * L_Dice + λ_B * L_BCE
    lambda_D = 0.5           # Dice loss weight
    lambda_B = 0.5           # Binary Cross-Entropy loss weight
    
    # Unsupervised losses
    lambda_U = 1.0           # Confidence-weighted CE loss weight
    lambda_C = 0.1           # Consistency loss weight
    
    # Entropy regularization weight
    # Algorithm 2: γ = 0.01
    gamma = 0.01
    
    
    # ========== EMA SETTINGS ==========
    # Exponential Moving Average decay factor
    # Formula: φ* ← α·φ* + (1-α)·θ
    # Algorithm 2: α = 0.999
    ema_decay = 0.999
    
    
    # ========== AUGMENTATION SETTINGS ==========
    # Weak augmentation (for teacher pseudo-label generation)
    # Paper: "flips/resize/color-jitter operations"
    weak_aug = {
        'horizontal_flip': 0.5,
        'vertical_flip': 0.5,
        'resize_scale': (0.9, 1.1),
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'p': 0.5
        }
    }
    
    # Strong augmentation (for student training)
    # Paper: "RandAugment + CutOut"
    strong_aug = {
        'horizontal_flip': 0.5,
        'vertical_flip': 0.5,
        'random_rotate90': 0.5,
        'shift_scale_rotate': {
            'shift_limit': 0.1,
            'scale_limit': 0.2,
            'rotate_limit': 30,
            'p': 0.5
        },
        'color_jitter': {
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.15,
            'p': 0.7
        },
        'rand_augment': {
            'n': 2,           # Number of augmentations
            'm': 9,           # Magnitude
            'p': 0.5
        },
        'cutout': {
            'max_holes': 8,
            'max_h_size': 32,
            'max_w_size': 32,
            'fill_value': 0,
            'p': 0.5
        }
    }
    
    
    # ========== RAMP-UP SETTINGS ==========
    # Number of epochs for β(t) ramp-up
    # Algorithm 2, lines 28-32:
    # if t < 80: β(t) = exp(-5(1 - t/80)²)
    # else: β(t) = 1.0
    ramp_up_epochs = 80
    
    
    # ========== SAVE SETTINGS ==========
    # Directory to save checkpoints
    save_dir = './checkpoints'
    
    # Directory to save results and predictions
    results_dir = './results'
    
    # Save checkpoint every N epochs
    save_interval = 20
    
    # Keep only best N checkpoints
    keep_best_n = 3
    
    
    # ========== EVALUATION SETTINGS ==========
    # Threshold for binary segmentation
    seg_threshold = 0.5
    
    # Use Test Time Augmentation (TTA) during evaluation
    use_tta = False
    
    # Save prediction visualizations
    save_predictions = True
    
    # Number of sample predictions to save
    num_samples_to_save = 8
    
    
    # ========== DEVICE SETTINGS ==========
    # Use GPU if available
    use_cuda = True
    
    # GPU device IDs (for multi-GPU training)
    gpu_ids = [0]
    
    # Mixed precision training (faster, less memory)
    use_amp = False
    
    
    # ========== LOGGING SETTINGS ==========
    # Print training stats every N batches
    print_freq = 10
    
    # Enable tensorboard logging
    use_tensorboard = True
    tensorboard_dir = './runs'
    
    
    # ========== REPRODUCIBILITY ==========
    # Random seeds for reproducibility (Paper: seeds 42, 123, 456)
    seed = 42
    
    # Deterministic mode (may be slower)
    deterministic = True
    
    
    def __init__(self):
        """Initialize configuration and create directories"""
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        if self.use_tensorboard:
            os.makedirs(self.tensorboard_dir, exist_ok=True)
    
    def __repr__(self):
        """Print configuration"""
        lines = ["\n" + "="*70]
        lines.append("MIRA-U CONFIGURATION (Revised Manuscript)")
        lines.append("="*70)
        
        sections = {
            'DATA': ['data_dir', 'labeled_ratio', 'batch_size', 'mim_batch_size', 'image_size'],
            'MODEL': ['patch_size', 'embed_dim', 'depth', 'num_heads', 'base_channels'],
            'TRAINING': ['mim_epochs', 'train_epochs', 'teacher_lr', 'student_lr', 'weight_decay'],
            'MIM': ['mask_ratio'],
            'PSEUDO-LABEL': ['mc_samples', 'kappa', 'tau_u'],
            'LOSS WEIGHTS': ['lambda_D', 'lambda_B', 'lambda_U', 'lambda_C', 'gamma'],
            'EMA': ['ema_decay'],
            'RAMP-UP': ['ramp_up_epochs'],
        }
        
        for section, keys in sections.items():
            lines.append(f"\n[{section}]")
            for key in keys:
                value = getattr(self, key, 'N/A')
                lines.append(f"  {key:.<25} {value}")
        
        lines.append("\n" + "="*70)
        return "\n".join(lines)


# ========== EXPERIMENT CONFIGURATIONS ==========
# Pre-defined configurations for different label fractions

class ISIC2016_10_Config(Config):
    """Configuration for ISIC-2016 with 10% labeled data (90 images)"""
    data_dir = './data/ISIC2016'
    labeled_ratio = 0.1
    # N_L = 90, N_U = 630 (assuming 720 training images)


class ISIC2016_20_Config(Config):
    """Configuration for ISIC-2016 with 20% labeled data (180 images)"""
    data_dir = './data/ISIC2016'
    labeled_ratio = 0.2


class ISIC2016_30_Config(Config):
    """Configuration for ISIC-2016 with 30% labeled data (270 images)"""
    data_dir = './data/ISIC2016'
    labeled_ratio = 0.3


class ISIC2016_50_Config(Config):
    """Configuration for ISIC-2016 with 50% labeled data (360 images)"""
    data_dir = './data/ISIC2016'
    labeled_ratio = 0.5
    # Algorithm 2: N_L = 360, N_U = 360


class PH2_Config(Config):
    """Configuration for PH2 dataset (external validation only)"""
    data_dir = './data/PH2'
    # PH2 used for external testing only, no training


class HAM10000_Config(Config):
    """Configuration for HAM10000 dataset (external validation)"""
    data_dir = './data/HAM10000'
    # HAM10000 subset used for external testing


class FastDebugConfig(Config):
    """Fast configuration for debugging"""
    mim_epochs = 2
    train_epochs = 5
    batch_size = 2
    mim_batch_size = 4
    num_workers = 2
    save_interval = 2
    mc_samples = 4  # Faster uncertainty estimation


# ========== USAGE EXAMPLE ==========
if __name__ == '__main__':
    # Default configuration
    config = Config()
    print(config)
    
    print("\n" + "="*70)
    print("KEY PARAMETERS (from Revised Manuscript)")
    print("="*70)
    
    print("\nStage 1 - MIM Pretraining (Algorithm 1):")
    print(f"  Epochs: {config.mim_epochs}")
    print(f"  Mask ratio: {config.mask_ratio} (80%)")
    print(f"  Patch size: {config.patch_size}×{config.patch_size}")
    print(f"  Learning rate: {config.teacher_lr}")
    
    print("\nStage 2 - SSL Training (Algorithm 2):")
    print(f"  Epochs: {config.train_epochs}")
    print(f"  MC dropout passes (M): {config.mc_samples}")
    print(f"  Uncertainty threshold (τᵤ): {config.tau_u}")
    print(f"  Confidence scale (κ): {config.kappa}")
    print(f"  EMA decay (α): {config.ema_decay}")
    print(f"  Ramp-up epochs: {config.ramp_up_epochs}")
    
    print("\nLoss Weights:")
    print(f"  λ_D (Dice): {config.lambda_D}")
    print(f"  λ_B (BCE): {config.lambda_B}")
    print(f"  λ_U (Unsup CE): {config.lambda_U}")
    print(f"  λ_C (Consistency): {config.lambda_C}")
    print(f"  γ (Entropy): {config.gamma}")
