"""
MIRA-U Configuration File

This file contains all hyperparameters and settings for training and evaluation.
Modify these values to customize your experiments.

Based on the paper's specifications and Table 1.
"""

import os


class Config:
    """MIRA-U Configuration"""
    
    # ========== DATA SETTINGS ==========
    # Path to dataset directory containing 'images' and 'masks' subdirectories
    data_dir = './data/ISIC2016'
    
    # Ratio of labeled data to use (0.1 = 10%, 0.25 = 25%, 0.5 = 50%)
    # Paper experiments: 10%, 25%, 50%
    labeled_ratio = 0.5
    
    # Batch size for training
    # Paper uses: 4, 6, or 8 depending on GPU memory
    batch_size = 4
    
    # Image size (height and width)
    # Paper uses: 256x256
    image_size = 256
    
    # Number of worker threads for data loading
    num_workers = 4
    
    # Train/test split ratio
    train_test_split = 0.8  # 80% train, 20% test
    
    
    # ========== MODEL ARCHITECTURE SETTINGS ==========
    # Teacher (MIMTeacher) settings
    patch_size = 8           # 8×8 patches (paper specification)
    embed_dim = 256          # Embedding dimension (paper: 256)
    depth = 4                # Number of transformer layers (paper: 4)
    num_heads = 4            # Number of attention heads (paper: 4)
    decoder_embed_dim = 128  # Decoder embedding dimension
    mlp_ratio = 2.0          # MLP expansion ratio
    dropout = 0.1            # Dropout rate
    
    # Student (HybridCNNTransformerStudent) settings
    base_channels = 32       # Base number of channels
    
    
    # ========== TRAINING SETTINGS ==========
    # Number of epochs for Stage 1 (MIM pretraining)
    # Paper: 50 epochs
    mim_epochs = 50
    
    # Number of epochs for Stage 2 (semi-supervised training)
    # Paper: 200 epochs
    train_epochs = 200
    
    # Learning rates
    teacher_lr = 0.001       # Teacher learning rate
    student_lr = 0.001       # Student learning rate
    
    # Weight decay for AdamW optimizer
    weight_decay = 0.01
    
    # Learning rate scheduler patience
    lr_patience = 10
    
    # Gradient clipping max norm
    max_grad_norm = 1.0
    
    
    # ========== MASKED IMAGE MODELING (MIM) SETTINGS ==========
    # Masking ratio for MIM pretraining
    # Paper: 75% (0.75)
    mask_ratio = 0.75
    
    
    # ========== PSEUDO-LABELING SETTINGS ==========
    # Number of Monte Carlo dropout samples for uncertainty estimation
    # Paper: M = 8 (Algorithm 2, line 1)
    mc_samples = 8
    
    # Uncertainty scale parameter κ (kappa)
    # Used in confidence weight: w = exp(-σ/κ)
    # Paper: κ = 1.0
    kappa = 1.0
    
    # Uncertainty threshold τ_u for filtering
    # Paper mentions this but doesn't specify exact value
    tau_u = 0.5
    
    
    # ========== LOSS FUNCTION WEIGHTS ==========
    # Supervised losses (Section 3.3)
    lambda_D = 1.0           # Dice loss weight
    lambda_B = 1.0           # Binary Cross-Entropy loss weight
    
    # Unsupervised losses
    lambda_U = 1.0           # Confidence-weighted CE loss weight
    lambda_C = 1.0           # Consistency loss weight
    
    # Entropy regularization weight
    gamma = 0.1              # Entropy loss weight
    
    
    # ========== EMA SETTINGS ==========
    # Exponential Moving Average decay factor
    # Formula: φ ← α*φ + (1-α)*θ
    # Paper: α = 0.99 (Algorithm 1, line 12)
    ema_decay = 0.99
    
    
    # ========== AUGMENTATION SETTINGS ==========
    # Weak augmentation (for teacher)
    weak_aug = {
        'horizontal_flip': 0.5,
        'vertical_flip': 0.5,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'p': 0.5
        }
    }
    
    # Strong augmentation (for student)
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
        'noise_blur': {
            'gauss_noise': (10.0, 50.0),
            'gauss_blur': (3, 7),
            'p': 0.3
        },
        'cutout': {
            'max_holes': 8,
            'max_h_size': 32,
            'max_w_size': 32,
            'p': 0.5
        }
    }
    
    
    # ========== RAMP-UP SETTINGS ==========
    # Number of epochs for β(t) ramp-up
    # The unsupervised loss weight gradually increases from 0 to 1
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
    use_tensorboard = False
    tensorboard_dir = './runs'
    
    
    # ========== REPRODUCIBILITY ==========
    # Random seed for reproducibility
    seed = 42
    
    # Deterministic mode (may be slower)
    deterministic = False
    
    
    def __init__(self):
        """Initialize configuration and create directories"""
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        if self.use_tensorboard:
            os.makedirs(self.tensorboard_dir, exist_ok=True)
    
    def __repr__(self):
        """Print configuration"""
        lines = ["\n" + "="*70]
        lines.append("MIRA-U CONFIGURATION")
        lines.append("="*70)
        
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                lines.append(f"{key:.<30} {value}")
        
        lines.append("="*70 + "\n")
        return "\n".join(lines)


# ========== EXPERIMENT CONFIGURATIONS ==========
# Pre-defined configurations for different experiments

class ISIC2016_10_Config(Config):
    """Configuration for ISIC-2016 with 10% labeled data"""
    data_dir = './data/ISIC2016'
    labeled_ratio = 0.1
    train_epochs = 200


class ISIC2016_25_Config(Config):
    """Configuration for ISIC-2016 with 25% labeled data"""
    data_dir = './data/ISIC2016'
    labeled_ratio = 0.25
    train_epochs = 200


class ISIC2016_50_Config(Config):
    """Configuration for ISIC-2016 with 50% labeled data"""
    data_dir = './data/ISIC2016'
    labeled_ratio = 0.5
    train_epochs = 200


class PH2_Config(Config):
    """Configuration for PH2 dataset (cross-dataset evaluation)"""
    data_dir = './data/PH2'
    labeled_ratio = 0.5
    train_epochs = 200
    batch_size = 4  # PH2 has fewer images


class FastConfig(Config):
    """Fast configuration for debugging"""
    mim_epochs = 5
    train_epochs = 10
    batch_size = 2
    num_workers = 2
    save_interval = 5


# ========== USAGE EXAMPLE ==========
if __name__ == '__main__':
    # Default configuration
    config = Config()
    print(config)
    
    # Experiment-specific configuration
    print("\nISIC-2016 with 10% labels:")
    config_10 = ISIC2016_10_Config()
    print(f"Labeled ratio: {config_10.labeled_ratio}")
    print(f"Training epochs: {config_10.train_epochs}")
    
    print("\nISIC-2016 with 50% labels:")
    config_50 = ISIC2016_50_Config()
    print(f"Labeled ratio: {config_50.labeled_ratio}")
    print(f"Training epochs: {config_50.train_epochs}")
    
    print("\nFast debug configuration:")
    config_fast = FastConfig()
    print(f"MIM epochs: {config_fast.mim_epochs}")
    print(f"Training epochs: {config_fast.train_epochs}")
