"""
MIRA-U Training Pipeline 
Implements the two-stage training procedure from the revised manuscript:
- Stage 1: Masked Image Modeling (MIM) Pretraining (Algorithm 1)
- Stage 2: Semi-Supervised Student Training (Algorithm 2)

Key corrections from manuscript:
- Proper two-stage separation
- Correct ramp-up function: β(t) = exp(-5(1 - t/80)²) for t < 80
- MC dropout with M=20 passes
- Confidence-weighted soft pseudo-labels: ỹᵢ = wᵢ · μ̂ᵢ
- EMA update: φ* ← α·φ* + (1-α)·θ with α=0.999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
import math
from model import MIMTeacher, HybridCNNTransformerStudent, update_ema


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    L_Dice = 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε)
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class MIRAUTrainer:
    """
    MIRA-U Training Pipeline
    
    Implements:
    - Stage 1: MIM pretraining of teacher (Algorithm 1)
    - Stage 2: Semi-supervised student training (Algorithm 2)
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.teacher = MIMTeacher(
            img_size=config['image_size'],
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            dropout=config.get('dropout', 0.1)
        ).to(self.device)
        
        self.student = HybridCNNTransformerStudent(
            in_channels=3,
            out_channels=1,
            base_channels=config['base_channels']
        ).to(self.device)
        
        # Loss functions
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss(reduction='none')  # Per-pixel for weighting
        
        # Print model info
        teacher_params = sum(p.numel() for p in self.teacher.parameters()) / 1e6
        student_params = sum(p.numel() for p in self.student.parameters()) / 1e6
        print(f"Teacher parameters: {teacher_params:.2f}M")
        print(f"Student parameters: {student_params:.2f}M")
        
    def compute_mim_loss(self, pred, target, mask):
        """
        Compute MIM reconstruction loss (Algorithm 1, line 11)
        L_MIM = (1/|M|) Σ_{m∈M} ||Î_m - I_m||₁
        
        Args:
            pred: (B, N, patch_size² * 3) - predicted patches
            target: (B, 3, H, W) - original image
            mask: (B, N) - binary mask (1 = masked, 0 = visible)
        Returns:
            L1 loss on masked patches only
        """
        B, C, H, W = target.shape
        p = self.config['patch_size']
        
        # Patchify target image
        target = target.reshape(B, C, H//p, p, W//p, p)
        target = target.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, p*p*C)
        
        # Compute L1 loss only on masked patches
        loss = (pred - target).abs()
        loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * p * p * C + 1e-8)
        
        return loss
    
    def pretrain_teacher_mim(self, train_loader, epochs=50):
        """
        Stage 1: Pretrain teacher with Masked Image Modeling (Algorithm 1)
        
        Input: Dataset D (all images), mask_ratio p=0.80, epochs=50
        Output: Pretrained teacher parameters φ*
        
        Note: This stage uses ONLY L_MIM. No student, no pseudo-labels,
        no consistency loss, no entropy loss, no EMA updates.
        """
        print("\n" + "="*70)
        print("STAGE 1: Masked Image Modeling (MIM) Pretraining")
        print("="*70)
        print(f"Epochs: {epochs}")
        print(f"Mask ratio: {self.config['mask_ratio']} (80%)")
        print(f"Patch size: {self.config['patch_size']}×{self.config['patch_size']}")
        print("="*70)
        
        # Optimizer for teacher (Algorithm 1: AdamW with η=0.001)
        optimizer = AdamW(
            self.teacher.parameters(),
            lr=self.config['teacher_lr'],
            weight_decay=self.config.get('weight_decay', 0.05)
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        self.teacher.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"MIM Epoch {epoch+1}/{epochs}")
            
            for batch_idx, images in enumerate(pbar):
                if isinstance(images, (list, tuple)):
                    images = images[0]  # Handle different dataloader formats
                images = images.to(self.device)
                
                # Forward pass with masking (Algorithm 1, lines 5-9)
                pred, mask = self.teacher(
                    images, 
                    mask_ratio=self.config['mask_ratio'], 
                    mode='mim'
                )
                
                # Compute MIM loss (Algorithm 1, lines 10-11)
                loss = self.compute_mim_loss(pred, images, mask)
                
                # Backward pass (Algorithm 1, line 12)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'L_MIM': f"{loss.item():.4f}"})
            
            scheduler.step()
            avg_loss = total_loss / num_batches
            
            print(f"Epoch {epoch+1}/{epochs} | L_MIM: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(epoch, 'teacher_pretrained_best.pth', stage=1)
        
        # Save final pretrained teacher (Algorithm 1, line 15: φ* ← φ)
        self.save_checkpoint(epochs-1, 'teacher_pretrained_final.pth', stage=1)
        print(f"\nMIM pretraining completed! Best L_MIM: {best_loss:.4f}")
        print("Pretrained teacher parameters φ* saved.")
        
        return best_loss
    
    def generate_pseudo_labels(self, images, M=20, kappa=0.5, tau_u=0.15):
        """
        Generate uncertainty-aware pseudo-labels using Monte Carlo dropout
        (Algorithm 2, lines 14-24)
        
        Args:
            images: (B, 3, H, W) - weakly augmented images
            M: Number of MC dropout passes (default: 20)
            kappa: Confidence scale κ (default: 0.5)
            tau_u: Uncertainty threshold τᵤ (default: 0.15)
        
        Returns:
            pseudo_labels: (B, 1, H, W) - soft pseudo-labels ỹᵢ = wᵢ · μ̂ᵢ
            confidence: (B, 1, H, W) - confidence weights wᵢ
            mask: (B, 1, H, W) - binary mask where wᵢ ≥ τᵤ
        """
        self.teacher.eval()
        
        # Enable dropout for MC sampling (Algorithm 2, line 15: "dropout enabled")
        for module in self.teacher.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        with torch.no_grad():
            predictions = []
            
            # M stochastic forward passes (Algorithm 2, lines 15-17)
            for m in range(M):
                pred = self.teacher(images, mode='seg')
                predictions.append(pred)
            
            # Stack predictions: (M, B, 1, H, W)
            predictions = torch.stack(predictions, dim=0)
            
            # Compute mean μ̂ᵢ (Algorithm 2, line 18)
            mean_pred = predictions.mean(dim=0)  # (B, 1, H, W)
            
            # Compute variance σ̂²ᵢ (Algorithm 2, line 19)
            # Using unbiased estimator: 1/(M-1) Σ(p - μ)²
            var_pred = predictions.var(dim=0, unbiased=True)  # (B, 1, H, W)
            
            # Standard deviation σ̂ᵢ (Algorithm 2, line 20)
            std_pred = torch.sqrt(var_pred + 1e-8)
            
            # Confidence weights wᵢ = exp(-σ̂ᵢ/κ) (Algorithm 2, line 21)
            confidence = torch.exp(-std_pred / kappa)
            
            # Soft pseudo-labels ỹᵢ = wᵢ · μ̂ᵢ (Algorithm 2, line 22)
            pseudo_labels = confidence * mean_pred
            
            # Thresholding mask mᵢ = 1[wᵢ ≥ τᵤ] (Algorithm 2, line 26)
            mask = (confidence >= tau_u).float()
        
        return pseudo_labels, confidence, mask
    
    def compute_entropy_loss(self, pred):
        """
        Compute entropy regularization loss (Algorithm 2, line 33)
        L_ent = -(1/HW) Σᵢ,c p_S^(i,c) log p_S^(i,c)
        
        For binary segmentation:
        L_ent = -(1/HW) Σᵢ [p log(p) + (1-p) log(1-p)]
        """
        eps = 1e-8
        entropy = -pred * torch.log(pred + eps) - (1 - pred) * torch.log(1 - pred + eps)
        return entropy.mean()
    
    def compute_ramp_up_weight(self, epoch, ramp_up_epochs=80):
        """
        Compute ramp-up weight β(t) for unsupervised loss (Algorithm 2, lines 34-38)
        
        if t < 80:
            β(t) = exp(-5(1 - t/80)²)
        else:
            β(t) = 1.0
        """
        if epoch < ramp_up_epochs:
            # Gaussian ramp-up: β(t) = exp(-5(1 - t/80)²)
            phase = 1.0 - epoch / ramp_up_epochs
            return math.exp(-5.0 * phase * phase)
        return 1.0
    
    def train_student(self, labeled_loader, unlabeled_loader_weak, 
                      unlabeled_loader_strong, val_loader=None, epochs=150):
        """
        Stage 2: Train student with semi-supervised learning (Algorithm 2)
        
        Input: 
            - Labeled set D_L with N_L images
            - Unlabeled set D_U with N_U images  
            - Pretrained teacher T(φ*) from Stage 1
        
        Output: Trained student parameters θ*
        """
        print("\n" + "="*70)
        print("STAGE 2: Semi-Supervised Student Training")
        print("="*70)
        print(f"Epochs: {epochs}")
        print(f"MC dropout passes (M): {self.config['mc_samples']}")
        print(f"Uncertainty threshold (τᵤ): {self.config['tau_u']}")
        print(f"Confidence scale (κ): {self.config['kappa']}")
        print(f"EMA decay (α): {self.config['ema_decay']}")
        print(f"Ramp-up epochs: {self.config.get('ramp_up_epochs', 80)}")
        print("="*70)
        
        # Optimizer for student (Algorithm 2: Adam with η=0.001)
        optimizer = AdamW(
            self.student.parameters(),
            lr=self.config['student_lr'],
            weight_decay=self.config.get('weight_decay', 0.05)
        )
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Loss weights from config
        lambda_D = self.config['lambda_D']  # Dice loss weight
        lambda_B = self.config['lambda_B']  # BCE loss weight
        lambda_U = self.config['lambda_U']  # Unsupervised CE weight
        lambda_C = self.config['lambda_C']  # Consistency weight
        gamma = self.config['gamma']         # Entropy weight
        
        best_loss = float('inf')
        best_dice = 0.0
        
        for epoch in range(epochs):
            self.student.train()
            self.teacher.eval()  # Teacher frozen for gradients
            
            # Initialize epoch losses (Algorithm 2, lines 5-6)
            total_sup_loss = 0
            total_unsup_loss = 0
            total_ent_loss = 0
            total_loss_epoch = 0
            num_batches = 0
            
            # Compute β(t) for this epoch (Algorithm 2, lines 34-38)
            beta = self.compute_ramp_up_weight(epoch, self.config.get('ramp_up_epochs', 80))
            
            # Iterators for unlabeled data
            unlabeled_iter_weak = iter(unlabeled_loader_weak)
            unlabeled_iter_strong = iter(unlabeled_loader_strong)
            
            pbar = tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (labeled_images, labeled_masks) in enumerate(pbar):
                labeled_images = labeled_images.to(self.device)
                labeled_masks = labeled_masks.to(self.device)
                if labeled_masks.dim() == 3:
                    labeled_masks = labeled_masks.unsqueeze(1)
                
                # ===== SUPERVISED BRANCH (Algorithm 2, lines 7-10) =====
                student_pred = self.student(labeled_images)
                
                # L_sup = λ_D · L_Dice + λ_B · L_BCE (Algorithm 2, line 10)
                sup_dice = self.dice_loss(student_pred, labeled_masks)
                sup_bce = F.binary_cross_entropy(student_pred, labeled_masks)
                sup_loss = lambda_D * sup_dice + lambda_B * sup_bce
                
                # ===== UNSUPERVISED BRANCH (Algorithm 2, lines 12-31) =====
                try:
                    unlabeled_weak = next(unlabeled_iter_weak)
                    unlabeled_strong = next(unlabeled_iter_strong)
                except StopIteration:
                    unlabeled_iter_weak = iter(unlabeled_loader_weak)
                    unlabeled_iter_strong = iter(unlabeled_loader_strong)
                    unlabeled_weak = next(unlabeled_iter_weak)
                    unlabeled_strong = next(unlabeled_iter_strong)
                
                if isinstance(unlabeled_weak, (list, tuple)):
                    unlabeled_weak = unlabeled_weak[0]
                if isinstance(unlabeled_strong, (list, tuple)):
                    unlabeled_strong = unlabeled_strong[0]
                    
                unlabeled_weak = unlabeled_weak.to(self.device)
                unlabeled_strong = unlabeled_strong.to(self.device)
                
                # Generate pseudo-labels from teacher (Algorithm 2, lines 14-26)
                pseudo_labels, confidence, conf_mask = self.generate_pseudo_labels(
                    unlabeled_weak,
                    M=self.config['mc_samples'],
                    kappa=self.config['kappa'],
                    tau_u=self.config['tau_u']
                )
                
                # Student predictions (Algorithm 2, lines 23-25)
                student_pred_strong = self.student(unlabeled_strong)  # ŷ on strong aug
                student_pred_weak = self.student(unlabeled_weak)      # ŷ^w on weak aug
                
                # Confidence-weighted unsupervised loss (Algorithm 2, line 27)
                # L_unsup = (1/Σmᵢwᵢ) Σᵢ mᵢwᵢ [λ_U·CE(ỹᵢ, ŷᵢ) + λ_C·||ŷᵢ - ŷ^w_i||²]
                
                # CE loss with soft pseudo-labels
                unsup_ce = F.binary_cross_entropy(
                    student_pred_strong,
                    pseudo_labels.detach(),
                    reduction='none'
                )
                
                # Consistency loss (weak vs strong predictions)
                consistency = F.mse_loss(
                    student_pred_strong,
                    student_pred_weak.detach(),
                    reduction='none'
                )
                
                # Apply confidence weighting and masking
                weight_sum = (conf_mask * confidence).sum() + 1e-8
                unsup_ce_weighted = (unsup_ce * conf_mask * confidence).sum() / weight_sum
                consistency_weighted = (consistency * conf_mask * confidence).sum() / weight_sum
                
                unsup_loss = lambda_U * unsup_ce_weighted + lambda_C * consistency_weighted
                
                # ===== ENTROPY REGULARIZATION (Algorithm 2, line 33) =====
                ent_loss = self.compute_entropy_loss(student_pred)
                
                # ===== TOTAL LOSS (Algorithm 2, line 39) =====
                # L = L_sup + β(t)·L_unsup + γ·L_ent
                total_loss = sup_loss + beta * unsup_loss + gamma * ent_loss
                
                # ===== BACKWARD PASS (Algorithm 2, line 41) =====
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                optimizer.step()
                
                # ===== EMA UPDATE (Algorithm 2, line 42) =====
                # φ* ← α·φ* + (1-α)·θ
                update_ema(self.teacher, self.student, alpha=self.config['ema_decay'])
                
                # Accumulate losses
                total_sup_loss += sup_loss.item()
                total_unsup_loss += unsup_loss.item()
                total_ent_loss += ent_loss.item()
                total_loss_epoch += total_loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'L_sup': f"{sup_loss.item():.3f}",
                    'L_unsup': f"{unsup_loss.item():.3f}",
                    'L_ent': f"{ent_loss.item():.3f}",
                    'β': f"{beta:.2f}"
                })
            
            # Epoch statistics
            avg_sup = total_sup_loss / num_batches
            avg_unsup = total_unsup_loss / num_batches
            avg_ent = total_ent_loss / num_batches
            avg_total = total_loss_epoch / num_batches
            
            scheduler.step(avg_total)
            
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  L_sup: {avg_sup:.4f} | L_unsup: {avg_unsup:.4f} | L_ent: {avg_ent:.4f}")
            print(f"  Total Loss: {avg_total:.4f} | β(t): {beta:.3f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Validation (if provided)
            if val_loader is not None:
                val_dice = self.validate(val_loader)
                print(f"  Val Dice: {val_dice:.4f}")
                
                if val_dice > best_dice:
                    best_dice = val_dice
                    self.save_checkpoint(epoch, 'student_best_dice.pth', stage=2)
                    print(f"  [NEW BEST] Dice: {best_dice:.4f}")
            
            # Save best model by loss
            if avg_total < best_loss:
                best_loss = avg_total
                self.save_checkpoint(epoch, 'student_best_loss.pth', stage=2)
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.get('save_interval', 20) == 0:
                self.save_checkpoint(epoch, f'student_epoch_{epoch+1}.pth', stage=2)
        
        # Save final model (Algorithm 2, line 44: θ* ← θ)
        self.save_checkpoint(epochs-1, 'student_final.pth', stage=2)
        print(f"\nTraining completed! Best loss: {best_loss:.4f}")
        
        return best_loss
    
    def validate(self, val_loader):
        """Validate student model and return Dice score"""
        self.student.eval()
        total_dice = 0
        num_samples = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                
                pred = self.student(images)
                pred_binary = (pred > 0.5).float()
                
                # Compute Dice
                intersection = (pred_binary * masks).sum(dim=(1,2,3))
                union = pred_binary.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
                dice = (2 * intersection + 1) / (union + 1)
                
                total_dice += dice.sum().item()
                num_samples += images.size(0)
        
        return total_dice / num_samples
    
    def save_checkpoint(self, epoch, filename, stage=2):
        """Save model checkpoint"""
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'teacher_state_dict': self.teacher.state_dict(),
            'student_state_dict': self.student.state_dict(),
            'config': self.config
        }
        
        filepath = os.path.join(self.config['save_dir'], filename)
        torch.save(checkpoint, filepath)
        print(f"  Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.student.load_state_dict(checkpoint['student_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint.get('stage', 2)


def main():
    """Main training function with corrected parameters"""
    
    # Configuration aligned with revised manuscript
    config = {
        # Data settings
        'data_dir': './data/ISIC2016',
        'labeled_ratio': 0.5,        # 50% labeled (N_L=360, N_U=360)
        'batch_size': 4,             # Stage 2 batch size
        'mim_batch_size': 16,        # Stage 1 batch size
        'image_size': 256,
        'num_workers': 4,
        
        # Model settings
        'patch_size': 8,             # 8×8 patches
        'embed_dim': 256,
        'depth': 4,
        'num_heads': 4,
        'base_channels': 32,
        'dropout': 0.1,
        
        # Training settings (Algorithm 1 & 2)
        'mim_epochs': 50,            # Stage 1: E_pretrain = 50
        'train_epochs': 150,         # Stage 2: E_train = 150
        'teacher_lr': 0.001,         # η = 0.001
        'student_lr': 0.001,         # η = 0.001
        'weight_decay': 0.05,
        
        # MIM settings (Algorithm 1)
        'mask_ratio': 0.80,          # p = 0.80 (80%)
        
        # Pseudo-labeling settings (Algorithm 2)
        'mc_samples': 20,            # M = 20
        'kappa': 0.5,                # κ = 0.5
        'tau_u': 0.15,               # τᵤ = 0.15
        
        # Loss weights (Algorithm 2)
        'lambda_D': 0.5,             # Dice loss
        'lambda_B': 0.5,             # BCE loss
        'lambda_U': 1.0,             # Unsupervised CE
        'lambda_C': 0.1,             # Consistency
        'gamma': 0.01,               # Entropy
        
        # EMA settings (Algorithm 2)
        'ema_decay': 0.999,          # α = 0.999
        
        # Ramp-up settings (Algorithm 2)
        'ramp_up_epochs': 80,        # β(t) ramp-up over 80 epochs
        
        # Save settings
        'save_dir': './checkpoints',
        'save_interval': 20,
        
        # Device settings
        'use_cuda': True,
    }
    
    print("\n" + "="*70)
    print("MIRA-U TRAINING PIPELINE")
    print("Confidence-Weighted Semi-Supervised Learning for Skin Lesion Segmentation")
    print("="*70)
    
    # Import data loaders (assuming they exist)
    try:
        from data_loader import create_dataloaders, create_mim_dataloader
        
        # Create data loaders
        print("\nCreating data loaders...")
        labeled_loader, unlabeled_loader_weak, unlabeled_loader_strong, test_loader = create_dataloaders(
            data_dir=config['data_dir'],
            labeled_ratio=config['labeled_ratio'],
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            num_workers=config['num_workers']
        )
        
        # Create MIM data loader (uses all images)
        mim_loader = create_mim_dataloader(
            data_dir=config['data_dir'],
            batch_size=config['mim_batch_size'],
            image_size=config['image_size'],
            num_workers=config['num_workers']
        )
        
    except ImportError:
        print("\nWarning: data_loader module not found. Using dummy data for testing.")
        # Create dummy data for testing
        from torch.utils.data import DataLoader, TensorDataset
        
        dummy_images = torch.randn(32, 3, 256, 256)
        dummy_masks = torch.randint(0, 2, (32, 1, 256, 256)).float()
        
        dataset = TensorDataset(dummy_images, dummy_masks)
        labeled_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        unlabeled_loader_weak = DataLoader(TensorDataset(dummy_images), batch_size=4, shuffle=True)
        unlabeled_loader_strong = DataLoader(TensorDataset(dummy_images), batch_size=4, shuffle=True)
        mim_loader = DataLoader(TensorDataset(dummy_images), batch_size=8, shuffle=True)
        test_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Initialize trainer
    trainer = MIRAUTrainer(config)
    
    # ========== STAGE 1: MIM Pretraining ==========
    print("\n" + "="*70)
    print("Starting Stage 1: MIM Pretraining (Algorithm 1)")
    print("="*70)
    trainer.pretrain_teacher_mim(mim_loader, epochs=config['mim_epochs'])
    
    # ========== STAGE 2: Semi-Supervised Training ==========
    print("\n" + "="*70)
    print("Starting Stage 2: Semi-Supervised Training (Algorithm 2)")
    print("="*70)
    trainer.train_student(
        labeled_loader,
        unlabeled_loader_weak,
        unlabeled_loader_strong,
        val_loader=test_loader,
        epochs=config['train_epochs']
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)


if __name__ == '__main__':
    main()
