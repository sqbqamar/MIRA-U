import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
from model import MIMTeacher, HybridCNNTransformerStudent, update_ema
from data_loader import create_dataloaders, create_mim_dataloader


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
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
    """MIRA-U Training Pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.teacher = MIMTeacher(
            img_size=config['image_size'],
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads']
        ).to(self.device)
        
        self.student = HybridCNNTransformerStudent(
            in_channels=3,
            out_channels=1,
            base_channels=config['base_channels']
        ).to(self.device)
        
        # Loss functions
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        # Optimizers
        self.teacher_optimizer = AdamW(
            self.teacher.parameters(),
            lr=config['teacher_lr'],
            weight_decay=config['weight_decay']
        )
        
        self.student_optimizer = AdamW(
            self.student.parameters(),
            lr=config['student_lr'],
            weight_decay=config['weight_decay']
        )
        
        # Schedulers
        self.teacher_scheduler = ReduceLROnPlateau(
            self.teacher_optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.student_scheduler = ReduceLROnPlateau(
            self.student_optimizer, mode='min', factor=0.5, patience=10
        )
        
    def compute_mim_loss(self, pred, target, mask):
        """Compute MIM reconstruction loss"""
        # Patchify target
        B, C, H, W = target.shape
        p = self.config['patch_size']
        target = target.reshape(B, C, H//p, p, W//p, p)
        target = target.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, p*p*C)
        
        # Compute L1 loss only on masked patches
        loss = (pred - target).abs()
        loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
        
        return loss
    
    def pretrain_teacher_mim(self, train_loader, epochs=50):
        """Stage 1: Pretrain teacher with MIM"""
        print("Stage 1: Pretraining teacher with Masked Image Modeling...")
        
        self.teacher.train()
        
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"MIM Epoch {epoch+1}/{epochs}")
            
            for batch_idx, images in enumerate(pbar):
                images = images.to(self.device)
                
                # Forward pass
                pred, mask = self.teacher(images, mask_ratio=self.config['mask_ratio'], mode='mim')
                
                # Compute loss
                loss = self.compute_mim_loss(pred, images, mask)
                
                # Backward pass
                self.teacher_optimizer.zero_grad()
                loss.backward()
                self.teacher_optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            self.teacher_scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Average MIM Loss: {avg_loss:.4f}")
        
        print("MIM pretraining completed!")
    
    def generate_pseudo_labels(self, images, M=8, kappa=1.0):
        """Generate uncertainty-aware pseudo-labels using Monte Carlo dropout"""
        self.teacher.eval()
        
        with torch.no_grad():
            predictions = []
            
            # Enable dropout for MC sampling
            for module in self.teacher.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
            
            # M stochastic forward passes
            for _ in range(M):
                pred = self.teacher(images, mode='seg')
                predictions.append(pred)
            
            # Stack predictions
            predictions = torch.stack(predictions)  # (M, B, 1, H, W)
            
            # Compute mean and variance
            mean_pred = predictions.mean(dim=0)  # (B, 1, H, W)
            var_pred = predictions.var(dim=0)    # (B, 1, H, W)
            std_pred = torch.sqrt(var_pred + 1e-8)
            
            # Compute confidence weights
            confidence = torch.exp(-std_pred / kappa)  # (B, 1, H, W)
            
            # Confidence-weighted pseudo-labels
            pseudo_labels = confidence * mean_pred
            
        return pseudo_labels, confidence
    
    def entropy_loss(self, pred):
        """Compute entropy regularization"""
        eps = 1e-8
        entropy = -pred * torch.log(pred + eps) - (1 - pred) * torch.log(1 - pred + eps)
        return entropy.mean()
    
    def ramp_up_weight(self, epoch, max_epochs, ramp_up_epochs=80):
        """Ramp up function for unsupervised loss weight"""
        if epoch < ramp_up_epochs:
            return float(epoch) / ramp_up_epochs
        return 1.0
    
    def train_student(self, labeled_loader, unlabeled_loader_weak, 
                     unlabeled_loader_strong, epochs=200):
        """Stage 2: Train student with uncertainty-aware pseudo-labeling"""
        print("\nStage 2: Training student with semi-supervised learning...")
        
        # Copy teacher weights to initialize student encoder (optional)
        # Note: This is simplified; in practice, you'd map compatible layers
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.student.train()
            self.teacher.eval()
            
            total_sup_loss = 0
            total_unsup_loss = 0
            total_ent_loss = 0
            
            # Get beta for this epoch
            beta = self.ramp_up_weight(epoch, epochs, ramp_up_epochs=80)
            
            # Iterate over labeled and unlabeled data
            unlabeled_iter_weak = iter(unlabeled_loader_weak)
            unlabeled_iter_strong = iter(unlabeled_loader_strong)
            
            pbar = tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (labeled_images, labeled_masks) in enumerate(pbar):
                labeled_images = labeled_images.to(self.device)
                labeled_masks = labeled_masks.to(self.device).unsqueeze(1)
                
                # ===== Supervised Loss =====
                student_pred = self.student(labeled_images)
                
                sup_dice = self.dice_loss(student_pred, labeled_masks)
                sup_bce = self.bce_loss(student_pred, labeled_masks)
                sup_loss = self.config['lambda_D'] * sup_dice + self.config['lambda_B'] * sup_bce
                
                # ===== Unsupervised Loss =====
                try:
                    unlabeled_images_weak = next(unlabeled_iter_weak).to(self.device)
                    unlabeled_images_strong = next(unlabeled_iter_strong).to(self.device)
                except StopIteration:
                    unlabeled_iter_weak = iter(unlabeled_loader_weak)
                    unlabeled_iter_strong = iter(unlabeled_loader_strong)
                    unlabeled_images_weak = next(unlabeled_iter_weak).to(self.device)
                    unlabeled_images_strong = next(unlabeled_iter_strong).to(self.device)
                
                # Generate pseudo-labels from teacher on weak augmentation
                pseudo_labels, confidence = self.generate_pseudo_labels(
                    unlabeled_images_weak,
                    M=self.config['mc_samples'],
                    kappa=self.config['kappa']
                )
                
                # Student prediction on strong augmentation
                student_pred_unlabeled = self.student(unlabeled_images_strong)
                
                # Confidence-weighted unsupervised loss
                unsup_ce = F.binary_cross_entropy(
                    student_pred_unlabeled,
                    pseudo_labels.detach(),
                    reduction='none'
                )
                unsup_ce = (unsup_ce * confidence).sum() / (confidence.sum() + 1e-8)
                
                # Consistency loss
                consistency = F.mse_loss(student_pred_unlabeled, pseudo_labels.detach())
                
                unsup_loss = self.config['lambda_U'] * unsup_ce + self.config['lambda_C'] * consistency
                
                # ===== Entropy Regularization =====
                ent_loss = self.entropy_loss(student_pred)
                
                # ===== Total Loss =====
                total_loss = sup_loss + beta * unsup_loss + self.config['gamma'] * ent_loss
                
                # Backward pass
                self.student_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.student_optimizer.step()
                
                # Update teacher with EMA
                update_ema(self.teacher, self.student, alpha=self.config['ema_decay'])
                
                # Accumulate losses
                total_sup_loss += sup_loss.item()
                total_unsup_loss += unsup_loss.item()
                total_ent_loss += ent_loss.item()
                
                pbar.set_postfix({
                    'sup': f"{sup_loss.item():.4f}",
                    'unsup': f"{unsup_loss.item():.4f}",
                    'ent': f"{ent_loss.item():.4f}",
                    'beta': f"{beta:.2f}"
                })
            
            # Average losses
            avg_sup_loss = total_sup_loss / len(labeled_loader)
            avg_unsup_loss = total_unsup_loss / len(labeled_loader)
            avg_ent_loss = total_ent_loss / len(labeled_loader)
            avg_total_loss = avg_sup_loss + beta * avg_unsup_loss + self.config['gamma'] * avg_ent_loss
            
            self.student_scheduler.step(avg_total_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Sup Loss: {avg_sup_loss:.4f}")
            print(f"  Unsup Loss: {avg_unsup_loss:.4f}")
            print(f"  Ent Loss: {avg_ent_loss:.4f}")
            print(f"  Total Loss: {avg_total_loss:.4f}")
            
            # Save best model
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"  Saved best model with loss: {best_loss:.4f}")
            
            # Save periodic checkpoints
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        os.makedirs(self.config['save_dir'], exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'teacher_state_dict': self.teacher.state_dict(),
            'student_state_dict': self.student.state_dict(),
            'teacher_optimizer': self.teacher_optimizer.state_dict(),
            'student_optimizer': self.student_optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, os.path.join(self.config['save_dir'], filename))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.student.load_state_dict(checkpoint['student_state_dict'])
        return checkpoint['epoch']


def main():
    """Main training function"""
    
    # Configuration
    config = {
        # Data settings
        'data_dir': './data/ISIC2016',
        'labeled_ratio': 0.5,
        'batch_size': 4,
        'image_size': 256,
        'num_workers': 4,
        
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
        'weight_decay': 0.01,
        
        # MIM settings
        'mask_ratio': 0.75,
        
        # Pseudo-labeling settings
        'mc_samples': 8,
        'kappa': 1.0,
        
        # Loss weights
        'lambda_D': 1.0,   # Dice loss
        'lambda_B': 1.0,   # BCE loss
        'lambda_U': 1.0,   # Unsupervised CE
        'lambda_C': 1.0,   # Consistency
        'gamma': 0.1,      # Entropy
        
        # EMA settings
        'ema_decay': 0.99,
        
        # Save settings
        'save_dir': './checkpoints'
    }
    
    # Create data loaders
    print("Creating data loaders...")
    labeled_loader, unlabeled_loader_weak, unlabeled_loader_strong, test_loader = create_dataloaders(
        data_dir=config['data_dir'],
        labeled_ratio=config['labeled_ratio'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers']
    )
    
    # Create MIM data loader
    mim_loader = create_mim_dataloader(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'] * 2,
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        mask_ratio=config['mask_ratio'],
        num_workers=config['num_workers']
    )
    
    # Initialize trainer
    trainer = MIRAUTrainer(config)
    
    # Stage 1: Pretrain teacher with MIM
    trainer.pretrain_teacher_mim(mim_loader, epochs=config['mim_epochs'])
    
    # Stage 2: Train student with semi-supervised learning
    trainer.train_student(
        labeled_loader,
        unlabeled_loader_weak,
        unlabeled_loader_strong,
        epochs=config['train_epochs']
    )
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()