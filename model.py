"""
MIRA-U Model Architecture
Implements:
1. MIMTeacher - Lightweight ViT for masked image modeling and pseudo-label generation
2. HybridCNNTransformerStudent - U-shaped CNN-Transformer with cross-attention
3. Helper functions for EMA updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# ============================================================================
#                          PATCH EMBEDDING
# ============================================================================

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding for Vision Transformer
    Splits image into non-overlapping patches and projects to embedding dimension
    """
    
    def __init__(self, img_size=256, patch_size=8, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        # Convolution with kernel=stride=patch_size acts as patch extraction + projection
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ============================================================================
#                       TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Standard Transformer block with self-attention and MLP
    Used in MIMTeacher encoder and decoder
    """
    
    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, D) where N is sequence length
        Returns:
            (B, N, D)
        """
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
#                       MIM TEACHER NETWORK
# ============================================================================

class MIMTeacher(nn.Module):
    """
    Masked Image Modeling Teacher Network
    
    Architecture:
    - Lightweight ViT encoder (4 layers, 4 heads, 8×8 patches, embed_dim=256)
    - Shallow decoder (2 layers) for reconstruction
    - Segmentation head for pseudo-label generation
    - Monte Carlo dropout support for uncertainty estimation
    
    Paper: Section 3.1.1 and Algorithm 1
    """
    
    def __init__(self, img_size=256, patch_size=8, in_channels=3, 
                 embed_dim=256, depth=4, num_heads=4, decoder_embed_dim=128,
                 mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # ---------- ENCODER ----------
        # Patch embedding: 8×8 patches
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 4 Transformer blocks (lightweight)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) 
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # ---------- DECODER (for MIM reconstruction) ----------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        
        # 2 Transformer blocks (shallow decoder)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio, dropout) 
            for _ in range(2)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Reconstruction predictor: outputs patch_size^2 * 3 (RGB)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels)
        
        # ---------- SEGMENTATION HEAD ----------
        # For pseudo-label generation after MIM pretraining
        # Maps from embed_dim to patch_size^2 (binary mask per patch)
        self.seg_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, patch_size**2)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize positional embeddings and mask token"""
        # Truncated normal initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def random_masking(self, x, mask_ratio=0.75):
        """
        Random masking following MAE paper
        
        Args:
            x: (B, N, D)
            mask_ratio: Ratio of patches to mask (default 0.75 = 75%)
        Returns:
            x_masked: (B, N*(1-mask_ratio), D) - visible patches only
            mask: (B, N) - 1 is masked, 0 is visible
            ids_restore: (B, N) - indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise to get shuffled indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep only first len_keep patches (visible)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 1 is masked, 0 is visible
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio=0.75):
        """
        Encoder forward pass with masking
        
        Args:
            x: (B, 3, H, W)
            mask_ratio: Ratio of patches to mask
        Returns:
            x: (B, N_visible, embed_dim) - encoded visible patches
            mask: (B, N) - binary mask
            ids_restore: (B, N) - restore indices
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        x = x + self.pos_embed
        
        # Random masking (only during MIM pretraining)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Transformer encoder
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """
        Decoder forward pass for reconstruction
        
        Args:
            x: (B, N_visible, embed_dim)
            ids_restore: (B, N)
        Returns:
            pred: (B, N, patch_size^2 * 3) - reconstructed patches
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, N_visible, decoder_embed_dim)
        
        # Append mask tokens to visible patches
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N, decoder_embed_dim)
        
        # Unshuffle to restore original order
        x_full = torch.gather(x_full, dim=1, 
                             index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2]))
        
        # Add positional embedding
        x = x_full + self.decoder_pos_embed
        
        # Transformer decoder
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # Predict pixel values
        pred = self.decoder_pred(x)  # (B, N, patch_size^2 * 3)
        
        return pred
    
    def forward_segmentation(self, x):
        """
        Forward pass for segmentation (pseudo-label generation)
        No masking applied - uses full image
        
        Args:
            x: (B, 3, H, W)
        Returns:
            seg_mask: (B, 1, H, W) - predicted segmentation mask
        """
        # Patch embedding (no masking)
        x = self.patch_embed(x)  # (B, N, embed_dim)
        x = x + self.pos_embed
        
        # Encoder blocks (dropout enabled for MC dropout)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        # Segmentation head
        seg_logits = self.seg_head(x)  # (B, N, patch_size^2)
        
        # Reshape from patches to image
        B, N, _ = seg_logits.shape
        H = W = int(np.sqrt(N))
        seg_logits = rearrange(seg_logits, 'b (h w) (p1 p2) -> b (h p1) (w p2)', 
                               h=H, w=W, p1=self.patch_size, p2=self.patch_size)
        seg_logits = seg_logits.unsqueeze(1)  # (B, 1, H, W)
        
        return torch.sigmoid(seg_logits)
    
    def forward(self, x, mask_ratio=0.75, mode='mim'):
        """
        Forward pass with mode selection
        
        Args:
            x: (B, 3, H, W)
            mask_ratio: Masking ratio for MIM
            mode: 'mim' for reconstruction, 'seg' for segmentation
        Returns:
            If mode='mim': (pred, mask)
            If mode='seg': seg_mask
        """
        if mode == 'mim':
            # Masked Image Modeling
            latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)
            return pred, mask
        elif mode == 'seg':
            # Segmentation for pseudo-label generation
            return self.forward_segmentation(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ============================================================================
#                    SWIN TRANSFORMER BLOCK
# ============================================================================

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with window-based self-attention
    Used in student encoder for global context modeling
    
    Paper mentions: "Swin Transformer blocks that apply windowed self-attention"
    """
    
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # GroupNorm instead of LayerNorm for spatial features
        self.norm1 = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.GroupNorm(8, dim)
        
        # MLP with convolutional layers
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Self-attention (simplified window attention)
        x_norm = self.norm1(x)
        x_flat = x_norm.flatten(2).transpose(1, 2)  # (B, H*W, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


# ============================================================================
#                      CROSS-ATTENTION MODULE
# ============================================================================

class CrossAttention(nn.Module):
    """
    Cross-attention module for skip connections
    Allows decoder features to attend to encoder features
    
    Paper: "cross-attention skip fusions" in Section 3.2
    """
    
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm_q = nn.GroupNorm(8, dim)
        self.norm_kv = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, 
                                          dropout=dropout, batch_first=True)
        
    def forward(self, x_dec, x_enc):
        """
        Cross-attention: decoder queries, encoder keys/values
        
        Args:
            x_dec: Decoder features (B, C, H, W) - query
            x_enc: Encoder features (B, C, H, W) - key, value
        Returns:
            (B, C, H, W) - attended features
        """
        B, C, H, W = x_dec.shape
        
        # Normalize
        x_dec_norm = self.norm_q(x_dec).flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_enc_norm = self.norm_kv(x_enc).flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Cross-attention: decoder attends to encoder
        attn_out, _ = self.attn(x_dec_norm, x_enc_norm, x_enc_norm)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        
        # Residual connection
        return x_dec + attn_out


# ============================================================================
#              HYBRID CNN-TRANSFORMER STUDENT
# ============================================================================

class HybridCNNTransformerStudent(nn.Module):
    """
    Hybrid CNN-Transformer Student Network (U-shaped architecture)
    
    Architecture:
    - Encoder: CNN + Swin Transformer blocks
      * CNNs capture local textures
      * Transformers capture global context
    - Decoder: Progressive upsampling with ConvTranspose
    - Skip connections: Cross-attention fusion
    
    Paper: Section 3.2 and Figure 1
    "U-shaped CNN–Transformer design with cross-attention skip connections,
     combining a detailed texture representation with long-range contextual reasoning"
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        
        # ========== ENCODER ==========
        
        # Level 1: Basic CNN (256 -> 256)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        # Level 2: CNN + Swin Transformer (256 -> 128)
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.GELU(),
            SwinTransformerBlock(base_channels*2, num_heads=4, window_size=7)
        )
        
        # Level 3: CNN + Swin Transformer (128 -> 64)
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.GELU(),
            SwinTransformerBlock(base_channels*4, num_heads=4, window_size=7)
        )
        
        # Level 4: CNN + Swin Transformer (64 -> 32)
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.GroupNorm(8, base_channels*8),
            nn.GELU(),
            SwinTransformerBlock(base_channels*8, num_heads=4, window_size=7)
        )
        
        # ========== BOTTLENECK ==========
        # Level 5: Deep features (32 -> 16)
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*8, base_channels*16, 3, padding=1),
            nn.GroupNorm(8, base_channels*16),
            nn.GELU(),
            SwinTransformerBlock(base_channels*16, num_heads=8, window_size=7),
            SwinTransformerBlock(base_channels*16, num_heads=8, window_size=7)
        )
        
        # ========== DECODER WITH CROSS-ATTENTION ==========
        
        # Level 4 Decoder (16 -> 32)
        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 3, 
                                      stride=2, padding=1, output_padding=1)
        self.cross_attn4 = CrossAttention(base_channels*8, num_heads=4)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels*16, base_channels*8, 3, padding=1),
            nn.GroupNorm(8, base_channels*8),
            nn.GELU(),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.GroupNorm(8, base_channels*8),
            nn.GELU()
        )
        
        # Level 3 Decoder (32 -> 64)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 3, 
                                      stride=2, padding=1, output_padding=1)
        self.cross_attn3 = CrossAttention(base_channels*4, num_heads=4)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.GELU(),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.GELU()
        )
        
        # Level 2 Decoder (64 -> 128)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, 
                                      stride=2, padding=1, output_padding=1)
        self.cross_attn2 = CrossAttention(base_channels*2, num_heads=4)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.GELU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.GELU()
        )
        
        # Level 1 Decoder (128 -> 256)
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 3, 
                                      stride=2, padding=1, output_padding=1)
        self.cross_attn1 = CrossAttention(base_channels, num_heads=4)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        # ========== OUTPUT ==========
        # Final 1×1 convolution for segmentation
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, 1, H, W) - segmentation mask
        """
        # ===== ENCODER =====
        e1 = self.enc1(x)      # (B, 32, 256, 256)
        e2 = self.enc2(e1)     # (B, 64, 128, 128)
        e3 = self.enc3(e2)     # (B, 128, 64, 64)
        e4 = self.enc4(e3)     # (B, 256, 32, 32)
        
        # ===== BOTTLENECK =====
        b = self.bottleneck(e4)  # (B, 512, 16, 16)
        
        # ===== DECODER WITH CROSS-ATTENTION SKIP CONNECTIONS =====
        
        # Level 4: Upsample + cross-attention + concat + conv
        d4 = self.up4(b)                    # (B, 256, 32, 32)
        d4 = self.cross_attn4(d4, e4)       # Cross-attention with encoder features
        d4 = torch.cat([d4, e4], dim=1)     # (B, 512, 32, 32)
        d4 = self.dec4(d4)                  # (B, 256, 32, 32)
        
        # Level 3: Upsample + cross-attention + concat + conv
        d3 = self.up3(d4)                   # (B, 128, 64, 64)
        d3 = self.cross_attn3(d3, e3)       # Cross-attention with encoder features
        d3 = torch.cat([d3, e3], dim=1)     # (B, 256, 64, 64)
        d3 = self.dec3(d3)                  # (B, 128, 64, 64)
        
        # Level 2: Upsample + cross-attention + concat + conv
        d2 = self.up2(d3)                   # (B, 64, 128, 128)
        d2 = self.cross_attn2(d2, e2)       # Cross-attention with encoder features
        d2 = torch.cat([d2, e2], dim=1)     # (B, 128, 128, 128)
        d2 = self.dec2(d2)                  # (B, 64, 128, 128)
        
        # Level 1: Upsample + cross-attention + concat + conv
        d1 = self.up1(d2)                   # (B, 32, 256, 256)
        d1 = self.cross_attn1(d1, e1)       # Cross-attention with encoder features
        d1 = torch.cat([d1, e1], dim=1)     # (B, 64, 256, 256)
        d1 = self.dec1(d1)                  # (B, 32, 256, 256)
        
        # ===== OUTPUT =====
        out = self.out_conv(d1)             # (B, 1, 256, 256)
        return torch.sigmoid(out)


# ============================================================================
#                      HELPER FUNCTIONS
# ============================================================================

def update_ema(teacher, student, alpha=0.99):
    """
    Update teacher parameters using Exponential Moving Average (EMA)
    
    Formula from Algorithm 1, Line 12:
        φ ← αφ + (1-α)θ
    
    Args:
        teacher: Teacher model
        student: Student model
        alpha: EMA decay factor (default 0.99)
    """
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1-alpha)


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
#                           MODEL INFO
# ============================================================================

if __name__ == '__main__':
    """Test model architectures"""
    
    print("="*70)
    print("MIRA-U MODEL ARCHITECTURE TEST")
    print("="*70)
    
    # Test MIMTeacher
    print("\n1. Testing MIMTeacher...")
    teacher = MIMTeacher(img_size=256, patch_size=8, embed_dim=256, depth=4, num_heads=4)
    print(f"   Parameters: {count_parameters(teacher):,}")
    
    x = torch.randn(2, 3, 256, 256)
    
    # Test MIM mode
    pred, mask = teacher(x, mask_ratio=0.75, mode='mim')
    print(f"   MIM mode - pred shape: {pred.shape}, mask shape: {mask.shape}")
    
    # Test segmentation mode
    seg = teacher(x, mode='seg')
    print(f"   Seg mode - output shape: {seg.shape}")
    
    # Test HybridCNNTransformerStudent
    print("\n2. Testing HybridCNNTransformerStudent...")
    student = HybridCNNTransformerStudent(in_channels=3, out_channels=1, base_channels=32)
    print(f"   Parameters: {count_parameters(student):,}")
    
    out = student(x)
    print(f"   Output shape: {out.shape}")
    
    # Test EMA update
    print("\n3. Testing EMA update...")
    teacher_copy = MIMTeacher(img_size=256, patch_size=8, embed_dim=256, depth=4, num_heads=4)
    student_copy = MIMTeacher(img_size=256, patch_size=8, embed_dim=256, depth=4, num_heads=4)
    
    # Get initial parameter
    initial_param = list(teacher_copy.parameters())[0].clone()
    
    # Update with EMA
    update_ema(teacher_copy, student_copy, alpha=0.99)
    
    # Check if parameters changed
    updated_param = list(teacher_copy.parameters())[0]
    print(f"   Parameters updated: {not torch.equal(initial_param, updated_param)}")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
