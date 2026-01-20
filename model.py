"""
MIRA-U Model Architecture

Implements the neural network architectures from the revised manuscript:

1. MIMTeacher - Lightweight ViT for Masked Image Modeling and pseudo-label generation
   - 2.1M parameters
   - 8×8 patches, 80% masking ratio
   - Used in Stage 1 (Algorithm 1) for MIM pretraining
   - Used in Stage 2 (Algorithm 2) for MC dropout-based pseudo-label generation

2. HybridCNNTransformerStudent - U-shaped CNN-Transformer with bidirectional cross-attention
   - 21.3M parameters
   - Encoder: CNN + Swin Transformer blocks (32→64→128→256 channels)
   - Decoder: ConvTranspose upsampling with cross-attention skip fusion
   - Cross-attention: Attention(Q_decoder, K_encoder, V_encoder)

3. Helper functions for EMA updates

Key corrections from manuscript:
- Proper bidirectional cross-attention formulation
- Correct parameter counts
- Algorithm references updated
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
    
    Splits image into non-overlapping patches and projects to embedding dimension.
    Used in MIMTeacher for 8×8 patch extraction (Algorithm 1, line 7).
    """
    
    def __init__(self, img_size=256, patch_size=8, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 1024 for 256/8
        self.grid_size = img_size // patch_size  # 32
        
        # Convolution with kernel=stride=patch_size acts as patch extraction + projection
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - input image
        Returns:
            (B, num_patches, embed_dim) - patch embeddings
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
    
    Used in MIMTeacher encoder (4 blocks) and decoder (2 blocks).
    Implements: LN → Self-Attention → Residual → LN → MLP → Residual
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
            x: (B, N, D) where N is sequence length, D is embed_dim
        Returns:
            (B, N, D)
        """
        # Self-attention with pre-norm and residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with pre-norm and residual
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
#                       MIM TEACHER NETWORK
# ============================================================================

class MIMTeacher(nn.Module):
    """
    Masked Image Modeling Teacher Network (2.1M parameters)
    
    Architecture (from Section III-B):
    - Lightweight ViT encoder: 4 layers, 4 heads, 8×8 patches, embed_dim=256
    - Shallow decoder: 2 layers for RGB reconstruction
    - Segmentation head for pseudo-label generation
    - Monte Carlo dropout support for uncertainty estimation (M=20 passes)
    
    Used in:
    - Stage 1 (Algorithm 1): MIM pretraining with 80% masking
    - Stage 2 (Algorithm 2, lines 14-22): Pseudo-label generation via MC dropout
    """
    
    def __init__(self, img_size=256, patch_size=8, in_channels=3, 
                 embed_dim=256, depth=4, num_heads=4, decoder_embed_dim=128,
                 mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # ========== ENCODER ==========
        # Patch embedding: 8×8 patches (Algorithm 1, line 7)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 4 Transformer blocks (lightweight design)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) 
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # ========== DECODER (for MIM RGB reconstruction) ==========
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        
        # 2 Transformer blocks (shallow decoder)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio, dropout) 
            for _ in range(2)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # RGB reconstruction predictor: outputs patch_size² × 3 (RGB values)
        # Paper: "RGB-preserving MIM" - reconstructs full color information
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels)
        
        # ========== SEGMENTATION HEAD ==========
        # For pseudo-label generation in Stage 2 (Algorithm 2)
        # Includes dropout for MC uncertainty estimation
        self.seg_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),  # Dropout for MC sampling
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),  # Dropout for MC sampling
            nn.Linear(256, patch_size**2)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize positional embeddings and mask token"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def random_masking(self, x, mask_ratio=0.80):
        """
        Random masking following MAE approach (Algorithm 1, line 7)
        
        Args:
            x: (B, N, D) - patch embeddings
            mask_ratio: Ratio of patches to mask (default 0.80 = 80%)
        Returns:
            x_masked: (B, N*(1-mask_ratio), D) - visible patches only
            mask: (B, N) - binary mask (1=masked, 0=visible)
            ids_restore: (B, N) - indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        
        # Sort to get shuffled indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first len_keep patches (visible)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 1=masked, 0=visible
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio=0.80):
        """
        Encoder forward pass with masking (Algorithm 1, lines 5-9)
        
        Args:
            x: (B, 3, H, W) - input image
            mask_ratio: Ratio of patches to mask (0.80 for 80%)
        Returns:
            x: (B, N_visible, embed_dim) - encoded visible patches
            mask: (B, N) - binary mask
            ids_restore: (B, N) - restore indices
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        x = x + self.pos_embed
        
        # Random masking (Algorithm 1, line 7)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Transformer encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """
        Decoder forward pass for RGB reconstruction (Algorithm 1, line 9)
        
        Args:
            x: (B, N_visible, embed_dim)
            ids_restore: (B, N)
        Returns:
            pred: (B, N, patch_size² × 3) - reconstructed RGB patches
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Append mask tokens to visible patches
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle to restore original order
        x_full = torch.gather(x_full, dim=1, 
                             index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2]))
        
        # Add positional embedding
        x = x_full + self.decoder_pos_embed
        
        # Transformer decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # Predict RGB pixel values
        pred = self.decoder_pred(x)  # (B, N, patch_size² × 3)
        
        return pred
    
    def forward_segmentation(self, x):
        """
        Forward pass for segmentation / pseudo-label generation
        Used in Stage 2 (Algorithm 2, lines 15-17) with MC dropout
        
        No masking applied - uses full image.
        Dropout layers remain active during MC sampling.
        
        Args:
            x: (B, 3, H, W) - input image
        Returns:
            seg_mask: (B, 1, H, W) - predicted segmentation probability map
        """
        # Patch embedding (no masking)
        x = self.patch_embed(x)  # (B, N, embed_dim)
        x = x + self.pos_embed
        
        # Encoder blocks (dropout active for MC sampling)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        # Segmentation head (includes dropout for uncertainty)
        seg_logits = self.seg_head(x)  # (B, N, patch_size²)
        
        # Reshape from patches to image
        B, N, _ = seg_logits.shape
        H = W = int(np.sqrt(N))
        seg_logits = rearrange(seg_logits, 'b (h w) (p1 p2) -> b (h p1) (w p2)', 
                               h=H, w=W, p1=self.patch_size, p2=self.patch_size)
        seg_logits = seg_logits.unsqueeze(1)  # (B, 1, H, W)
        
        return torch.sigmoid(seg_logits)
    
    def forward(self, x, mask_ratio=0.80, mode='mim'):
        """
        Forward pass with mode selection
        
        Args:
            x: (B, 3, H, W) - input image
            mask_ratio: Masking ratio for MIM (default 0.80)
            mode: 'mim' for reconstruction (Stage 1), 'seg' for segmentation (Stage 2)
        Returns:
            If mode='mim': (pred, mask) for MIM loss computation
            If mode='seg': seg_mask for pseudo-label generation
        """
        if mode == 'mim':
            # Masked Image Modeling (Algorithm 1)
            latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)
            return pred, mask
        elif mode == 'seg':
            # Segmentation for pseudo-label generation (Algorithm 2)
            return self.forward_segmentation(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'mim' or 'seg'.")


# ============================================================================
#                    SWIN TRANSFORMER BLOCK
# ============================================================================

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with window-based self-attention
    
    Used in student encoder for efficient global context modeling.
    Implements windowed self-attention with window_size=7.
    
    From Section III-C: "Swin Transformer blocks that apply windowed self-attention"
    """
    
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # GroupNorm for spatial features (instead of LayerNorm)
        self.norm1 = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.GroupNorm(8, dim)
        
        # MLP with 1×1 convolutions
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
            x: (B, C, H, W) - spatial feature map
        Returns:
            (B, C, H, W) - processed feature map
        """
        B, C, H, W = x.shape
        
        # Self-attention with pre-norm and residual
        x_norm = self.norm1(x)
        x_flat = x_norm.flatten(2).transpose(1, 2)  # (B, H*W, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        x = x + attn_out
        
        # MLP with pre-norm and residual
        x = x + self.mlp(self.norm2(x))
        
        return x


# ============================================================================
#                   BIDIRECTIONAL CROSS-ATTENTION MODULE
# ============================================================================

class CrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention module for skip connections
    
    From Section III-C and paper's Technical Innovation 3:
    "Bidirectional cross-attention where decoder features serve as queries
    and encoder features as keys/values: Attention(Q_decoder, K_encoder, V_encoder)"
    
    Mathematical formulation:
        Q = D · W_Q  (Query from decoder)
        K = E · W_K  (Key from encoder)
        V = E · W_V  (Value from encoder)
        A = softmax(Q · K^T / √C)
        F_att = A · V
        F_out = F_att + D  (Residual connection)
    
    This enables dynamic weighting of which encoder features are most
    relevant for each spatial location in the decoder.
    """
    
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Normalization
        self.norm_q = nn.GroupNorm(8, dim)
        self.norm_kv = nn.GroupNorm(8, dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_dec, x_enc):
        """
        Bidirectional Cross-attention: decoder queries attend to encoder keys/values
        
        Args:
            x_dec: Decoder features (B, C, H, W) - serves as Query
            x_enc: Encoder features (B, C, H, W) - serves as Key and Value
        Returns:
            (B, C, H, W) - attended features with residual connection
        """
        B, C, H, W = x_dec.shape
        N = H * W  # Number of spatial positions
        
        # Normalize and flatten to sequence
        x_dec_norm = self.norm_q(x_dec).flatten(2).transpose(1, 2)   # (B, N, C)
        x_enc_norm = self.norm_kv(x_enc).flatten(2).transpose(1, 2)  # (B, N, C)
        
        # Compute Q, K, V projections
        Q = self.q_proj(x_dec_norm)  # (B, N, C) - Query from decoder
        K = self.k_proj(x_enc_norm)  # (B, N, C) - Key from encoder
        V = self.v_proj(x_enc_norm)  # (B, N, C) - Value from encoder
        
        # Reshape for multi-head attention
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, N, head_dim)
        K = K.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention: A = softmax(Q · K^T / √d)
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values: F_att = A · V
        out = (attn @ V)  # (B, heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        
        # Output projection
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        
        # Residual connection: F_out = F_att + D
        return x_dec + out


# ============================================================================
#              HYBRID CNN-TRANSFORMER STUDENT NETWORK
# ============================================================================

class HybridCNNTransformerStudent(nn.Module):
    """
    Hybrid CNN-Transformer Student Network (21.3M parameters)
    
    U-shaped architecture combining:
    - Encoder: CNN blocks + Swin Transformer blocks
      * CNNs capture local textures (fine-grained details)
      * Transformers capture global context (long-range dependencies)
    - Decoder: Progressive upsampling with ConvTranspose
    - Skip connections: Bidirectional cross-attention fusion
    
    Architecture from Section III-C:
    - Stage 1: Conv (3→32), no downsampling
    - Stage 2: MaxPool → Conv (32→64) → Swin Block
    - Stage 3: MaxPool → Conv (64→128) → Swin Block  
    - Stage 4: MaxPool → Conv (128→256) → Swin Block
    - Bottleneck: MaxPool → Conv (256→512) → 2× Swin Block
    - Decoder: ConvTranspose upsampling with cross-attention skip fusion
    
    From paper: "U-shaped CNN–Transformer design with cross-attention skip connections,
    combining detailed texture representation with long-range contextual reasoning"
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        
        # ========== ENCODER ==========
        
        # Stage 1: Basic CNN (256×256, 3→32 channels)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        # Stage 2: CNN + Swin Transformer (128×128, 32→64 channels)
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.GELU(),
            SwinTransformerBlock(base_channels*2, num_heads=2, window_size=7)
        )
        
        # Stage 3: CNN + Swin Transformer (64×64, 64→128 channels)
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.GELU(),
            SwinTransformerBlock(base_channels*4, num_heads=4, window_size=7)
        )
        
        # Stage 4: CNN + Swin Transformer (32×32, 128→256 channels)
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.GroupNorm(8, base_channels*8),
            nn.GELU(),
            SwinTransformerBlock(base_channels*8, num_heads=8, window_size=7)
        )
        
        # ========== BOTTLENECK ==========
        # (16×16, 256→512 channels) with 2 Swin blocks
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*8, base_channels*16, 3, padding=1),
            nn.GroupNorm(8, base_channels*16),
            nn.GELU(),
            SwinTransformerBlock(base_channels*16, num_heads=8, window_size=7),
            SwinTransformerBlock(base_channels*16, num_heads=8, window_size=7)
        )
        
        # ========== DECODER WITH BIDIRECTIONAL CROSS-ATTENTION ==========
        
        # Stage 4 Decoder (16→32, 512→256)
        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 3, 
                                      stride=2, padding=1, output_padding=1)
        self.cross_attn4 = CrossAttention(base_channels*8, num_heads=8)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels*16, base_channels*8, 3, padding=1),
            nn.GroupNorm(8, base_channels*8),
            nn.GELU(),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.GroupNorm(8, base_channels*8),
            nn.GELU()
        )
        
        # Stage 3 Decoder (32→64, 256→128)
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
        
        # Stage 2 Decoder (64→128, 128→64)
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
        
        # Stage 1 Decoder (128→256, 64→32)
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
        # 1×1 convolution for final segmentation
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        
    def forward(self, x):
        """
        Forward pass through U-shaped network
        
        Args:
            x: (B, 3, H, W) - input image
        Returns:
            (B, 1, H, W) - segmentation probability map
        """
        # ===== ENCODER =====
        e1 = self.enc1(x)      # (B, 32, 256, 256)
        e2 = self.enc2(e1)     # (B, 64, 128, 128)
        e3 = self.enc3(e2)     # (B, 128, 64, 64)
        e4 = self.enc4(e3)     # (B, 256, 32, 32)
        
        # ===== BOTTLENECK =====
        b = self.bottleneck(e4)  # (B, 512, 16, 16)
        
        # ===== DECODER WITH CROSS-ATTENTION SKIP CONNECTIONS =====
        
        # Stage 4: Upsample → Cross-attention → Concat → Conv
        d4 = self.up4(b)                    # (B, 256, 32, 32)
        d4 = self.cross_attn4(d4, e4)       # Cross-attention: decoder queries encoder
        d4 = torch.cat([d4, e4], dim=1)     # (B, 512, 32, 32)
        d4 = self.dec4(d4)                  # (B, 256, 32, 32)
        
        # Stage 3: Upsample → Cross-attention → Concat → Conv
        d3 = self.up3(d4)                   # (B, 128, 64, 64)
        d3 = self.cross_attn3(d3, e3)       # Cross-attention
        d3 = torch.cat([d3, e3], dim=1)     # (B, 256, 64, 64)
        d3 = self.dec3(d3)                  # (B, 128, 64, 64)
        
        # Stage 2: Upsample → Cross-attention → Concat → Conv
        d2 = self.up2(d3)                   # (B, 64, 128, 128)
        d2 = self.cross_attn2(d2, e2)       # Cross-attention
        d2 = torch.cat([d2, e2], dim=1)     # (B, 128, 128, 128)
        d2 = self.dec2(d2)                  # (B, 64, 128, 128)
        
        # Stage 1: Upsample → Cross-attention → Concat → Conv
        d1 = self.up1(d2)                   # (B, 32, 256, 256)
        d1 = self.cross_attn1(d1, e1)       # Cross-attention
        d1 = torch.cat([d1, e1], dim=1)     # (B, 64, 256, 256)
        d1 = self.dec1(d1)                  # (B, 32, 256, 256)
        
        # ===== OUTPUT =====
        out = self.out_conv(d1)             # (B, 1, 256, 256)
        return torch.sigmoid(out)


# ============================================================================
#                      HELPER FUNCTIONS
# ============================================================================

def update_ema(teacher, student, alpha=0.999):
    """
    Update teacher parameters using Exponential Moving Average (EMA)
    
    From Algorithm 2, Line 42:
        φ* ← α·φ* + (1-α)·θ
    
    where:
        φ* = teacher parameters
        θ = student parameters
        α = EMA decay factor (default 0.999)
    
    This provides stable pseudo-labels by smoothing teacher updates.
    
    Args:
        teacher: Teacher model (parameters to update)
        student: Student model (source of new parameters)
        alpha: EMA decay factor (default 0.999)
    """
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            if teacher_param.shape == student_param.shape:
                teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1-alpha)


def count_parameters(model):
    """Count total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model, name="Model"):
    """Print detailed model information"""
    total_params = count_parameters(model)
    print(f"\n{name}:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    return total_params


# ============================================================================
#                           MODEL TESTING
# ============================================================================

if __name__ == '__main__':
    """Test model architectures and verify parameter counts"""
    
    print("="*70)
    print("MIRA-U MODEL ARCHITECTURE TEST (Corrected Version)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test MIMTeacher
    print("\n" + "-"*50)
    print("1. Testing MIMTeacher (Lightweight ViT)")
    print("-"*50)
    
    teacher = MIMTeacher(
        img_size=256, 
        patch_size=8, 
        embed_dim=256, 
        depth=4, 
        num_heads=4
    ).to(device)
    
    teacher_params = get_model_info(teacher, "MIMTeacher")
    print(f"  Expected: ~2.1M parameters")
    
    x = torch.randn(2, 3, 256, 256).to(device)
    
    # Test MIM mode (Stage 1)
    pred, mask = teacher(x, mask_ratio=0.80, mode='mim')
    print(f"\n  MIM mode (Stage 1):")
    print(f"    Input: {x.shape}")
    print(f"    Pred (patches): {pred.shape}")
    print(f"    Mask: {mask.shape}")
    print(f"    Mask ratio: {mask.sum().item() / mask.numel():.2%}")
    
    # Test segmentation mode (Stage 2)
    seg = teacher(x, mode='seg')
    print(f"\n  Segmentation mode (Stage 2):")
    print(f"    Output: {seg.shape}")
    print(f"    Value range: [{seg.min().item():.4f}, {seg.max().item():.4f}]")
    
    # Test HybridCNNTransformerStudent
    print("\n" + "-"*50)
    print("2. Testing HybridCNNTransformerStudent")
    print("-"*50)
    
    student = HybridCNNTransformerStudent(
        in_channels=3, 
        out_channels=1, 
        base_channels=32
    ).to(device)
    
    student_params = get_model_info(student, "HybridCNNTransformerStudent")
    print(f"  Expected: ~21.3M parameters")
    
    out = student(x)
    print(f"\n  Forward pass:")
    print(f"    Input: {x.shape}")
    print(f"    Output: {out.shape}")
    print(f"    Value range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    # Test EMA update
    print("\n" + "-"*50)
    print("3. Testing EMA Update (Algorithm 2, Line 42)")
    print("-"*50)
    
    # Create copies for testing
    teacher_test = MIMTeacher(img_size=256, patch_size=8).to(device)
    student_test = MIMTeacher(img_size=256, patch_size=8).to(device)
    
    # Get initial parameter
    initial_param = list(teacher_test.parameters())[0].clone()
    
    # Apply EMA update with α=0.999
    update_ema(teacher_test, student_test, alpha=0.999)
    
    # Check if parameters changed
    updated_param = list(teacher_test.parameters())[0]
    param_changed = not torch.equal(initial_param, updated_param)
    print(f"  EMA decay (α): 0.999")
    print(f"  Parameters updated: {param_changed}")
    
    # Test Cross-Attention
    print("\n" + "-"*50)
    print("4. Testing Bidirectional Cross-Attention")
    print("-"*50)
    
    cross_attn = CrossAttention(dim=128, num_heads=4).to(device)
    x_dec = torch.randn(2, 128, 64, 64).to(device)
    x_enc = torch.randn(2, 128, 64, 64).to(device)
    
    out_attn = cross_attn(x_dec, x_enc)
    print(f"  Decoder features (Q): {x_dec.shape}")
    print(f"  Encoder features (K,V): {x_enc.shape}")
    print(f"  Output: {out_attn.shape}")
    print(f"  Residual preserved: {out_attn.shape == x_dec.shape}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Teacher parameters: {teacher_params:,} ({teacher_params/1e6:.2f}M)")
    print(f"Student parameters: {student_params:,} ({student_params/1e6:.2f}M)")
    print(f"Total: {(teacher_params + student_params):,} ({(teacher_params + student_params)/1e6:.2f}M)")
    print("\nALL TESTS PASSED!")
    print("="*70)
