"""
MIRA-U Data Loader

Implements data loading and augmentation for the two-stage training:
- Stage 1 (MIM Pretraining): All images for masked image modeling
- Stage 2 (SSL Training): Labeled + Unlabeled (weak & strong augmentations)

Aligned with revised manuscript and Algorithms 1 & 2.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class SkinLesionDataset(Dataset):
    """
    Dataset class for skin lesion segmentation
    
    Supports both labeled and unlabeled data.
    For unlabeled data, only returns images (no masks).
    """
    
    def __init__(self, image_dir, mask_dir, image_list, transform=None, is_labeled=True):
        """
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            image_list: List of image filenames
            transform: Albumentations transform
            is_labeled: Whether dataset has labels (masks)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.transform = transform
        self.is_labeled = is_labeled
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_labeled:
            # Load mask for labeled data
            mask_name = img_name.replace('.jpg', '_segmentation.png')
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            return image, mask
        else:
            # For unlabeled data, only return image
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image


def get_weak_augmentation(image_size=256):
    """
    Weak augmentation for teacher pseudo-label generation
    
    From paper: "flips/resize/color-jitter operations"
    Used with weakly augmented images in Algorithm 2, line 16
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_strong_augmentation(image_size=256):
    """
    Strong augmentation for student training
    
    From paper: "RandAugment + CutOut"
    Used with strongly augmented images in Algorithm 2, line 10
    
    Includes:
    - Geometric transforms (flips, rotations, shifts)
    - Color jittering (more aggressive than weak)
    - Random noise/blur
    - Coarse dropout (CutOut)
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                        min_holes=1, min_height=8, min_width=8, 
                        fill_value=0, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_test_augmentation(image_size=256):
    """
    Test time augmentation - only resize and normalize
    No random augmentations for consistent evaluation
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_dataloaders(data_dir, labeled_ratio=0.5, batch_size=4, 
                       image_size=256, num_workers=4, seed=42):
    """
    Create dataloaders for semi-supervised learning (Stage 2)
    
    Implements the data split from Algorithm 2:
    - D_L: Labeled training data with strong augmentation
    - D_U: Unlabeled training data (ALL training images)
      - D_U_weak: With weak augmentation (for teacher pseudo-labels)
      - D_U_strong: With strong augmentation (for student training)
    - D_test: Test data with no augmentation
    
    Args:
        data_dir: Root directory containing 'images' and 'masks' subdirectories
        labeled_ratio: Ratio of labeled data (0.1, 0.2, 0.3, 0.5)
                      Paper experiments: 10%, 20%, 30%, 50%
        batch_size: Batch size (paper uses 4 for Stage 2)
        image_size: Image size for resizing (paper uses 256×256)
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
    
    Returns:
        labeled_loader: Loader for labeled data (D_L)
        unlabeled_loader_weak: Loader for unlabeled data with weak aug
        unlabeled_loader_strong: Loader for unlabeled data with strong aug
        test_loader: Loader for test data
    """
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    
    # Get all image filenames
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    # Split into train and test (80/20 split)
    np.random.seed(seed)
    np.random.shuffle(all_images)
    
    train_size = int(0.8 * len(all_images))
    train_images = all_images[:train_size]
    test_images = all_images[train_size:]
    
    # Split train into labeled and unlabeled
    # Note: Unlabeled set uses ALL training images (following common SSL practice)
    labeled_size = int(labeled_ratio * len(train_images))
    labeled_images = train_images[:labeled_size]
    unlabeled_images = train_images  # Use all training images as unlabeled
    
    print(f"\n{'='*70}")
    print("DATA SPLIT SUMMARY")
    print(f"{'='*70}")
    print(f"Total images: {len(all_images)}")
    print(f"Training images: {len(train_images)}")
    print(f"  - Labeled (D_L): {len(labeled_images)} ({labeled_ratio*100:.0f}%)")
    print(f"  - Unlabeled (D_U): {len(unlabeled_images)} (all training data)")
    print(f"Test images: {len(test_images)}")
    print(f"{'='*70}\n")
    
    # ========== CREATE DATASETS ==========
    
    # Labeled dataset with strong augmentation
    # Algorithm 2, line 10: (x_i, y_i) ~ D_L with strong augmentation
    labeled_dataset = SkinLesionDataset(
        image_dir, mask_dir, labeled_images,
        transform=get_strong_augmentation(image_size),
        is_labeled=True
    )
    
    # Unlabeled dataset with WEAK augmentation (for teacher pseudo-labels)
    # Algorithm 2, line 16: x̂_j with weak augmentation
    unlabeled_dataset_weak = SkinLesionDataset(
        image_dir, mask_dir, unlabeled_images,
        transform=get_weak_augmentation(image_size),
        is_labeled=False
    )
    
    # Unlabeled dataset with STRONG augmentation (for student training)
    # Algorithm 2, line 10: x_j ~ D_U with strong augmentation
    unlabeled_dataset_strong = SkinLesionDataset(
        image_dir, mask_dir, unlabeled_images,
        transform=get_strong_augmentation(image_size),
        is_labeled=False
    )
    
    # Test dataset with no random augmentation
    test_dataset = SkinLesionDataset(
        image_dir, mask_dir, test_images,
        transform=get_test_augmentation(image_size),
        is_labeled=True
    )
    
    # ========== CREATE DATALOADERS ==========
    
    labeled_loader = DataLoader(
        labeled_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    unlabeled_loader_weak = DataLoader(
        unlabeled_dataset_weak, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    unlabeled_loader_strong = DataLoader(
        unlabeled_dataset_strong, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return labeled_loader, unlabeled_loader_weak, unlabeled_loader_strong, test_loader


class MIMDataset(Dataset):
    """
    Dataset for Masked Image Modeling pretraining (Stage 1)
    
    Used in Algorithm 1 for MIM pretraining.
    Returns only images (no masks needed for MIM).
    """
    
    def __init__(self, image_dir, image_list, transform=None):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            transform: Albumentations transform
        """
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image


def get_mim_augmentation(image_size=256):
    """
    Augmentation for MIM pretraining (Stage 1)
    
    From Algorithm 1: Basic augmentations for MIM
    Less aggressive than strong augmentation since masking provides regularization
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_mim_dataloader(data_dir, batch_size=16, image_size=256, 
                          num_workers=4, seed=42):
    """
    Create dataloader for MIM pretraining (Stage 1)
    
    From Algorithm 1: Uses ALL available images (labeled + unlabeled)
    for self-supervised pretraining with masked image modeling.
    
    Args:
        data_dir: Root directory containing 'images' subdirectory
        batch_size: Batch size (paper uses 16 for Stage 1)
        image_size: Image size for resizing (paper uses 256×256)
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
    
    Returns:
        mim_loader: DataLoader for MIM pretraining
    """
    image_dir = os.path.join(data_dir, 'images')
    
    # Use ALL images for MIM pretraining (both labeled and unlabeled)
    # Algorithm 1, line 2: "Input: Dataset D (all images)"
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    print(f"\n{'='*70}")
    print("MIM PRETRAINING DATA")
    print(f"{'='*70}")
    print(f"Total images for MIM: {len(all_images)}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}×{image_size}")
    print(f"{'='*70}\n")
    
    # Create dataset with MIM augmentation
    dataset = MIMDataset(
        image_dir, 
        all_images, 
        transform=get_mim_augmentation(image_size)
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True  # Drop last for stable batch size
    )
    
    return loader


# ============================================================================
#                           TESTING & VALIDATION
# ============================================================================

def test_dataloaders():
    """Test function to verify dataloaders work correctly"""
    print("\n" + "="*70)
    print("TESTING DATA LOADERS")
    print("="*70)
    
    # Test parameters
    data_dir = './data/ISIC2016'
    labeled_ratio = 0.5
    batch_size = 4
    mim_batch_size = 16
    image_size = 256
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\nWarning: Data directory '{data_dir}' not found!")
        print("Please ensure you have the ISIC-2016 dataset downloaded.")
        print("\nExpected directory structure:")
        print("  data/ISIC2016/")
        print("    ├── images/          (contains *.jpg files)")
        print("    └── masks/           (contains *_segmentation.png files)")
        return
    
    try:
        # Test Stage 2 dataloaders
        print("\n[1] Testing Stage 2 dataloaders...")
        labeled_loader, unlabeled_weak, unlabeled_strong, test_loader = create_dataloaders(
            data_dir=data_dir,
            labeled_ratio=labeled_ratio,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=2
        )
        
        # Test labeled loader
        images, masks = next(iter(labeled_loader))
        print(f"\nLabeled batch:")
        print(f"  Images: {images.shape} (dtype: {images.dtype})")
        print(f"  Masks: {masks.shape} (dtype: {masks.dtype})")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
        
        # Test unlabeled weak loader
        images_weak = next(iter(unlabeled_weak))
        print(f"\nUnlabeled (weak) batch:")
        print(f"  Images: {images_weak.shape}")
        
        # Test unlabeled strong loader
        images_strong = next(iter(unlabeled_strong))
        print(f"\nUnlabeled (strong) batch:")
        print(f"  Images: {images_strong.shape}")
        
        # Test MIM dataloader
        print("\n[2] Testing Stage 1 (MIM) dataloader...")
        mim_loader = create_mim_dataloader(
            data_dir=data_dir,
            batch_size=mim_batch_size,
            image_size=image_size,
            num_workers=2
        )
        
        mim_images = next(iter(mim_loader))
        print(f"\nMIM batch:")
        print(f"  Images: {mim_images.shape}")
        print(f"  Image range: [{mim_images.min():.3f}, {mim_images.max():.3f}]")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    """Test the data loaders"""
    test_dataloaders()
