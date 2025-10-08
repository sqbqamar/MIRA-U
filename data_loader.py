import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class SkinLesionDataset(Dataset):
    """Dataset class for skin lesion segmentation"""
    
    def __init__(self, image_dir, mask_dir, image_list, transform=None, is_labeled=True):
        """
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            image_list: List of image filenames
            transform: Albumentations transform
            is_labeled: Whether dataset has labels
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
            # Load mask
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
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image


def get_weak_augmentation(image_size=256):
    """Weak augmentation for teacher network"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_strong_augmentation(image_size=256):
    """Strong augmentation for student network"""
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
    """Test time augmentation"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_dataloaders(data_dir, labeled_ratio=0.5, batch_size=4, 
                       image_size=256, num_workers=4):
    """
    Create dataloaders for semi-supervised learning
    
    Args:
        data_dir: Root directory containing 'images' and 'masks' subdirectories
        labeled_ratio: Ratio of labeled data (0.1, 0.25, 0.5)
        batch_size: Batch size
        image_size: Image size for resizing
        num_workers: Number of workers for data loading
    
    Returns:
        labeled_loader, unlabeled_loader, test_loader
    """
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    
    # Get all image filenames
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    # Split into train and test
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    train_size = int(0.8 * len(all_images))
    train_images = all_images[:train_size]
    test_images = all_images[train_size:]
    
    # Split train into labeled and unlabeled
    labeled_size = int(labeled_ratio * len(train_images))
    labeled_images = train_images[:labeled_size]
    unlabeled_images = train_images  # Use all training images as unlabeled
    
    print(f"Total images: {len(all_images)}")
    print(f"Labeled: {len(labeled_images)}, Unlabeled: {len(unlabeled_images)}, Test: {len(test_images)}")
    
    # Create datasets
    labeled_dataset = SkinLesionDataset(
        image_dir, mask_dir, labeled_images,
        transform=get_strong_augmentation(image_size),
        is_labeled=True
    )
    
    unlabeled_dataset_weak = SkinLesionDataset(
        image_dir, mask_dir, unlabeled_images,
        transform=get_weak_augmentation(image_size),
        is_labeled=False
    )
    
    unlabeled_dataset_strong = SkinLesionDataset(
        image_dir, mask_dir, unlabeled_images,
        transform=get_strong_augmentation(image_size),
        is_labeled=False
    )
    
    test_dataset = SkinLesionDataset(
        image_dir, mask_dir, test_images,
        transform=get_test_augmentation(image_size),
        is_labeled=True
    )
    
    # Create dataloaders
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    unlabeled_loader_weak = DataLoader(
        unlabeled_dataset_weak, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    unlabeled_loader_strong = DataLoader(
        unlabeled_dataset_strong, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return labeled_loader, unlabeled_loader_weak, unlabeled_loader_strong, test_loader


class MIMDataset(Dataset):
    """Dataset for Masked Image Modeling pretraining"""
    
    def __init__(self, image_dir, image_list, transform=None, patch_size=8, mask_ratio=0.75):
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
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


def create_mim_dataloader(data_dir, batch_size=8, image_size=256, 
                          patch_size=8, mask_ratio=0.75, num_workers=4):
    """Create dataloader for MIM pretraining"""
    image_dir = os.path.join(data_dir, 'images')
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    dataset = MIMDataset(
        image_dir, all_images, transform=transform,
        patch_size=patch_size, mask_ratio=mask_ratio
    )
    
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    return loader
