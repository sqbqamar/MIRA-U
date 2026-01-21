"""
MIRA-U Prediction and Evaluation

Implements model evaluation and inference with comprehensive metrics:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Accuracy (ACC)
- Precision (PRE)
- Recall/Sensitivity (REC/SEN)
- Specificity (SPE)

Supports:
- Single image prediction
- Batch prediction
- Test Time Augmentation (TTA)
- Visualization of results

Aligned with revised manuscript evaluation protocols.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from model import HybridCNNTransformerStudent
from data_loader import create_dataloaders, get_test_augmentation
from config import Config


class SegmentationMetrics:
    """
    Calculate comprehensive segmentation metrics
    
    Metrics computed:
    - DSC (Dice Similarity Coefficient): 2|P∩G| / (|P|+|G|)
    - IoU (Intersection over Union): |P∩G| / |P∪G|
    - ACC (Accuracy): (TP+TN) / (TP+TN+FP+FN)
    - PRE (Precision): TP / (TP+FP)
    - REC/SEN (Recall/Sensitivity): TP / (TP+FN)
    - SPE (Specificity): TN / (TN+FP)
    
    where P = Prediction, G = Ground truth
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metric accumulators"""
        self.dice_scores = []
        self.iou_scores = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.sensitivities = []
        self.specificities = []
    
    def update(self, pred, target, threshold=0.5):
        """
        Update metrics with a batch
        
        Args:
            pred: (B, 1, H, W) or (1, H, W) - prediction probabilities
            target: (B, 1, H, W) or (1, H, W) - ground truth binary mask
            threshold: Threshold for binarizing predictions (default: 0.5)
        """
        # Ensure correct shape and type
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        # Flatten for confusion matrix
        pred_flat = pred_binary.view(-1).cpu().numpy()
        target_flat = target_binary.view(-1).cpu().numpy()
        
        # Compute confusion matrix
        # Note: sklearn returns [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(
            target_flat, pred_flat, labels=[0, 1]
        ).ravel()
        
        # Dice Similarity Coefficient (DSC / F1 Score)
        # DSC = 2TP / (2TP + FP + FN)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        self.dice_scores.append(dice)
        
        # Intersection over Union (IoU / Jaccard Index)
        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-8)
        self.iou_scores.append(iou)
        
        # Accuracy (ACC)
        # ACC = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        self.accuracies.append(accuracy)
        
        # Precision (PRE)
        # PRE = TP / (TP + FP)
        precision = tp / (tp + fp + 1e-8)
        self.precisions.append(precision)
        
        # Recall / Sensitivity (REC / SEN)
        # REC = TP / (TP + FN)
        recall = tp / (tp + fn + 1e-8)
        self.recalls.append(recall)
        self.sensitivities.append(recall)  # Same as recall
        
        # Specificity (SPE)
        # SPE = TN / (TN + FP)
        specificity = tn / (tn + fp + 1e-8)
        self.specificities.append(specificity)
    
    def get_metrics(self):
        """
        Get average metrics across all samples
        
        Returns:
            dict: Dictionary containing all metrics with mean and std
        """
        return {
            'DSC': np.mean(self.dice_scores),
            'IoU': np.mean(self.iou_scores),
            'ACC': np.mean(self.accuracies),
            'PRE': np.mean(self.precisions),
            'REC': np.mean(self.recalls),
            'SEN': np.mean(self.sensitivities),
            'SPE': np.mean(self.specificities),
            'DSC_std': np.std(self.dice_scores),
            'IoU_std': np.std(self.iou_scores),
            'ACC_std': np.std(self.accuracies),
            'PRE_std': np.std(self.precisions),
            'REC_std': np.std(self.recalls),
            'SEN_std': np.std(self.sensitivities),
            'SPE_std': np.std(self.specificities),
        }


class MIRAUPredictor:
    """
    MIRA-U Prediction and Evaluation Interface
    
    Provides methods for:
    - Loading trained models
    - Single image prediction
    - Batch prediction
    - Test set evaluation with comprehensive metrics
    - Test Time Augmentation (TTA)
    - Visualization of results
    """
    
    def __init__(self, checkpoint_path, config):
        """
        Initialize predictor
        
        Args:
            checkpoint_path: Path to saved model checkpoint
            config: Configuration dict or Config object
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if isinstance(config, dict) else config.__dict__
        
        print(f"Initializing MIRA-U Predictor")
        print(f"Device: {self.device}")
        
        # Initialize student model (for inference)
        self.model = HybridCNNTransformerStudent(
            in_channels=3,
            out_channels=1,
            base_channels=self.config.get('base_channels', 32)
        ).to(self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Model info
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Model parameters: {params:.2f}M")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file (.pth)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load student model state dict
        if 'student_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['student_state_dict'])
            print(f"✓ Checkpoint loaded successfully!")
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch'] + 1}")
            if 'stage' in checkpoint:
                print(f"  Stage: {checkpoint['stage']}")
        else:
            raise KeyError("Checkpoint does not contain 'student_state_dict'")
    
    def predict(self, image):
        """
        Predict segmentation mask for a single image tensor
        
        Args:
            image: (3, H, W) or (B, 3, H, W) - preprocessed image tensor
        
        Returns:
            pred: (1, H, W) or (B, 1, H, W) - prediction probability map
        """
        self.model.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            pred = self.model(image)
        return pred
    
    def predict_with_tta(self, image):
        """
        Predict with Test Time Augmentation (TTA)
        
        Performs predictions with:
        1. Original image
        2. Horizontal flip
        3. Vertical flip
        4. Both flips
        
        Final prediction is the average of all augmentations.
        
        Args:
            image: (3, H, W) or (B, 3, H, W) - preprocessed image tensor
        
        Returns:
            pred: (1, H, W) or (B, 1, H, W) - averaged prediction
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            
            # Original prediction
            pred = self.model(image)
            predictions.append(pred)
            
            # Horizontal flip (flip width dimension)
            pred_hflip = self.model(torch.flip(image, dims=[3]))
            predictions.append(torch.flip(pred_hflip, dims=[3]))
            
            # Vertical flip (flip height dimension)
            pred_vflip = self.model(torch.flip(image, dims=[2]))
            predictions.append(torch.flip(pred_vflip, dims=[2]))
            
            # Both flips
            pred_hvflip = self.model(torch.flip(image, dims=[2, 3]))
            predictions.append(torch.flip(pred_hvflip, dims=[2, 3]))
        
        # Average all predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred
    
    def evaluate(self, test_loader, use_tta=False, save_results=False, output_dir='./results'):
        """
        Evaluate model on test set with comprehensive metrics
        
        Computes and reports:
        - DSC (Dice Similarity Coefficient)
        - IoU (Intersection over Union)
        - ACC (Accuracy)
        - PRE (Precision)
        - REC (Recall)
        - SEN (Sensitivity)
        - SPE (Specificity)
        
        Args:
            test_loader: DataLoader for test set
            use_tta: Whether to use Test Time Augmentation
            save_results: Whether to save predictions and metrics
            output_dir: Directory to save results
        
        Returns:
            dict: Dictionary containing all metrics
        """
        print("\n" + "="*70)
        print("EVALUATING MODEL ON TEST SET")
        print("="*70)
        print(f"Test Time Augmentation (TTA): {'Enabled' if use_tta else 'Disabled'}")
        print("="*70)
        
        self.model.eval()
        metrics = SegmentationMetrics()
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            pred_dir = os.path.join(output_dir, 'predictions')
            os.makedirs(pred_dir, exist_ok=True)
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating")
            sample_count = 0
            
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Ensure masks have channel dimension
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                
                # Predict
                if use_tta:
                    # TTA for each image individually
                    for i in range(images.shape[0]):
                        pred = self.predict_with_tta(images[i])
                        metrics.update(pred, masks[i])
                        
                        # Save sample predictions
                        if save_results and sample_count < 8:
                            save_path = os.path.join(pred_dir, f'sample_{sample_count:03d}.png')
                            self.save_prediction(images[i], masks[i], pred, save_path)
                            sample_count += 1
                else:
                    # Standard prediction
                    preds = self.model(images)
                    
                    for i in range(images.shape[0]):
                        metrics.update(preds[i:i+1], masks[i:i+1])
                        
                        # Save sample predictions
                        if save_results and sample_count < 8:
                            save_path = os.path.join(pred_dir, f'sample_{sample_count:03d}.png')
                            self.save_prediction(images[i], masks[i], preds[i], save_path)
                            sample_count += 1
                
                # Update progress bar with current metrics
                current_metrics = metrics.get_metrics()
                pbar.set_postfix({
                    'DSC': f"{current_metrics['DSC']:.4f}",
                    'IoU': f"{current_metrics['IoU']:.4f}",
                    'ACC': f"{current_metrics['ACC']:.4f}"
                })
        
        # Get final metrics
        final_metrics = metrics.get_metrics()
        
        # Print results table (matching manuscript format)
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"{'Metric':<20} {'Value':<15} {'Std Dev':<15}")
        print("-"*70)
        print(f"{'DSC':<20} {final_metrics['DSC']:.4f}{'':<10} {final_metrics['DSC_std']:.4f}")
        print(f"{'IoU':<20} {final_metrics['IoU']:.4f}{'':<10} {final_metrics['IoU_std']:.4f}")
        print(f"{'ACC':<20} {final_metrics['ACC']:.4f}{'':<10} {final_metrics['ACC_std']:.4f}")
        print(f"{'PRE':<20} {final_metrics['PRE']:.4f}{'':<10} {final_metrics['PRE_std']:.4f}")
        print(f"{'REC':<20} {final_metrics['REC']:.4f}{'':<10} {final_metrics['REC_std']:.4f}")
        print(f"{'SEN':<20} {final_metrics['SEN']:.4f}{'':<10} {final_metrics['SEN_std']:.4f}")
        print(f"{'SPE':<20} {final_metrics['SPE']:.4f}{'':<10} {final_metrics['SPE_std']:.4f}")
        print("="*70)
        
        # Save metrics to file
        if save_results:
            metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write("MIRA-U EVALUATION RESULTS\n")
                f.write("="*70 + "\n\n")
                f.write(f"{'Metric':<20} {'Value':<15} {'Std Dev':<15}\n")
                f.write("-"*70 + "\n")
                f.write(f"{'DSC':<20} {final_metrics['DSC']:.4f}{'':<10} {final_metrics['DSC_std']:.4f}\n")
                f.write(f"{'IoU':<20} {final_metrics['IoU']:.4f}{'':<10} {final_metrics['IoU_std']:.4f}\n")
                f.write(f"{'ACC':<20} {final_metrics['ACC']:.4f}{'':<10} {final_metrics['ACC_std']:.4f}\n")
                f.write(f"{'PRE':<20} {final_metrics['PRE']:.4f}{'':<10} {final_metrics['PRE_std']:.4f}\n")
                f.write(f"{'REC':<20} {final_metrics['REC']:.4f}{'':<10} {final_metrics['REC_std']:.4f}\n")
                f.write(f"{'SEN':<20} {final_metrics['SEN']:.4f}{'':<10} {final_metrics['SEN_std']:.4f}\n")
                f.write(f"{'SPE':<20} {final_metrics['SPE']:.4f}{'':<10} {final_metrics['SPE_std']:.4f}\n")
                f.write("="*70 + "\n\n")
                
                # Additional info
                f.write("Configuration:\n")
                f.write(f"  Test Time Augmentation: {use_tta}\n")
                f.write(f"  Number of test samples: {len(metrics.dice_scores)}\n")
            
            print(f"\n✓ Metrics saved to: {metrics_path}")
            print(f"✓ Sample predictions saved to: {pred_dir}")
        
        return final_metrics
    
    def save_prediction(self, image, ground_truth, prediction, save_path):
        """
        Save visualization comparing input, ground truth, and prediction
        
        Args:
            image: (3, H, W) - input image tensor
            ground_truth: (1, H, W) - ground truth mask
            prediction: (1, H, W) - prediction probability map
            save_path: Path to save the visualization
        """
        # Denormalize image (reverse ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Convert to numpy
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        gt_np = ground_truth.cpu().numpy().squeeze()
        pred_np = prediction.cpu().numpy().squeeze()
        pred_binary = (pred_np > 0.5).astype(np.float32)
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Input image
        axes[0].imshow(image_np)
        axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Prediction (soft)
        axes[2].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Prediction (Soft)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Prediction (binary)
        axes[3].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title('Prediction (Binary)', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def predict_single_image(self, image_path, save_path=None, visualize=False):
        """
        Predict segmentation mask for a single image file
        
        Args:
            image_path: Path to input image
            save_path: Path to save prediction (optional)
            visualize: Whether to save visualization (default: False)
        
        Returns:
            pred_np: Prediction probability map (H, W)
        """
        # Load and preprocess image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Apply test-time preprocessing
        transform = get_test_augmentation(self.config.get('image_size', 256))
        transformed = transform(image=image)
        image_tensor = transformed['image']
        
        # Predict
        pred = self.predict(image_tensor)
        pred_np = pred.cpu().numpy().squeeze()
        
        # Resize to original size
        pred_np = cv2.resize(pred_np, (original_size[1], original_size[0]))
        
        # Save if requested
        if save_path:
            pred_binary = (pred_np > 0.5).astype(np.uint8) * 255
            cv2.imwrite(save_path, pred_binary)
            print(f"✓ Prediction saved to: {save_path}")
        
        # Visualize if requested
        if visualize and save_path:
            vis_path = save_path.replace('.png', '_vis.png')
            self._visualize_single(image, pred_np, vis_path)
        
        return pred_np
    
    def _visualize_single(self, image, prediction, save_path):
        """Helper to visualize single image prediction"""
        pred_binary = (prediction > 0.5).astype(np.float32)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title('Prediction (Soft)')
        axes[1].axis('off')
        
        axes[2].imshow(pred_binary, cmap='gray')
        axes[2].set_title('Prediction (Binary)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def batch_predict(self, image_dir, output_dir, use_tta=False):
        """
        Predict segmentation for all images in a directory
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save predictions
            use_tta: Whether to use Test Time Augmentation
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if len(image_files) == 0:
            print(f"No images found in {image_dir}")
            return
        
        print(f"\nProcessing {len(image_files)} images...")
        print(f"Input directory: {image_dir}")
        print(f"Output directory: {output_dir}")
        print(f"TTA: {'Enabled' if use_tta else 'Disabled'}")
        
        for img_file in tqdm(image_files, desc="Predicting"):
            img_path = os.path.join(image_dir, img_file)
            save_path = os.path.join(output_dir, 
                                    img_file.replace('.jpg', '_pred.png')
                                           .replace('.jpeg', '_pred.png'))
            
            try:
                if use_tta:
                    # Load and predict with TTA
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    transform = get_test_augmentation(self.config.get('image_size', 256))
                    transformed = transform(image=image)
                    image_tensor = transformed['image']
                    pred = self.predict_with_tta(image_tensor)
                    pred_np = pred.cpu().numpy().squeeze()
                    pred_binary = (pred_np > 0.5).astype(np.uint8) * 255
                    cv2.imwrite(save_path, pred_binary)
                else:
                    self.predict_single_image(img_path, save_path)
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
        
        print(f"\n✓ Batch prediction completed!")
        print(f"✓ Results saved to: {output_dir}")


def main():
    """
    Main prediction function
    
    Evaluates trained MIRA-U model on test set and saves results.
    """
    
    # Load configuration
    try:
        from config import Config
        config = Config()
        print(config)
    except ImportError:
        print("Warning: config.py not found, using default configuration")
        config = {
            'data_dir': './data/ISIC2016',
            'labeled_ratio': 0.5,
            'batch_size': 4,
            'image_size': 256,
            'num_workers': 4,
            'base_channels': 32,
        }
    
    # Prediction settings
    checkpoint_path = './checkpoints/student_best_dice.pth'
    output_dir = './results'
    use_tta = False  # Set to True for Test Time Augmentation
    
    print("\n" + "="*70)
    print("MIRA-U MODEL EVALUATION")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or specify correct checkpoint path.")
        print("\nAvailable checkpoints:")
        ckpt_dir = './checkpoints'
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
            for ckpt in ckpts:
                print(f"  - {ckpt}")
        return
    
    # Create test data loader
    print("\nCreating test data loader...")
    try:
        _, _, _, test_loader = create_dataloaders(
            data_dir=config.data_dir if hasattr(config, 'data_dir') else config['data_dir'],
            labeled_ratio=config.labeled_ratio if hasattr(config, 'labeled_ratio') else config['labeled_ratio'],
            batch_size=config.batch_size if hasattr(config, 'batch_size') else config['batch_size'],
            image_size=config.image_size if hasattr(config, 'image_size') else config['image_size'],
            num_workers=config.num_workers if hasattr(config, 'num_workers') else config['num_workers']
        )
        print(f"✓ Test loader created with {len(test_loader)} batches")
    except Exception as e:
        print(f"Error creating data loader: {str(e)}")
        return
    
    # Initialize predictor
    try:
        predictor = MIRAUPredictor(checkpoint_path, config)
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")
        return
    
    # Evaluate on test set
    try:
        metrics = predictor.evaluate(
            test_loader,
            use_tta=use_tta,
            save_results=True,
            output_dir=output_dir
        )
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Results saved to: {output_dir}")
        print(f"  - Evaluation metrics: {output_dir}/evaluation_metrics.txt")
        print(f"  - Sample predictions: {output_dir}/predictions/")
        print("="*70)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
