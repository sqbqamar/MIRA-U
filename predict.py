import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from model import HybridCNNTransformerStudent
from data_loader import create_dataloaders


class SegmentationMetrics:
    """Calculate segmentation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
    
    def update(self, pred, target, threshold=0.5):
        """Update metrics with a batch"""
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        # Flatten
        pred_flat = pred_binary.view(-1).cpu().numpy()
        target_flat = target_binary.view(-1).cpu().numpy()
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat, labels=[0, 1]).ravel()
        
        # Dice Score (F1 Score)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        self.dice_scores.append(dice)
        
        # IoU (Jaccard Index)
        iou = tp / (tp + fp + fn + 1e-8)
        self.iou_scores.append(iou)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        self.accuracies.append(accuracy)
        
        # Precision
        precision = tp / (tp + fp + 1e-8)
        self.precisions.append(precision)
        
        # Recall (Sensitivity)
        recall = tp / (tp + fn + 1e-8)
        self.recalls.append(recall)
    
    def get_metrics(self):
        """Get average metrics"""
        return {
            'DSC': np.mean(self.dice_scores),
            'IoU': np.mean(self.iou_scores),
            'Accuracy': np.mean(self.accuracies),
            'Precision': np.mean(self.precisions),
            'Recall': np.mean(self.recalls),
            'DSC_std': np.std(self.dice_scores),
            'IoU_std': np.std(self.iou_scores)
        }


class MIRAUPredictor:
    """MIRA-U Prediction and Evaluation"""
    
    def __init__(self, checkpoint_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize model
        self.model = HybridCNNTransformerStudent(
            in_channels=3,
            out_channels=1,
            base_channels=config['base_channels']
        ).to(self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['student_state_dict'])
        print("Checkpoint loaded successfully!")
    
    def predict(self, image):
        """Predict segmentation mask for a single image"""
        self.model.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            pred = self.model(image)
        return pred
    
    def predict_with_tta(self, image, tta_transforms=None):
        """Predict with Test Time Augmentation"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            
            # Original prediction
            pred = self.model(image)
            predictions.append(pred)
            
            # Horizontal flip
            pred_hflip = self.model(torch.flip(image, dims=[3]))
            predictions.append(torch.flip(pred_hflip, dims=[3]))
            
            # Vertical flip
            pred_vflip = self.model(torch.flip(image, dims=[2]))
            predictions.append(torch.flip(pred_vflip, dims=[2]))
            
            # Both flips
            pred_hvflip = self.model(torch.flip(image, dims=[2, 3]))
            predictions.append(torch.flip(pred_hvflip, dims=[2, 3]))
        
        # Average predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred
    
    def evaluate(self, test_loader, use_tta=False, save_results=False, output_dir='./results'):
        """Evaluate model on test set"""
        print("Evaluating model...")
        
        self.model.eval()
        metrics = SegmentationMetrics()
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating")
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device).unsqueeze(1)
                
                # Predict
                if use_tta:
                    for i in range(images.shape[0]):
                        pred = self.predict_with_tta(images[i])
                        metrics.update(pred, masks[i])
                        
                        if save_results and batch_idx == 0 and i < 8:
                            self.save_prediction(
                                images[i], masks[i], pred,
                                os.path.join(output_dir, 'predictions', f'sample_{i}.png')
                            )
                else:
                    preds = self.model(images)
                    
                    for i in range(images.shape[0]):
                        metrics.update(preds[i], masks[i])
                        
                        if save_results and batch_idx == 0 and i < 8:
                            self.save_prediction(
                                images[i], masks[i], preds[i],
                                os.path.join(output_dir, 'predictions', f'sample_{i}.png')
                            )
                
                # Update progress bar
                current_metrics = metrics.get_metrics()
                pbar.set_postfix({
                    'DSC': f"{current_metrics['DSC']:.4f}",
                    'IoU': f"{current_metrics['IoU']:.4f}"
                })
        
        # Get final metrics
        final_metrics = metrics.get_metrics()
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Dice Similarity Coefficient (DSC): {final_metrics['DSC']:.4f} ± {final_metrics['DSC_std']:.4f}")
        print(f"Intersection over Union (IoU):    {final_metrics['IoU']:.4f} ± {final_metrics['IoU_std']:.4f}")
        print(f"Accuracy:                          {final_metrics['Accuracy']:.4f}")
        print(f"Precision:                         {final_metrics['Precision']:.4f}")
        print(f"Recall:                            {final_metrics['Recall']:.4f}")
        print("="*50)
        
        # Save metrics to file
        if save_results:
            with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
                f.write("EVALUATION RESULTS\n")
                f.write("="*50 + "\n")
                f.write(f"Dice Similarity Coefficient (DSC): {final_metrics['DSC']:.4f} ± {final_metrics['DSC_std']:.4f}\n")
                f.write(f"Intersection over Union (IoU):    {final_metrics['IoU']:.4f} ± {final_metrics['IoU_std']:.4f}\n")
                f.write(f"Accuracy:                          {final_metrics['Accuracy']:.4f}\n")
                f.write(f"Precision:                         {final_metrics['Precision']:.4f}\n")
                f.write(f"Recall:                            {final_metrics['Recall']:.4f}\n")
        
        return final_metrics
    
    def save_prediction(self, image, ground_truth, prediction, save_path):
        """Save visualization of prediction"""
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Convert to numpy
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        gt_np = ground_truth.cpu().numpy().squeeze()
        pred_np = prediction.cpu().numpy().squeeze()
        pred_binary = (pred_np > 0.5).astype(np.float32)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(image_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(gt_np, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_np, cmap='gray')
        axes[2].set_title('Prediction (Soft)')
        axes[2].axis('off')
        
        axes[3].imshow(pred_binary, cmap='gray')
        axes[3].set_title('Prediction (Binary)')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def predict_single_image(self, image_path, save_path=None):
        """Predict segmentation for a single image file"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        from data_loader import get_test_augmentation
        transform = get_test_augmentation(self.config['image_size'])
        transformed = transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)
        
        # Predict
        pred = self.predict(image_tensor)
        pred_np = pred.cpu().numpy().squeeze()
        
        # Save if requested
        if save_path:
            pred_binary = (pred_np > 0.5).astype(np.uint8) * 255
            cv2.imwrite(save_path, pred_binary)
            print(f"Prediction saved to {save_path}")
        
        return pred_np
    
    def batch_predict(self, image_dir, output_dir, use_tta=False):
        """Predict segmentation for all images in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(image_dir, img_file)
            save_path = os.path.join(output_dir, img_file.replace('.jpg', '_pred.png'))
            
            self.predict_single_image(img_path, save_path)


def main():
    """Main prediction function"""
    
    # Configuration
    config = {
        'data_dir': './data/ISIC2016',
        'checkpoint_path': './checkpoints/best_model.pth',
        'batch_size': 4,
        'image_size': 256,
        'num_workers': 4,
        'base_channels': 32,
        'output_dir': './results',
        'use_tta': False  # Set to True for Test Time Augmentation
    }
    
    # Create test data loader
    print("Creating test data loader...")
    _, _, _, test_loader = create_dataloaders(
        data_dir=config['data_dir'],
        labeled_ratio=0.5,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers']
    )
    
    # Initialize predictor
    predictor = MIRAUPredictor(config['checkpoint_path'], config)
    
    # Evaluate on test set
    metrics = predictor.evaluate(
        test_loader,
        use_tta=config['use_tta'],
        save_results=True,
        output_dir=config['output_dir']
    )
    
    print("\nEvaluation completed!")
    print(f"Results saved to {config['output_dir']}")


if __name__ == '__main__':
    main()
