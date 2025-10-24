"""
Utility functions for the Pneumonia Detection Project
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json
import os
from config import Config

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, optimizer, epoch, best_acc, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_accuracy']
    print(f"Model loaded from {path} (Epoch {epoch}, Best Acc: {best_acc:.4f})")
    return model, optimizer, epoch, best_acc

def plot_training_history(history, save_path=None):
    """Plot training and validation loss/accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add accuracy for each class
    for i in range(len(class_names)):
        accuracy = cm[i, i] / cm[i].sum() * 100
        plt.text(i + 0.5, i - 0.3, f'({accuracy:.1f}%)', 
                ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
    
    return cm

def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    plt.show()
    
    return roc_auc

def save_classification_report(y_true, y_pred, class_names, save_path):
    """Save classification report to file"""
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names, 
                                   digits=4)
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"Classification report saved to {save_path}")
    print("\n" + report)

def save_metrics_to_json(metrics, save_path):
    """Save metrics dictionary to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {save_path}")

def visualize_predictions(model, dataloader, device, class_names, num_images=8, save_path=None):
    """Visualize model predictions on sample images"""
    model.eval()
    
    images_shown = 0
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                
                # Denormalize image
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                mean = np.array(Config.MEAN)
                std = np.array(Config.STD)
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # Plot
                ax = axes[images_shown]
                ax.imshow(img)
                
                true_label = class_names[labels[i]]
                pred_label = class_names[predicted[i]]
                confidence = probs[i][predicted[i]].item() * 100
                
                color = 'green' if predicted[i] == labels[i] else 'red'
                ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                           color=color, fontsize=10)
                ax.axis('off')
                
                images_shown += 1
            
            if images_shown >= num_images:
                break
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    plt.show()

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0