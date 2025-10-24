"""
Evaluation script for Pneumonia Detection Model
Calculates comprehensive metrics and visualizations
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
import os

from config import Config
from dataset import get_data_loaders
from model import get_model
from utils import (
    load_model, plot_confusion_matrix, plot_roc_curve,
    save_classification_report, save_metrics_to_json,
    visualize_predictions
)

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset
    
    Args:
        model: Neural network model
        dataloader: Data loader
        device: Device to evaluate on
    
    Returns:
        y_true, y_pred, y_scores: True labels, predictions, and probability scores
    """
    model.eval()
    
    y_true = []
    y_pred = []
    y_scores = []
    
    print("\nEvaluating model...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probs[:, 1].cpu().numpy())  # Probability of pneumonia class
    
    return np.array(y_true), np.array(y_pred), np.array(y_scores)

def calculate_metrics(y_true, y_pred, y_scores):
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores
    
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
        'precision_per_class': precision_score(y_true, y_pred, average=None).tolist(),
        'recall_per_class': recall_score(y_true, y_pred, average=None).tolist(),
        'f1_score_per_class': f1_score(y_true, y_pred, average=None).tolist()
    }
    
    return metrics

def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
    print("="*60)
    
    print("\nPer-Class Metrics:")
    print("-"*60)
    for i, class_name in enumerate(Config.CLASS_NAMES):
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision_per_class'][i]*100:.2f}%")
        print(f"  Recall:    {metrics['recall_per_class'][i]*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_score_per_class'][i]*100:.2f}%")
    print("="*60 + "\n")

def run_evaluation(model_path=None):
    """
    Run complete evaluation pipeline
    
    Args:
        model_path (str): Path to model checkpoint
    """
    # Use default path if not provided
    if model_path is None:
        model_path = Config.CHECKPOINT_PATH
    
    print("\n" + "="*60)
    print("PNEUMONIA DETECTION MODEL EVALUATION")
    print("="*60)
    
    # Load data
    _, _, test_loader = get_data_loaders()
    
    # Load model
    model = get_model(pretrained=False)
    model, _, _, _ = load_model(model, None, model_path)
    model.eval()
    
    # Evaluate
    y_true, y_pred, y_scores = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics to JSON
    metrics_json_path = os.path.join(Config.METRICS_DIR, 'test_metrics.json')
    save_metrics_to_json(metrics, metrics_json_path)
    
    # Generate confusion matrix
    cm_path = os.path.join(Config.PLOTS_DIR, 'confusion_matrix.png')
    cm = plot_confusion_matrix(y_true, y_pred, Config.CLASS_NAMES, save_path=cm_path)
    
    # Generate ROC curve
    roc_path = os.path.join(Config.PLOTS_DIR, 'roc_curve.png')
    roc_auc = plot_roc_curve(y_true, y_scores, save_path=roc_path)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Save classification report
    report_path = os.path.join(Config.METRICS_DIR, 'classification_report.txt')
    save_classification_report(y_true, y_pred, Config.CLASS_NAMES, report_path)
    
    # Visualize predictions
    pred_path = os.path.join(Config.PLOTS_DIR, 'sample_predictions.png')
    visualize_predictions(model, test_loader, Config.DEVICE, 
                         Config.CLASS_NAMES, num_images=8, save_path=pred_path)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"Results saved to: {Config.RESULTS_DIR}")
    print("="*60 + "\n")
    
    return metrics, y_true, y_pred, y_scores

if __name__ == "__main__":
    # Run evaluation
    metrics, y_true, y_pred, y_scores = run_evaluation()