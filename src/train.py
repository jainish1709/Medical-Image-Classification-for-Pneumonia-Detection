"""
Training script for Pneumonia Detection Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os

from config import Config
from dataset import get_data_loaders
from model import get_model
from utils import (
    set_seed, save_model, plot_training_history,
    save_metrics_to_json, EarlyStopping
)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc='Training', leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def train_model(num_epochs=None, batch_size=None, learning_rate=None):
    """
    Main training function
    
    Args:
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
    """
    # Set defaults
    if num_epochs is None:
        num_epochs = Config.NUM_EPOCHS
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = Config.LEARNING_RATE
    
    # Set random seed
    set_seed(Config.SEED)
    
    # Print configuration
    Config.print_config()
    
    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(batch_size)
    
    # Get model
    model = get_model(pretrained=Config.PRETRAINED)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model, optimizer, epoch+1, best_val_acc,
                Config.CHECKPOINT_PATH
            )
            print(f"✓ New best model saved! (Val Acc: {best_val_acc:.2f}%)")
        
        print()
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print("="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*50 + "\n")
    
    # Plot training history
    plot_path = os.path.join(Config.PLOTS_DIR, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Save training history
    metrics_path = os.path.join(Config.METRICS_DIR, 'training_history.json')
    save_metrics_to_json(history, metrics_path)
    
    return model, history

if __name__ == "__main__":
    # Train the model
    model, history = train_model()