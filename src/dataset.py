"""
Custom Dataset class for Pneumonia Detection
Handles data loading, preprocessing, and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from config import Config

class PneumoniaDataset(Dataset):
    """Custom Dataset for loading chest X-ray images"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Found {len(self.images)} images in {root_dir}")
        print(f"Classes: {self.classes}")
        print(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get item at index"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms():
    """Get data transformations for train, validation, and test sets"""
    
    # Training data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomRotation(Config.ROTATION_RANGE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=Config.BRIGHTNESS,
            contrast=Config.CONTRAST
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])
    
    # Validation and test data (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])
    
    return train_transform, val_test_transform

def get_data_loaders(batch_size=None):
    """
    Create data loaders for train, validation, and test sets
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    # Get transforms
    train_transform, val_test_transform = get_data_transforms()
    
    # Create datasets
    print("\n" + "="*50)
    print("Loading Datasets")
    print("="*50)
    
    train_dataset = PneumoniaDataset(
        root_dir=Config.TRAIN_DIR,
        transform=train_transform
    )
    
    val_dataset = PneumoniaDataset(
        root_dir=Config.VAL_DIR,
        transform=val_test_transform
    )
    
    test_dataset = PneumoniaDataset(
        root_dir=Config.TEST_DIR,
        transform=val_test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("="*50 + "\n")
    
    return train_loader, val_loader, test_loader

def get_single_image_transform():
    """Get transform for single image inference"""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])