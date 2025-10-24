"""
Configuration file for Pneumonia Detection Project
Contains all hyperparameters and paths
"""

import torch
import os

class Config:
    """Configuration class for the project"""
    
    # ========== Paths ==========
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'chest_xray')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
    METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
    
    # ========== Data Paths ==========
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    
    # ========== Model Configuration ==========
    MODEL_NAME = 'efficientnet_b0'
    NUM_CLASSES = 2  # Normal and Pneumonia
    PRETRAINED = True
    
    # ========== Training Hyperparameters ==========
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # ========== Image Parameters ==========
    IMAGE_SIZE = 224  # EfficientNet-B0 default input size
    CHANNELS = 3
    
    # ImageNet normalization (standard for transfer learning)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # ========== Data Augmentation ==========
    ROTATION_RANGE = 10
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    
    # ========== Training Configuration ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2  # For data loading
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # ========== Early Stopping ==========
    PATIENCE = 5  # Stop if no improvement for 5 epochs
    MIN_DELTA = 0.001  # Minimum change to qualify as improvement
    
    # ========== Checkpoint ==========
    SAVE_BEST_ONLY = True
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
    
    # ========== Class Names ==========
    CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
    
    # ========== Random Seed ==========
    SEED = 42
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("=" * 50)
        print("Configuration Parameters")
        print("=" * 50)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")
        print("=" * 50)

# Create directories if they don't exist
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.PLOTS_DIR, exist_ok=True)
os.makedirs(Config.METRICS_DIR, exist_ok=True)