"""
Model architecture for Pneumonia Detection
Uses EfficientNet-B0 with transfer learning
"""

import torch
import torch.nn as nn
import timm
from config import Config

class PneumoniaClassifier(nn.Module):
    """
    Pneumonia Classifier using EfficientNet-B0
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Use pretrained weights
        """
        super(PneumoniaClassifier, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.model = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Get the number of input features for the classifier
        in_features = self.model.classifier.in_features
        
        # Replace the classifier with custom layers
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)

def get_model(pretrained=True, num_classes=None):
    """
    Create and return the model
    
    Args:
        pretrained (bool): Use pretrained weights
        num_classes (int): Number of classes
    
    Returns:
        model: PneumoniaClassifier model
    """
    if num_classes is None:
        num_classes = Config.NUM_CLASSES
    
    model = PneumoniaClassifier(
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    model = model.to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print("Model Information")
    print("="*50)
    print(f"Model: EfficientNet-B0")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {Config.DEVICE}")
    print("="*50 + "\n")
    
    return model

def freeze_backbone(model):
    """
    Freeze the backbone of the model (for fine-tuning)
    Only train the classifier layers
    """
    # Freeze all parameters
    for param in model.model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    for param in model.model.classifier.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone frozen. Trainable parameters: {trainable_params:,}")

def unfreeze_backbone(model):
    """
    Unfreeze the backbone for full fine-tuning
    """
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone unfrozen. Trainable parameters: {trainable_params:,}")

# Test the model
if __name__ == "__main__":
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(Config.DEVICE)
    
    # Create model
    model = get_model(pretrained=True)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")