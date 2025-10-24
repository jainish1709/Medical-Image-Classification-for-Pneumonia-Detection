"""
Inference script for Pneumonia Detection
Supports single image and batch inference
"""

import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import time

from config import Config
from model import get_model
from dataset import get_single_image_transform
from utils import load_model

class PneumoniaPredictor:
    """Pneumonia Prediction Class"""
    
    def __init__(self, model_path=None):
        """
        Initialize predictor
        
        Args:
            model_path (str): Path to trained model
        """
        if model_path is None:
            model_path = Config.CHECKPOINT_PATH
        
        # Load model
        self.model = get_model(pretrained=False)
        self.model, _, _, _ = load_model(self.model, None, model_path)
        self.model.eval()
        
        # Get transform
        self.transform = get_single_image_transform()
        
        self.device = Config.DEVICE
        
        print("Predictor initialized successfully!")
    
    def predict_single(self, image_path, return_probs=True):
        """
        Predict pneumonia for a single image
        
        Args:
            image_path (str): Path to image
            return_probs (bool): Return probability scores
        
        Returns:
            prediction (str): Class name
            confidence (float): Prediction confidence
            probs (dict): Probability for each class (if return_probs=True)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        # Get results
        predicted_class = Config.CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item() * 100
        
        if return_probs:
            prob_dict = {
                Config.CLASS_NAMES[i]: probs[0][i].item() * 100 
                for i in range(len(Config.CLASS_NAMES))
            }
            return predicted_class, confidence_score, prob_dict
        
        return predicted_class, confidence_score
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict pneumonia for multiple images
        
        Args:
            image_paths (list): List of image paths
            batch_size (int): Batch size for processing
        
        Returns:
            results (list): List of prediction dictionaries
        """
        results = []
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            # Load and preprocess batch
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_tensors.append(image_tensor)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack tensors
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probs, 1)
            
            # Store results
            for j, img_path in enumerate(batch_paths):
                if j < len(predictions):
                    result = {
                        'image_path': img_path,
                        'prediction': Config.CLASS_NAMES[predictions[j].item()],
                        'confidence': confidences[j].item() * 100,
                        'probabilities': {
                            Config.CLASS_NAMES[k]: probs[j][k].item() * 100
                            for k in range(len(Config.CLASS_NAMES))
                        }
                    }
                    results.append(result)
        
        return results
    
    def predict_from_folder(self, folder_path, output_csv=None):
        """
        Predict for all images in a folder
        
        Args:
            folder_path (str): Path to folder containing images
            output_csv (str): Path to save results CSV (optional)
        
        Returns:
            results (list): List of predictions
        """
        # Get all image paths
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_paths = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_paths:
            print(f"No images found in {folder_path}")
            return []
        
        # Predict
        results = self.predict_batch(image_paths)
        
        # Save to CSV if requested
        if output_csv:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")
        
        return results

def predict_single_image(image_path, model_path=None):
    """
    Convenience function to predict a single image
    
    Args:
        image_path (str): Path to image
        model_path (str): Path to model checkpoint
    
    Returns:
        prediction, confidence, probabilities
    """
    predictor = PneumoniaPredictor(model_path)
    prediction, confidence, probs = predictor.predict_single(image_path)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nProbabilities:")
    for class_name, prob in probs.items():
        print(f"  {class_name}: {prob:.2f}%")
    print("="*60 + "\n")
    
    return prediction, confidence, probs

def benchmark_inference_speed(predictor, test_image_path, num_iterations=100):
    """
    Benchmark inference speed
    
    Args:
        predictor: PneumoniaPredictor instance
        test_image_path (str): Path to test image
        num_iterations (int): Number of iterations
    """
    print(f"\nBenchmarking inference speed ({num_iterations} iterations)...")
    
    # Warm-up
    for _ in range(10):
        predictor.predict_single(test_image_path, return_probs=False)
    
    # Benchmark
    start_time = time.time()
    for _ in tqdm(range(num_iterations)):
        predictor.predict_single(test_image_path, return_probs=False)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = num_iterations / total_time
    
    print("\n" + "="*60)
    print("INFERENCE SPEED BENCHMARK")
    print("="*60)
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Image: {avg_time*1000:.2f} ms")
    print(f"Throughput: {fps:.2f} images/second")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Example usage
    
    # Single image prediction
    test_image = "path/to/test/image.jpg"  # Replace with actual path
    
    if os.path.exists(test_image):
        predict_single_image(test_image)
    else:
        print("Please provide a valid test image path")
    
    # Batch prediction example
    # predictor = PneumoniaPredictor()
    # results = predictor.predict_from_folder("path/to/folder", "results.csv")