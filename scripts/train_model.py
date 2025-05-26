# scripts/train_model.py
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from datetime import datetime

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_processor import DataProcessor
from src.models.recommendation_model import RecommendationEngine
from src.utils.evaluation import evaluate_recommendations

def train_and_evaluate_model():
    """Train and evaluate the recommendation model."""
    print("Starting model training and evaluation...")
    
    # Create data processor
    processor = DataProcessor()
    
    # Load and process data
    print("Loading and processing data...")
    processor.load_data()
    processor.process_data()
    feature_matrices = processor.create_feature_matrices()
    
    # Create train/test split
    print("Creating train/test split...")
    train_test_data = processor.create_train_test_split(test_size=0.2)
    
    # Prepare training data
    train_data = {}
    for key, value in feature_matrices.items():
        train_data[key] = value
        
    # Process train datasets
    for dataset_name, split in train_test_data.items():
        train_data[dataset_name] = split['train']
        
    # Initialize model
    print("Initializing recommendation engine...")
    model = RecommendationEngine()
    
    # Train model
    print("Training model...")
    model.train(train_data)
    
    # Evaluate model
    print("Evaluating model...")
    test_data = {dataset_name: split['test'] for dataset_name, split in train_test_data.items()}
    metrics = evaluate_recommendations(model, test_data)
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    # Save the trained model
    print("\nSaving trained model...")
    save_model(model)
    
    # Save the evaluation metrics
    save_metrics(metrics)
    
    print("Model training and evaluation complete!")
    
def save_model(model):
    """Save the trained model to disk."""
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/recommendation_model_{timestamp}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {model_path}")
    
def save_metrics(metrics):
    """Save the evaluation metrics to disk."""
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"models/metrics_{timestamp}.json"
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_and_evaluate_model()