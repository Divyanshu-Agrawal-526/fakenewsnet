#!/usr/bin/env python3
"""
Training script for FN-BERT-TFIDF model using the datasets
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import torch
from datetime import datetime

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from fn_bert_tfidf import FNBertTfidfModel
from fn_bert_tfidf_trainer import FNBertTfidfTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_fn_bert_tfidf.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_fakenewsnet_data(data_dir: str = "data"):
    """
    Load FakeNewsNet dataset
    """
    logger.info("Loading FakeNewsNet dataset...")
    
    try:
        # Load fake news data
        gossipcop_fake = pd.read_csv(os.path.join(data_dir, "fakenewsnet_dataset/gossipcop_fake.csv"))
        politifact_fake = pd.read_csv(os.path.join(data_dir, "fakenewsnet_dataset/politifact_fake.csv"))
        
        # Load real news data
        gossipcop_real = pd.read_csv(os.path.join(data_dir, "fakenewsnet_dataset/gossipcop_real.csv"))
        politifact_real = pd.read_csv(os.path.join(data_dir, "fakenewsnet_dataset/politifact_real.csv"))
        
        # Add labels
        gossipcop_fake['label'] = 1  # Fake
        politifact_fake['label'] = 1  # Fake
        gossipcop_real['label'] = 0   # Real
        politifact_real['label'] = 0  # Real
        
        # Combine datasets
        fake_data = pd.concat([gossipcop_fake, politifact_fake], ignore_index=True)
        real_data = pd.concat([gossipcop_real, politifact_real], ignore_index=True)
        
        # Combine all data
        all_data = pd.concat([fake_data, real_data], ignore_index=True)
        
        # Clean data
        all_data = all_data.dropna(subset=['title'])
        all_data = all_data[all_data['title'].str.len() > 10]  # Remove very short titles
        
        logger.info(f"FakeNewsNet dataset loaded: {len(all_data)} samples")
        logger.info(f"Fake news: {len(all_data[all_data['label'] == 1])}")
        logger.info(f"Real news: {len(all_data[all_data['label'] == 0])}")
        
        return all_data
        
    except Exception as e:
        logger.error(f"Error loading FakeNewsNet dataset: {e}")
        return None

def prepare_training_data(data):
    """
    Prepare data for training
    """
    logger.info("Preparing training data...")
    
    # Extract texts and labels
    texts = data['title'].fillna('').astype(str).tolist()
    labels = data['label'].astype(int).tolist()
    
    # Remove empty texts
    valid_indices = [i for i, text in enumerate(texts) if text.strip()]
    texts = [texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    logger.info(f"Prepared {len(texts)} texts for training")
    
    return texts, labels

def main():
    """
    Main training function
    """
    logger.info("Starting FN-BERT-TFIDF training...")
    
    # Check if CUDA is available
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data
    data = load_fakenewsnet_data()
    if data is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Prepare training data
    texts, labels = prepare_training_data(data)
    
    # Create output directory
    output_dir = Path("models/saved_models/fn_bert_tfidf")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    logger.info("Creating FN-BERT-TFIDF model...")
    model = FNBertTfidfModel(
        bert_model_name="bert-base-uncased",  # Use base model for faster training
        tfidf_dim=5000,
        bert_dim=768,  # Base BERT has 768 dimensions
        lstm_hidden=64,
        cnn_filters=16,
        cnn_kernel=128,
        num_classes=2,
        dropout=0.2
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = FNBertTfidfTrainer(
        model=model,
        device=device,
        learning_rate=2e-5,
        weight_decay=0.01,
        batch_size=8  # Smaller batch size for memory constraints
    )
    
    # Train model
    logger.info("Starting training...")
    try:
        history, test_metrics = trainer.train(
            texts=texts,
            labels=labels,
            num_epochs=5,  # Start with fewer epochs
            save_path=str(output_dir)
        )
        
        # Plot training history
        plot_path = output_dir / "training_history.png"
        trainer.plot_training_history(history, save_path=str(plot_path))
        
        # Save training results
        results = {
            'model': 'FN-BERT-TFIDF',
            'dataset': 'FakeNewsNet',
            'training_date': datetime.now().isoformat(),
            'device': device,
            'final_test_accuracy': test_metrics['test_accuracy'],
            'final_test_precision': test_metrics['test_precision'],
            'final_test_recall': test_metrics['test_recall'],
            'final_test_f1': test_metrics['test_f1'],
            'training_history': history
        }
        
        import json
        with open(output_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final Test Accuracy: {test_metrics['test_accuracy']:.2f}%")
        logger.info(f"Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
