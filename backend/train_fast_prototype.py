#!/usr/bin/env python3
"""
Fast Prototype Training - Get results in minutes, not hours!
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastFakeNewsDetector:
    """
    Fast fake news detector using DistilBERT + TF-IDF + Logistic Regression
    """
    
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Use DistilBERT (66M params instead of 110M)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert.to(self.device)
        
        # TF-IDF for fast feature extraction
        self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        
        # Simple classifier
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        
        logger.info("Fast model initialized!")
    
    def get_bert_features(self, texts, max_length=128):  # Shorter sequences
        """Extract BERT features quickly"""
        self.bert.eval()
        features = []
        
        with torch.no_grad():
            for text in texts[:1000]:  # Limit to first 1000 for speed
                inputs = self.tokenizer(
                    text, 
                    max_length=max_length, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.bert(**inputs)
                # Use mean pooling instead of [CLS]
                feature = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                features.append(feature.flatten())
        
        return np.array(features)
    
    def get_tfidf_features(self, texts):
        """Extract TF-IDF features"""
        return self.tfidf.fit_transform(texts[:1000]).toarray()
    
    def train(self, texts, labels):
        """Train the model quickly"""
        logger.info("Starting fast training...")
        
        # Use subset for speed
        sample_size = min(1000, len(texts))
        texts = texts[:sample_size]
        labels = labels[:sample_size]
        
        logger.info(f"Training on {sample_size} samples...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Get features
        logger.info("Extracting BERT features...")
        bert_features = self.get_bert_features(X_train)
        
        logger.info("Extracting TF-IDF features...")
        tfidf_features = self.tfidf.fit_transform(X_train).toarray()
        
        # Combine features
        combined_features = np.hstack([bert_features, tfidf_features])
        
        # Train classifier
        logger.info("Training classifier...")
        self.classifier.fit(combined_features, y_train)
        
        # Test
        logger.info("Testing model...")
        bert_test = self.get_bert_features(X_test)
        tfidf_test = self.tfidf.transform(X_test).toarray()
        combined_test = np.hstack([bert_test, tfidf_test])
        
        predictions = self.classifier.predict(combined_test)
        accuracy = accuracy_score(y_test, predictions)
        
        logger.info(f"Fast training completed! Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, predictions, target_names=['Real', 'Fake']))
        
        return accuracy, predictions, y_test

def load_data():
    """Load dataset quickly"""
    logger.info("Loading FakeNewsNet dataset...")
    
    try:
        # Load only first 1000 samples for speed
        gossipcop_fake = pd.read_csv("data/fakenewsnet_dataset/gossipcop_fake.csv").head(500)
        politifact_fake = pd.read_csv("data/fakenewsnet_dataset/politifact_fake.csv").head(500)
        gossipcop_real = pd.read_csv("data/fakenewsnet_dataset/gossipcop_real.csv").head(500)
        politifact_real = pd.read_csv("data/fakenewsnet_dataset/politifact_real.csv").head(500)
        
        # Add labels
        gossipcop_fake['label'] = 1
        politifact_fake['label'] = 1
        gossipcop_real['label'] = 0
        politifact_real['label'] = 0
        
        # Combine
        all_data = pd.concat([gossipcop_fake, politifact_fake, gossipcop_real, politifact_real], ignore_index=True)
        all_data = all_data.dropna(subset=['title'])
        all_data = all_data[all_data['title'].str.len() > 10]
        
        logger.info(f"Loaded {len(all_data)} samples")
        return all_data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def main():
    """Main function"""
    logger.info("🚀 Starting FAST prototype training...")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Prepare data
    texts = data['title'].fillna('').astype(str).tolist()
    labels = data['label'].astype(int).tolist()
    
    # Create and train model
    model = FastFakeNewsDetector()
    accuracy, predictions, true_labels = model.train(texts, labels)
    
    # Save results
    output_dir = Path("models/saved_models/fast_prototype")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model': 'FastFakeNewsDetector',
        'accuracy': accuracy,
        'training_date': datetime.now().isoformat(),
        'device': model.device,
        'sample_size': len(texts)
    }
    
    import json
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Fast training completed! Results saved to {output_dir}")
    logger.info(f"📊 Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
