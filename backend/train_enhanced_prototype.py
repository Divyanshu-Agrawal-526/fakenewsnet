#!/usr/bin/env python3
"""
Enhanced Fast Prototype Training - 10K+ samples with balanced fake/real data
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFakeNewsDetector:
    """
    Enhanced fake news detector using DistilBERT + TF-IDF + Logistic Regression
    """
    
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Use DistilBERT (66M params instead of 110M)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert.to(self.device)
        
        # TF-IDF for fast feature extraction
        self.tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 3))
        
        # Simple classifier
        self.classifier = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
        
        logger.info("Enhanced model initialized!")
    
    def get_bert_features(self, texts, max_length=128, batch_size=32):
        """Extract BERT features efficiently with batching"""
        self.bert.eval()
        features = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_features = []
            
            with torch.no_grad():
                for text in batch_texts:
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
                    batch_features.append(feature.flatten())
                
                features.extend(batch_features)
                
                # Clear GPU memory
                if self.device == 'mps':
                    torch.mps.empty_cache()
        
        return np.array(features)
    
    def get_tfidf_features(self, texts):
        """Extract TF-IDF features"""
        return self.tfidf.fit_transform(texts).toarray()
    
    def train(self, texts, labels):
        """Train the model with balanced data"""
        logger.info("Starting enhanced training...")
        
        # Ensure balanced dataset
        fake_indices = [i for i, label in enumerate(labels) if label == 1]
        real_indices = [i for i, label in enumerate(labels) if label == 0]
        
        # Balance the dataset
        min_samples = min(len(fake_indices), len(real_indices))
        balanced_fake = fake_indices[:min_samples]
        balanced_real = real_indices[:min_samples]
        
        balanced_indices = balanced_fake + balanced_real
        balanced_texts = [texts[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        logger.info(f"Balanced dataset: {len(balanced_fake)} fake, {len(balanced_real)} real = {len(balanced_texts)} total")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            balanced_texts, balanced_labels, test_size=0.2, random_state=42, stratify=balanced_labels
        )
        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Get features
        logger.info("Extracting BERT features...")
        bert_features = self.get_bert_features(X_train)
        
        logger.info("Extracting TF-IDF features...")
        tfidf_features = self.tfidf.fit_transform(X_train).toarray()
        
        # Combine features
        combined_features = np.hstack([bert_features, tfidf_features])
        
        logger.info(f"Feature matrix shape: {combined_features.shape}")
        
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
        
        logger.info(f"Enhanced training completed! Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, predictions, target_names=['Real', 'Fake']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        return accuracy, predictions, y_test, combined_features.shape

def load_balanced_data():
    """Load balanced dataset with 10K+ samples"""
    logger.info("Loading balanced FakeNewsNet dataset...")
    
    try:
        # Load more samples for better balance
        gossipcop_fake = pd.read_csv("data/fakenewsnet_dataset/gossipcop_fake.csv")
        politifact_fake = pd.read_csv("data/fakenewsnet_dataset/politifact_fake.csv")
        gossipcop_real = pd.read_csv("data/fakenewsnet_dataset/gossipcop_real.csv")
        politifact_real = pd.read_csv("data/fakenewsnet_dataset/politifact_real.csv")
        
        # Add labels
        gossipcop_fake['label'] = 1
        politifact_fake['label'] = 1
        gossipcop_real['label'] = 0
        politifact_real['label'] = 0
        
        # Combine all datasets
        all_data = pd.concat([gossipcop_fake, politifact_fake, gossipcop_real, politifact_real], ignore_index=True)
        all_data = all_data.dropna(subset=['title'])
        all_data = all_data[all_data['title'].str.len() > 10]
        
        # Count samples
        fake_count = len(all_data[all_data['label'] == 1])
        real_count = len(all_data[all_data['label'] == 0])
        
        logger.info(f"Total samples: {len(all_data)}")
        logger.info(f"Fake news: {fake_count}")
        logger.info(f"Real news: {real_count}")
        
        # Ensure we have enough samples
        if len(all_data) < 10000:
            logger.warning(f"Only {len(all_data)} samples available, less than 10K requested")
        else:
            logger.info(f"✅ Dataset has {len(all_data)} samples (>= 10K)")
        
        return all_data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def plot_results(y_true, y_pred, save_path):
    """Plot confusion matrix and results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticklabels(['Real', 'Fake'])
    ax1.set_yticklabels(['Real', 'Fake'])
    
    # Accuracy by class
    class_names = ['Real', 'Fake']
    accuracies = []
    for i in range(2):
        class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        accuracies.append(class_acc)
    
    ax2.bar(class_names, accuracies, color=['green', 'red'])
    ax2.set_title('Accuracy by Class')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    
    for i, acc in enumerate(accuracies):
        ax2.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Results plot saved to {save_path}")

def main():
    """Main function"""
    logger.info("🚀 Starting ENHANCED prototype training with 10K+ samples...")
    
    # Load data
    data = load_balanced_data()
    if data is None:
        return
    
    # Prepare data
    texts = data['title'].fillna('').astype(str).tolist()
    labels = data['label'].astype(int).tolist()
    
    logger.info(f"Prepared {len(texts)} texts for training")
    
    # Create and train model
    model = EnhancedFakeNewsDetector()
    accuracy, predictions, true_labels, feature_shape = model.train(texts, labels)
    
    # Save results
    output_dir = Path("models/saved_models/enhanced_prototype")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    plot_path = output_dir / "results_plot.png"
    plot_results(true_labels, predictions, str(plot_path))
    
    results = {
        'model': 'EnhancedFakeNewsDetector',
        'accuracy': accuracy,
        'training_date': datetime.now().isoformat(),
        'device': model.device,
        'total_samples': len(texts),
        'feature_dimensions': feature_shape,
        'balanced_dataset': True,
        'fake_samples': len([l for l in labels if l == 1]),
        'real_samples': len([l for l in labels if l == 0])
    }
    
    import json
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("🎉 ENHANCED TRAINING COMPLETED!")
    logger.info(f"📊 Final Accuracy: {accuracy:.4f}")
    logger.info(f"📈 Total Samples: {len(texts)}")
    logger.info(f"🔧 Feature Dimensions: {feature_shape}")
    logger.info(f"💾 Results saved to: {output_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
