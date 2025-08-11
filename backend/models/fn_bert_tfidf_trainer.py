#!/usr/bin/env python3
"""
Trainer class for FN-BERT-TFIDF model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Any
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class FakeNewsDataset(Dataset):
    """
    Custom dataset for fake news detection
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FNBertTfidfTrainer:
    """
    Trainer class for FN-BERT-TFIDF model
    """
    
    def __init__(self, 
                 model,
                 device: str = 'cpu',
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 batch_size: int = 16):
        """
        Initialize trainer
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=3,
            gamma=0.1
        )
        
        logger.info(f"Trainer initialized with device: {device}")
    
    def prepare_data(self, texts, labels, test_size=0.2, val_size=0.1):
        """
        Prepare train/validation/test splits
        """
        # First split: train + temp
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=test_size + val_size, random_state=42, stratify=labels
        )
        
        # Second split: validation + test
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=val_size/(test_size + val_size), random_state=42, stratify=temp_labels
        )
        
        # Create datasets
        train_dataset = FakeNewsDataset(train_texts, train_labels, self.model.bert_tokenizer)
        val_dataset = FakeNewsDataset(val_texts, val_labels, self.model.bert_tokenizer)
        test_dataset = FakeNewsDataset(test_texts, test_labels, self.model.bert_tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Get TF-IDF features
            texts = [self.model.bert_tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            tfidf_features = self.model.get_tfidf_features(texts).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, tfidf_features)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        # Update learning rate
        self.scheduler.step()
        
        metrics = {
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        logger.info(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return metrics
    
    def evaluate(self, data_loader, split_name: str = "Validation") -> Dict[str, float]:
        """
        Evaluate the model
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"{split_name}", leave=False):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get TF-IDF features
                texts = [self.model.bert_tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                tfidf_features = self.model.get_tfidf_features(texts).to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask, tfidf_features)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            f'{split_name.lower()}_loss': avg_loss,
            f'{split_name.lower()}_accuracy': accuracy,
            f'{split_name.lower()}_precision': precision,
            f'{split_name.lower()}_recall': recall,
            f'{split_name.lower()}_f1': f1
        }
        
        logger.info(f'{split_name}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        return metrics, all_predictions, all_labels
    
    def train(self, 
              texts, 
              labels, 
              num_epochs: int = 10,
              save_path: str = None) -> Dict[str, List[float]]:
        """
        Train the model
        """
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(texts, labels)
        
        best_val_accuracy = 0
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress", total=num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics, _, _ = self.evaluate(val_loader, "Validation")
            
            # Store history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_accuracy'].append(val_metrics['val_accuracy'])
            history['val_precision'].append(val_metrics['val_precision'])
            history['val_recall'].append(val_metrics['val_recall'])
            history['val_f1'].append(val_metrics['val_f1'])
            history['learning_rate'].append(train_metrics['learning_rate'])
            
            # Save best model
            if val_metrics['val_accuracy'] > best_val_accuracy and save_path:
                best_val_accuracy = val_metrics['val_accuracy']
                self.model.save_model(save_path)
                logger.info(f'New best model saved with validation accuracy: {best_val_accuracy:.2f}%')
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_metrics, test_predictions, test_labels = self.evaluate(test_loader, "Test")
        
        # Print final results
        logger.info("=" * 50)
        logger.info("FINAL RESULTS:")
        logger.info(f"Test Accuracy: {test_metrics['test_accuracy']:.2f}%")
        logger.info(f"Test Precision: {test_metrics['test_precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['test_recall']:.4f}")
        logger.info(f"Test F1: {test_metrics['test_f1']:.4f}")
        logger.info("=" * 50)
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(test_labels, test_predictions, target_names=['Real', 'Fake']))
        
        return history, test_metrics
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str = None):
        """
        Plot training history
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision, Recall, F1
        axes[1, 0].plot(history['val_precision'], label='Precision')
        axes[1, 0].plot(history['val_recall'], label='Recall')
        axes[1, 0].plot(history['val_f1'], label='F1 Score')
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from fn_bert_tfidf import FNBertTfidfModel
    
    # Create model
    model = FNBertTfidfModel()
    
    # Create trainer
    trainer = FNBertTfidfTrainer(model, device='cpu')
    
    # Example data
    texts = ["This is a real news article.", "This is fake news!"]
    labels = [0, 1]  # 0: real, 1: fake
    
    print("Trainer created successfully!")
