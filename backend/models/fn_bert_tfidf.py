#!/usr/bin/env python3
"""
FN-BERT-TFIDF: Fake News Detection Model combining BERT and TF-IDF
Based on the research paper: "ContCommRTD: A Distributed Content-Based Misinformation-Aware Community Detection System for Real-Time Disaster Reporting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FNBertTfidfModel(nn.Module):
    """
    FN-BERT-TFIDF: Fake News Detection Model combining BERT and TF-IDF
    """
    
    def __init__(self, 
                 bert_model_name: str = "bert-base-uncased",
                 tfidf_dim: int = 5000,
                 bert_dim: int = 768,
                 lstm_hidden: int = 128,
                 cnn_filters: int = 32,
                 cnn_kernel: int = 127,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        """
        Initialize the FN-BERT-TFIDF model
        """
        super(FNBertTfidfModel, self).__init__()
        
        self.bert_model_name = bert_model_name
        self.tfidf_dim = tfidf_dim
        self.bert_dim = bert_dim
        self.lstm_hidden = lstm_hidden
        self.cnn_filters = cnn_filters
        self.cnn_kernel = cnn_kernel
        self.num_classes = num_classes
        self.dropout = dropout
        
        # BERT model for text embeddings
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_dim,
            ngram_range=(1, 3),
            stop_words='english',
            max_df=1.0,
            min_df=1
        )
        
        # Feature fusion layer (will be created dynamically)
        self.feature_fusion = None
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if 1 > 1 else 0
        )
        
        # CNN layer
        self.conv = nn.Conv1d(
            in_channels=lstm_hidden * 2,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel,
            padding='same'
        )
        
        # Global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Dense layers
        self.dense1 = nn.Linear(cnn_filters, 128)
        self.dense2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, num_classes)
        
        # Dropout and batch norm
        self.dropout_layer = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        
        logger.info(f"FN-BERT-TFIDF model initialized with {bert_model_name}")
    
    def _create_feature_fusion(self, combined_features):
        """
        Create feature fusion layer dynamically based on input dimensions
        """
        if self.feature_fusion is None:
            input_dim = combined_features.shape[1]
            self.feature_fusion = nn.Linear(input_dim, self.bert_dim).to(combined_features.device, non_blocking=True)
        return self.feature_fusion(combined_features)
    
    def forward(self, input_ids, attention_mask, tfidf_features):
        """
        Forward pass through the model
        """
        # BERT encoding
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_outputs.last_hidden_state
        
        # Use [CLS] token representation
        bert_cls = bert_embeddings[:, 0, :]
        
        # Concatenate BERT and TF-IDF features
        combined_features = torch.cat([bert_cls, tfidf_features], dim=1)
        
        # Feature fusion
        fused_features = self._create_feature_fusion(combined_features)
        
        # Reshape for LSTM
        lstm_input = fused_features.unsqueeze(1)
        
        # BiLSTM processing
        lstm_output, _ = self.lstm(lstm_input)
        
        # Reshape for CNN
        cnn_input = lstm_output.transpose(1, 2)
        
        # CNN processing
        cnn_output = F.relu(self.conv(cnn_input))
        
        # Global max pooling
        pooled_output = self.global_max_pool(cnn_output).squeeze(-1)
        
        # Dense layers
        x = F.relu(self.batch_norm1(self.dense1(pooled_output)))
        x = self.dropout_layer(x)
        
        x = F.relu(self.batch_norm2(self.dense2(x)))
        x = self.dropout_layer(x)
        
        # Output layer
        logits = self.output_layer(x)
        
        return logits
    
    def get_bert_embeddings(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Get BERT embeddings for a list of texts
        """
        self.bert.eval()
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.bert_tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                outputs = self.bert(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding)
        
        return torch.cat(embeddings, dim=0)
    
    def get_tfidf_features(self, texts: List[str]) -> torch.Tensor:
        """
        Get TF-IDF features for a list of texts
        """
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            self.tfidf_vectorizer.fit(texts)
        
        tfidf_features = self.tfidf_vectorizer.transform(texts)
        features_tensor = torch.FloatTensor(tfidf_features.toarray())
        
        return features_tensor
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Get predictions for a list of texts
        """
        self.eval()
        
        tfidf_features = self.get_tfidf_features(texts)
        bert_embeddings = self.get_bert_embeddings(texts)
        
        inputs = self.bert_tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            logits = self.forward(
                inputs['input_ids'],
                inputs['attention_mask'],
                tfidf_features
            )
            
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions.numpy()
    
    def save_model(self, model_path: str):
        """
        Save the model and components
        """
        os.makedirs(model_path, exist_ok=True)
        
        torch.save(self.state_dict(), os.path.join(model_path, 'model.pth'))
        
        with open(os.path.join(model_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        config = {
            'bert_model_name': self.bert_model_name,
            'tfidf_dim': self.tfidf_dim,
            'bert_dim': self.bert_dim,
            'lstm_hidden': self.lstm_hidden,
            'cnn_filters': self.cnn_filters,
            'cnn_kernel': self.cnn_kernel,
            'num_classes': self.num_classes,
            'dropout': self.dropout
        }
        
        import json
        with open(os.path.join(model_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = FNBertTfidfModel()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Example text
    texts = ["This is a sample text for testing the model."]
    
    # Get predictions
    predictions = model.predict(texts)
    print(f"Predictions: {predictions}")
