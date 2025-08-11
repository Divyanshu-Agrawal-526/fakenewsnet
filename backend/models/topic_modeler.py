#!/usr/bin/env python3
"""
Topic Modeling and Community Detection System
Based on the research paper: "ContCommRTD: A Distributed Content-Based Misinformation-Aware Community Detection System for Real-Time Disaster Reporting"

This implements:
1. Topic modeling using LDA/OLDA
2. Content-based community detection
3. Topic coherence evaluation
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import pickle
import os
from typing import List, Dict, Tuple, Any
import re
from collections import Counter

logger = logging.getLogger(__name__)

class TopicModeler:
    """
    Topic modeling system for disaster-related content
    """
    
    def __init__(self, n_topics: int = 6, max_features: int = 5000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.lda_model = None
        self.vectorizer = None
        self.feature_names = None
        self.topic_keywords = {}
        
        # Initialize LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='batch',
            n_jobs=-1
        )
        
        # Initialize vectorizer
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            stop_words='english',
            max_df=0.95,
            min_df=2
        )
        
        logger.info(f"Topic modeler initialized with {n_topics} topics")
    
    def preprocess_text(self, texts: List[str]) -> List[str]:
        """
        Preprocess text for topic modeling
        """
        processed_texts = []
        
        for text in texts:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove special characters but keep spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Filter out very short texts
            if len(text) > 10:
                processed_texts.append(text)
        
        return processed_texts
    
    def fit(self, texts: List[str]) -> Dict[str, Any]:
        """
        Fit the topic model
        """
        logger.info("Preprocessing texts...")
        processed_texts = self.preprocess_text(texts)
        
        logger.info("Vectorizing texts...")
        # Fit and transform texts
        # Handle very small batches by relaxing df thresholds
        if len(processed_texts) < 3:
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                stop_words='english',
                max_df=1.0,
                min_df=1
            )
        doc_term_matrix = self.vectorizer.fit_transform(processed_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Document-term matrix shape: {doc_term_matrix.shape}")
        
        logger.info("Training LDA model...")
        # Fit LDA model
        self.lda_model.fit(doc_term_matrix)
        
        # Get topic distributions
        topic_distributions = self.lda_model.transform(doc_term_matrix)
        
        # Extract topic keywords
        self._extract_topic_keywords()
        
        # Evaluate model quality
        evaluation_metrics = self._evaluate_model(doc_term_matrix, topic_distributions)
        
        results = {
            'topic_distributions': topic_distributions,
            'doc_term_matrix': doc_term_matrix,
            'evaluation_metrics': evaluation_metrics,
            'processed_texts': processed_texts
        }
        
        logger.info("Topic modeling completed!")
        return results
    
    def _extract_topic_keywords(self):
        """
        Extract top keywords for each topic
        """
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_keywords_idx = topic.argsort()[-10:][::-1]
            top_keywords = [self.feature_names[i] for i in top_keywords_idx]
            self.topic_keywords[topic_idx] = top_keywords
            
            logger.info(f"Topic {topic_idx}: {', '.join(top_keywords[:5])}")
    
    def _evaluate_model(self, doc_term_matrix, topic_distributions) -> Dict[str, float]:
        """
        Evaluate topic model quality
        """
        # Perplexity (lower is better)
        perplexity = self.lda_model.perplexity(doc_term_matrix)
        
        # Log likelihood (higher is better)
        log_likelihood = self.lda_model.score(doc_term_matrix)
        
        # Topic coherence (simplified version)
        coherence_scores = []
        for topic_idx in range(self.n_topics):
            topic_words = self.topic_keywords[topic_idx][:10]
            # Simple coherence: count co-occurrences
            coherence = self._calculate_topic_coherence(topic_words, doc_term_matrix)
            coherence_scores.append(coherence)
        
        avg_coherence = np.mean(coherence_scores)
        
        metrics = {
            'perplexity': perplexity,
            'log_likelihood': log_likelihood,
            'avg_coherence': avg_coherence,
            'topic_coherence_scores': coherence_scores
        }
        
        logger.info(f"Model Evaluation - Perplexity: {perplexity:.2f}, Log Likelihood: {log_likelihood:.2f}, Avg Coherence: {avg_coherence:.4f}")
        
        return metrics
    
    def _calculate_topic_coherence(self, topic_words: List[str], doc_term_matrix) -> float:
        """
        Calculate topic coherence based on word co-occurrences
        """
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get indices of topic words
        word_indices = []
        for word in topic_words:
            try:
                idx = np.where(feature_names == word)[0][0]
                word_indices.append(idx)
            except:
                continue
        
        if len(word_indices) < 2:
            return 0.0
        
        # Calculate pairwise co-occurrences
        coherence_score = 0.0
        pairs = 0
        
        for i in range(len(word_indices)):
            for j in range(i + 1, len(word_indices)):
                word1_idx = word_indices[i]
                word2_idx = word_indices[j]
                
                # Count co-occurrences - convert to dense array for boolean operations
                doc_term_dense = doc_term_matrix.toarray() if hasattr(doc_term_matrix, 'toarray') else doc_term_matrix
                co_occurrences = np.sum((doc_term_dense[:, word1_idx] > 0) & (doc_term_dense[:, word2_idx] > 0))
                total_docs = doc_term_dense.shape[0]
                
                if total_docs > 0:
                    coherence_score += co_occurrences / total_docs
                    pairs += 1
        
        return coherence_score / pairs if pairs > 0 else 0.0
    
    def get_topic_assignments(self, texts: List[str]) -> np.ndarray:
        """
        Get topic assignments for new texts
        """
        processed_texts = self.preprocess_text(texts)
        doc_term_matrix = self.vectorizer.transform(processed_texts)
        topic_distributions = self.lda_model.transform(doc_term_matrix)
        
        # Return dominant topic for each document
        return np.argmax(topic_distributions, axis=1)
    
    def plot_topics(self, save_path: str = None):
        """
        Plot topic visualization
        """
        if not hasattr(self, 'lda_model') or self.lda_model is None:
            logger.error("Model not fitted yet!")
            return
        
        # Create topic-word heatmap
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Topic-word heatmap
        topic_word_matrix = self.lda_model.components_[:, :20]  # Top 20 words
        sns.heatmap(topic_word_matrix, 
                   xticklabels=self.feature_names[:20], 
                   yticklabels=[f'Topic {i}' for i in range(self.n_topics)],
                   ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Topic-Word Distribution')
        axes[0, 0].set_xlabel('Words')
        axes[0, 0].set_ylabel('Topics')
        
        # Top keywords per topic
        for i in range(min(4, self.n_topics)):
            topic_words = self.topic_keywords[i][:8]
            word_scores = self.lda_model.components_[i][:8]
            
            axes[0, 1].barh(range(len(topic_words)), word_scores)
            axes[0, 1].set_yticks(range(len(topic_words)))
            axes[0, 1].set_yticklabels(topic_words)
            axes[0, 1].set_title(f'Topic {i} Keywords')
            axes[0, 1].set_xlabel('Word Score')
        
        # Topic distribution
        topic_weights = np.sum(self.lda_model.components_, axis=1)
        axes[1, 0].pie(topic_weights, labels=[f'Topic {i}' for i in range(self.n_topics)], autopct='%1.1f%%')
        axes[1, 0].set_title('Topic Distribution')
        
        # Evaluation metrics
        if hasattr(self, 'evaluation_metrics'):
            metrics = ['perplexity', 'log_likelihood', 'avg_coherence']
            values = [self.evaluation_metrics[m] for m in metrics]
            axes[1, 1].bar(metrics, values)
            axes[1, 1].set_title('Model Evaluation Metrics')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Topic visualization saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str):
        """
        Save the trained topic model
        """
        os.makedirs(model_path, exist_ok=True)
        
        with open(os.path.join(model_path, 'lda_model.pkl'), 'wb') as f:
            pickle.dump(self.lda_model, f)
        
        with open(os.path.join(model_path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(os.path.join(model_path, 'topic_keywords.pkl'), 'wb') as f:
            pickle.dump(self.topic_keywords, f)
        
        logger.info(f"Topic model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path: str):
        """
        Load a saved topic model
        """
        model = cls()
        
        with open(os.path.join(model_path, 'lda_model.pkl'), 'rb') as f:
            model.lda_model = pickle.load(f)
        
        with open(os.path.join(model_path, 'vectorizer.pkl'), 'rb') as f:
            model.vectorizer = pickle.load(f)
        
        with open(os.path.join(model_path, 'topic_keywords.pkl'), 'rb') as f:
            model.topic_keywords = pickle.load(f)
        
        model.feature_names = model.vectorizer.get_feature_names_out()
        
        logger.info(f"Topic model loaded from {model_path}")
        return model

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create topic modeler
    topic_modeler = TopicModeler(n_topics=6)
    
    # Example texts (replace with your actual data)
    sample_texts = [
        "Hurricane warning issued for coastal areas",
        "Earthquake hits region with magnitude 6.5",
        "Flooding reported in downtown area",
        "Wildfire spreads rapidly through forest",
        "Tornado warning for multiple counties",
        "Heavy rainfall causes flash floods"
    ]
    
    # Fit model
    results = topic_modeler.fit(sample_texts)
    
    # Plot results
    topic_modeler.plot_topics()
    
    print("Topic modeling example completed!")
