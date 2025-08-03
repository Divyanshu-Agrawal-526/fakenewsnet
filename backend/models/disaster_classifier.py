import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import os
import logging
from typing import Dict, Any
import re
import nltk
from nltk.corpus import stopwords

class DisasterClassifier:
    def __init__(self):
        self.model_path = 'models/saved/disaster_classifier.pkl'
        self.bert_model_path = 'models/saved/bert_disaster_classifier'
        self.tokenizer_path = 'models/saved/bert_disaster_tokenizer'
        self.vectorizer_path = 'models/saved/disaster_tfidf_vectorizer.pkl'
        
        self.model = None
        self.bert_model = None
        self.tokenizer = None
        self.vectorizer = None
        self.is_model_loaded = False
        
        # Disaster categories
        self.disaster_categories = ['wildfire', 'flood', 'hurricane', 'earthquake']
        self.category_mapping = {0: 'wildfire', 1: 'flood', 2: 'hurricane', 3: 'earthquake'}
        
        # Statistics
        self.total_classifications = 0
        self.category_counts = {category: 0 for category in self.disaster_categories}
        
        # Create models directory if it doesn't exist
        os.makedirs('models/saved', exist_ok=True)
        
    def load_model(self):
        """Load the trained disaster classification model"""
        try:
            # Load BERT model and tokenizer
            if os.path.exists(self.bert_model_path):
                self.bert_model = BertForSequenceClassification.from_pretrained(self.bert_model_path)
                self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
                self.bert_model.eval()
            else:
                # Initialize with pre-trained BERT
                self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Load ensemble model
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            
            # Load TF-IDF vectorizer
            if os.path.exists(self.vectorizer_path):
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            else:
                self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
            
            self.is_model_loaded = True
            logging.info("Disaster classifier model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading disaster classifier model: {e}")
            self.is_model_loaded = False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for disaster classification"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_disaster_features(self, text: str) -> Dict[str, Any]:
        """Extract disaster-specific features from text"""
        features = {}
        
        # Basic text features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Disaster-specific keywords for each category
        disaster_keywords = {
            'wildfire': ['fire', 'wildfire', 'blaze', 'burning', 'smoke', 'flame', 'forest fire', 'brush fire', 'bush fire'],
            'flood': ['flood', 'water', 'rain', 'overflow', 'river', 'stream', 'drainage', 'sewage', 'water level'],
            'hurricane': ['hurricane', 'storm', 'wind', 'tropical', 'cyclone', 'typhoon', 'gale', 'gust', 'rainfall'],
            'earthquake': ['earthquake', 'quake', 'tremor', 'seismic', 'shaking', 'ground', 'magnitude', 'epicenter']
        }
        
        text_lower = text.lower()
        
        # Count keywords for each disaster type
        for disaster_type, keywords in disaster_keywords.items():
            features[f'{disaster_type}_keyword_count'] = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Sentiment analysis
        from textblob import TextBlob
        blob = TextBlob(text)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity
        
        # Urgency indicators
        urgency_words = ['urgent', 'emergency', 'immediate', 'critical', 'dangerous', 'evacuate', 'warning', 'alert']
        features['urgency_count'] = sum(1 for word in urgency_words if word in text_lower)
        
        # Location indicators
        location_words = ['area', 'region', 'city', 'town', 'county', 'state', 'street', 'road', 'highway']
        features['location_count'] = sum(1 for word in location_words if word in text_lower)
        
        return features
    
    def predict_bert(self, text: str) -> Dict[str, Any]:
        """Predict disaster type using BERT model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            return {
                'prediction': self.category_mapping.get(prediction, 'unknown'),
                'confidence': confidence,
                'probabilities': probabilities[0].tolist()
            }
            
        except Exception as e:
            logging.error(f"Error in BERT disaster prediction: {e}")
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'probabilities': [0.25, 0.25, 0.25, 0.25]
            }
    
    def predict_ensemble(self, text: str) -> Dict[str, Any]:
        """Predict disaster type using ensemble model"""
        try:
            if self.model is None:
                return {'prediction': 'unknown', 'confidence': 0.0}
            
            # Extract features
            features = self.extract_disaster_features(text)
            feature_vector = list(features.values())
            
            # Get TF-IDF features
            if self.vectorizer:
                tfidf_features = self.vectorizer.transform([text]).toarray()[0]
                feature_vector.extend(tfidf_features)
            
            # Make prediction
            prediction = self.model.predict([feature_vector])[0]
            confidence = self.model.predict_proba([feature_vector])[0].max()
            
            return {
                'prediction': self.category_mapping.get(prediction, 'unknown'),
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error in ensemble disaster prediction: {e}")
            return {'prediction': 'unknown', 'confidence': 0.0}
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Main prediction method combining BERT and ensemble for disaster classification"""
        if not self.is_model_loaded:
            self.load_model()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get predictions from both models
        bert_result = self.predict_bert(processed_text)
        ensemble_result = self.predict_ensemble(processed_text)
        
        # Combine predictions
        if bert_result['prediction'] != 'unknown' and ensemble_result['prediction'] != 'unknown':
            # Weighted average of confidences
            bert_weight = 0.6
            ensemble_weight = 0.4
            
            final_confidence = (bert_weight * bert_result['confidence'] + 
                              ensemble_weight * ensemble_result['confidence'])
            
            # Determine final prediction
            if bert_result['prediction'] == ensemble_result['prediction']:
                final_prediction = bert_result['prediction']
            else:
                # Use the one with higher confidence
                final_prediction = bert_result['prediction'] if bert_result['confidence'] > ensemble_result['confidence'] else ensemble_result['prediction']
        else:
            # Fallback to BERT if ensemble fails
            final_prediction = bert_result['prediction']
            final_confidence = bert_result['confidence']
        
        # Generate explanation
        explanation = self.generate_disaster_explanation(processed_text, final_prediction, final_confidence)
        
        # Update statistics
        self.total_classifications += 1
        if final_prediction in self.category_counts:
            self.category_counts[final_prediction] += 1
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'explanation': explanation,
            'bert_result': bert_result,
            'ensemble_result': ensemble_result
        }
    
    def generate_disaster_explanation(self, text: str, prediction: str, confidence: float) -> str:
        """Generate human-readable explanation for disaster classification"""
        if prediction == 'wildfire':
            if confidence > 0.8:
                return "High confidence wildfire classification based on fire-related keywords and context."
            elif confidence > 0.6:
                return "Moderate confidence wildfire classification."
            else:
                return "Low confidence wildfire classification - may need additional verification."
        
        elif prediction == 'flood':
            if confidence > 0.8:
                return "High confidence flood classification based on water-related keywords and context."
            elif confidence > 0.6:
                return "Moderate confidence flood classification."
            else:
                return "Low confidence flood classification - may need additional verification."
        
        elif prediction == 'hurricane':
            if confidence > 0.8:
                return "High confidence hurricane classification based on storm-related keywords and context."
            elif confidence > 0.6:
                return "Moderate confidence hurricane classification."
            else:
                return "Low confidence hurricane classification - may need additional verification."
        
        elif prediction == 'earthquake':
            if confidence > 0.8:
                return "High confidence earthquake classification based on seismic-related keywords and context."
            elif confidence > 0.6:
                return "Moderate confidence earthquake classification."
            else:
                return "Low confidence earthquake classification - may need additional verification."
        
        else:
            return "Unable to classify disaster type with sufficient confidence."
    
    def train_model(self, training_data: list, labels: list):
        """Train the disaster classification model with new data"""
        try:
            from sklearn.model_selection import train_test_split
            
            # Preprocess training data
            processed_texts = [self.preprocess_text(text) for text in training_data]
            
            # Convert labels to numeric
            label_mapping = {category: idx for idx, category in enumerate(self.disaster_categories)}
            numeric_labels = [label_mapping.get(label, 0) for label in labels]
            
            # Extract features
            features_list = []
            for text in processed_texts:
                features = self.extract_disaster_features(text)
                features_list.append(list(features.values()))
            
            # Fit TF-IDF vectorizer
            self.vectorizer.fit(processed_texts)
            
            # Combine features
            tfidf_features = self.vectorizer.transform(processed_texts).toarray()
            combined_features = np.hstack([features_list, tfidf_features])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                combined_features, numeric_labels, test_size=0.2, random_state=42
            )
            
            # Train ensemble model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Save models
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Evaluate
            accuracy = self.model.score(X_test, y_test)
            logging.info(f"Disaster classifier trained with accuracy: {accuracy:.3f}")
            
            return accuracy
            
        except Exception as e:
            logging.error(f"Error training disaster classifier: {e}")
            return 0.0
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.is_model_loaded
    
    def get_total_classifications(self) -> int:
        """Get total number of classifications performed"""
        return self.total_classifications
    
    def get_disaster_distribution(self) -> Dict[str, int]:
        """Get distribution of disaster classifications"""
        return self.category_counts.copy() 