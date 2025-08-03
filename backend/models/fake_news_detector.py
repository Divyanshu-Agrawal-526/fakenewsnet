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
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FakeNewsDetector:
    def __init__(self):
        self.model_path = 'models/saved/fake_news_detector.pkl'
        self.bert_model_path = 'models/saved/bert_fake_news'
        self.tokenizer_path = 'models/saved/bert_tokenizer'
        self.vectorizer_path = 'models/saved/tfidf_vectorizer.pkl'
        
        self.model = None
        self.bert_model = None
        self.tokenizer = None
        self.vectorizer = None
        self.is_model_loaded = False
        
        # Statistics
        self.total_analyses = 0
        self.correct_predictions = 0
        
        # Create models directory if it doesn't exist
        os.makedirs('models/saved', exist_ok=True)
        
    def load_model(self):
        """Load the trained model"""
        try:
            # Load BERT model and tokenizer
            if os.path.exists(self.bert_model_path):
                self.bert_model = BertForSequenceClassification.from_pretrained(self.bert_model_path)
                self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
                self.bert_model.eval()
            else:
                # Initialize with pre-trained BERT
                self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
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
                self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            
            self.is_model_loaded = True
            logging.info("Fake news detector model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading fake news detector model: {e}")
            self.is_model_loaded = False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
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
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract various features from text"""
        features = {}
        
        # Basic text features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Sentiment features
        from textblob import TextBlob
        blob = TextBlob(text)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity
        
        # Language features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        
        # Disaster-related keywords
        disaster_keywords = [
            'fire', 'flood', 'hurricane', 'earthquake', 'tsunami', 'tornado',
            'disaster', 'emergency', 'evacuation', 'damage', 'destruction',
            'rescue', 'help', 'urgent', 'warning', 'alert'
        ]
        
        text_lower = text.lower()
        features['disaster_keyword_count'] = sum(1 for keyword in disaster_keywords if keyword in text_lower)
        
        return features
    
    def predict_bert(self, text: str) -> Dict[str, Any]:
        """Predict using BERT model"""
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
                'prediction': 'real' if prediction == 1 else 'fake',
                'confidence': confidence,
                'probabilities': probabilities[0].tolist()
            }
            
        except Exception as e:
            logging.error(f"Error in BERT prediction: {e}")
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'probabilities': [0.5, 0.5]
            }
    
    def predict_ensemble(self, text: str) -> Dict[str, Any]:
        """Predict using ensemble model"""
        try:
            if self.model is None:
                return {'prediction': 'unknown', 'confidence': 0.0}
            
            # Extract features
            features = self.extract_features(text)
            feature_vector = list(features.values())
            
            # Get TF-IDF features
            if self.vectorizer:
                tfidf_features = self.vectorizer.transform([text]).toarray()[0]
                feature_vector.extend(tfidf_features)
            
            # Make prediction
            prediction = self.model.predict([feature_vector])[0]
            confidence = self.model.predict_proba([feature_vector])[0].max()
            
            return {
                'prediction': 'real' if prediction == 1 else 'fake',
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error in ensemble prediction: {e}")
            return {'prediction': 'unknown', 'confidence': 0.0}
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Main prediction method combining BERT and ensemble"""
        if not self.is_model_loaded:
            self.load_model()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get predictions from both models
        bert_result = self.predict_bert(processed_text)
        ensemble_result = self.predict_ensemble(processed_text)
        
        # Combine predictions (ensemble approach)
        if bert_result['prediction'] != 'unknown' and ensemble_result['prediction'] != 'unknown':
            # Weighted average of confidences
            bert_weight = 0.7
            ensemble_weight = 0.3
            
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
        explanation = self.generate_explanation(processed_text, final_prediction, final_confidence)
        
        # Update statistics
        self.total_analyses += 1
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'explanation': explanation,
            'bert_result': bert_result,
            'ensemble_result': ensemble_result
        }
    
    def generate_explanation(self, text: str, prediction: str, confidence: float) -> str:
        """Generate human-readable explanation for the prediction"""
        if prediction == 'real':
            if confidence > 0.8:
                return "High confidence that this is a real disaster report based on credible language patterns and disaster-related keywords."
            elif confidence > 0.6:
                return "Moderate confidence that this appears to be a genuine disaster report."
            else:
                return "Low confidence prediction - this may be a real disaster report but requires additional verification."
        else:
            if confidence > 0.8:
                return "High confidence that this is likely fake news based on suspicious language patterns and lack of credible disaster indicators."
            elif confidence > 0.6:
                return "Moderate confidence that this appears to be fake news."
            else:
                return "Low confidence prediction - this may be fake news but requires additional fact-checking."
    
    def train_model(self, training_data: list, labels: list):
        """Train the ensemble model with new data"""
        try:
            from sklearn.model_selection import train_test_split
            
            # Preprocess training data
            processed_texts = [self.preprocess_text(text) for text in training_data]
            
            # Extract features
            features_list = []
            for text in processed_texts:
                features = self.extract_features(text)
                features_list.append(list(features.values()))
            
            # Fit TF-IDF vectorizer
            self.vectorizer.fit(processed_texts)
            
            # Combine features
            tfidf_features = self.vectorizer.transform(processed_texts).toarray()
            combined_features = np.hstack([features_list, tfidf_features])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                combined_features, labels, test_size=0.2, random_state=42
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
            logging.info(f"Model trained with accuracy: {accuracy:.3f}")
            
            return accuracy
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return 0.0
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.is_model_loaded
    
    def get_total_analyses(self) -> int:
        """Get total number of analyses performed"""
        return self.total_analyses
    
    def get_accuracy(self) -> float:
        """Get current accuracy (placeholder for actual tracking)"""
        return 0.85  # Placeholder accuracy 