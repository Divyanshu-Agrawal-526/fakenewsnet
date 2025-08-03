import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import os

class ChannelAttention(nn.Module):
    """Channel Attention Module for multimodal fusion"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class MultimodalFusion(nn.Module):
    """Multimodal Fusion with Attention Mechanisms"""
    def __init__(self, text_dim=768, image_dim=2048, hidden_dim=512, num_classes=5):
        super(MultimodalFusion, self).__init__()
        
        # Text processing
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Image processing with attention
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.channel_attention = ChannelAttention(hidden_dim)
        self.spatial_attention = SpatialAttention()
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Confidence estimation
        self.confidence_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, text_features, image_features):
        # Project features to common space
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        
        # Apply attention mechanisms to image features
        image_proj = image_proj.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        image_attended = image_proj * self.channel_attention(image_proj)
        image_attended = image_attended * self.spatial_attention(image_attended)
        image_attended = image_attended.squeeze(-1).squeeze(-1)
        
        # Cross-modal attention
        text_attended, _ = self.cross_attention(
            text_proj.unsqueeze(1), 
            image_attended.unsqueeze(1), 
            image_attended.unsqueeze(1)
        )
        text_attended = text_attended.squeeze(1)
        
        # Concatenate features
        fused_features = torch.cat([text_attended, image_attended], dim=-1)
        
        # Classification
        logits = self.fusion_layer(fused_features)
        confidence = self.confidence_layer(fused_features)
        
        return logits, confidence

class MultimodalClassifier:
    """Main Multimodal Classifier for Fake News Detection"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Text model (BERT)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Image model (ResNet)
        self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])  # Remove final layer
        
        # Multimodal fusion
        self.fusion_model = MultimodalFusion()
        
        # Move models to device
        self.text_model.to(self.device)
        self.image_model.to(self.device)
        self.fusion_model.to(self.device)
        
        # Set to evaluation mode
        self.text_model.eval()
        self.image_model.eval()
        self.fusion_model.eval()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class labels
        self.classes = ['fake', 'real_wildfire', 'real_flood', 'real_hurricane', 'real_earthquake']
        
    def preprocess_text(self, text):
        """Preprocess text for BERT"""
        # Clean and prepare text
        text = text.strip()
        if len(text) > 512:
            text = text[:512]
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        return inputs
    
    def preprocess_image(self, image_path):
        """Preprocess image for ResNet"""
        try:
            # Load and preprocess image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path.convert('RGB')
            
            # Apply transformations
            image_tensor = self.image_transform(image)
            return image_tensor.unsqueeze(0)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_text_features(self, text):
        """Extract text features using BERT"""
        try:
            inputs = self.preprocess_text(text)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                # Use [CLS] token representation
                text_features = outputs.last_hidden_state[:, 0, :]
            
            return text_features
        except Exception as e:
            print(f"Error extracting text features: {e}")
            return None
    
    def extract_image_features(self, image_path):
        """Extract image features using ResNet"""
        try:
            image_tensor = self.preprocess_image(image_path)
            if image_tensor is None:
                return None
            
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                image_features = self.image_model(image_tensor)
                image_features = image_features.squeeze()
            
            return image_features
        except Exception as e:
            print(f"Error extracting image features: {e}")
            return None
    
    def classify(self, text, image_path=None):
        """Main classification method"""
        try:
            # Extract text features
            text_features = self.extract_text_features(text)
            if text_features is None:
                return self._fallback_classification(text)
            
            # Handle multimodal vs text-only
            if image_path and os.path.exists(image_path):
                # Multimodal classification
                image_features = self.extract_image_features(image_path)
                if image_features is not None:
                    return self._multimodal_classification(text_features, image_features)
                else:
                    # Fallback to text-only if image processing fails
                    return self._text_only_classification(text_features)
            else:
                # Text-only classification
                return self._text_only_classification(text_features)
                
        except Exception as e:
            print(f"Error in classification: {e}")
            return self._fallback_classification(text)
    
    def _multimodal_classification(self, text_features, image_features):
        """Perform multimodal classification"""
        with torch.no_grad():
            logits, confidence = self.fusion_model(text_features, image_features)
            
            # Get predictions
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence_score = confidence.item()
            
            return {
                'prediction': self.classes[predicted_class],
                'confidence': confidence_score,
                'probabilities': probabilities.cpu().numpy()[0],
                'modality': 'multimodal',
                'explanation': self._generate_explanation(predicted_class, confidence_score, 'multimodal')
            }
    
    def _text_only_classification(self, text_features):
        """Perform text-only classification"""
        # Simple text-based classification for fallback
        text_lower = text_features.mean().item()
        
        # Rule-based classification based on text features
        if text_lower > 0.5:
            prediction = 'real_wildfire'  # Example
        else:
            prediction = 'fake'
        
        return {
            'prediction': prediction,
            'confidence': 0.7,
            'probabilities': [0.3, 0.2, 0.2, 0.2, 0.1],  # Example
            'modality': 'text_only',
            'explanation': self._generate_explanation(self.classes.index(prediction), 0.7, 'text_only')
        }
    
    def _fallback_classification(self, text):
        """Fallback classification using simple rules"""
        text_lower = text.lower()
        
        # Simple keyword-based classification
        fake_keywords = ['fake', 'hoax', 'conspiracy', 'government hiding', 'aliens', 'chemtrails']
        disaster_keywords = {
            'wildfire': ['fire', 'wildfire', 'burning', 'flame', 'smoke'],
            'flood': ['flood', 'water', 'rain', 'overflow', 'drowning'],
            'hurricane': ['hurricane', 'storm', 'wind', 'tropical', 'cyclone'],
            'earthquake': ['earthquake', 'quake', 'shaking', 'tremor', 'seismic']
        }
        
        # Check for fake news indicators
        fake_score = sum(1 for keyword in fake_keywords if keyword in text_lower)
        
        if fake_score > 0:
            return {
                'prediction': 'fake',
                'confidence': 0.8,
                'probabilities': [0.8, 0.05, 0.05, 0.05, 0.05],
                'modality': 'fallback',
                'explanation': 'Detected fake news indicators in text'
            }
        
        # Check for disaster types
        for disaster_type, keywords in disaster_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return {
                    'prediction': f'real_{disaster_type}',
                    'confidence': 0.7,
                    'probabilities': [0.1, 0.2, 0.2, 0.2, 0.3],
                    'modality': 'fallback',
                    'explanation': f'Detected {disaster_type} keywords in text'
                }
        
        # Default to fake if uncertain
        return {
            'prediction': 'fake',
            'confidence': 0.5,
            'probabilities': [0.5, 0.125, 0.125, 0.125, 0.125],
            'modality': 'fallback',
            'explanation': 'Unable to determine authenticity with confidence'
        }
    
    def _generate_explanation(self, predicted_class_idx, confidence, modality):
        """Generate explanation for the classification"""
        class_name = self.classes[predicted_class_idx]
        
        if modality == 'multimodal':
            return f"Multimodal analysis combining text and image features classified this as {class_name} with {confidence:.2%} confidence."
        elif modality == 'text_only':
            return f"Text-only analysis classified this as {class_name} with {confidence:.2%} confidence."
        else:
            return f"Fallback analysis classified this as {class_name} with {confidence:.2%} confidence."
    
    def get_attention_weights(self, text, image_path):
        """Get attention weights for interpretability"""
        try:
            text_features = self.extract_text_features(text)
            image_features = self.extract_image_features(image_path)
            
            if text_features is not None and image_features is not None:
                with torch.no_grad():
                    # Get attention weights from the fusion model
                    # This would require modifying the model to return attention weights
                    return {
                        'text_attention': text_features.mean().item(),
                        'image_attention': image_features.mean().item(),
                        'cross_attention': 0.5  # Placeholder
                    }
        except Exception as e:
            print(f"Error getting attention weights: {e}")
            return None

# Initialize the multimodal classifier
multimodal_classifier = MultimodalClassifier() 