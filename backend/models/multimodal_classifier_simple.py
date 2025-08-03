import numpy as np
import re
from PIL import Image
import os

class SimpleMultimodalClassifier:
    """Simplified Multimodal Classifier for Demo Purposes"""
    
    def __init__(self):
        self.classes = ['fake', 'real_wildfire', 'real_flood', 'real_hurricane', 'real_earthquake']
        
        # Keywords for different categories
        self.fake_keywords = [
            'fake', 'hoax', 'conspiracy', 'government hiding', 'aliens', 'chemtrails',
            'fake news', 'false', 'misinformation', 'disinformation', 'clickbait',
            'BREAKING', 'SHOCKING', 'YOU WON\'T BELIEVE', 'VIRAL', 'MUST SHARE'
        ]
        
        self.disaster_keywords = {
            'wildfire': ['fire', 'wildfire', 'burning', 'flame', 'smoke', 'blaze', 'forest fire'],
            'flood': ['flood', 'water', 'rain', 'overflow', 'drowning', 'submerged', 'water level'],
            'hurricane': ['hurricane', 'storm', 'wind', 'tropical', 'cyclone', 'typhoon', 'gale'],
            'earthquake': ['earthquake', 'quake', 'shaking', 'tremor', 'seismic', 'magnitude', 'epicenter']
        }
        
        # Image analysis patterns (simplified)
        self.image_patterns = {
            'wildfire': ['orange', 'red', 'fire', 'smoke', 'burning'],
            'flood': ['blue', 'water', 'flooded', 'submerged', 'rain'],
            'hurricane': ['gray', 'storm', 'clouds', 'wind', 'tropical'],
            'earthquake': ['gray', 'debris', 'cracked', 'damaged', 'destruction']
        }
    
    def analyze_text(self, text):
        """Analyze text for fake news indicators and disaster types"""
        text_lower = text.lower()
        
        # Check for fake news indicators
        fake_score = 0
        for keyword in self.fake_keywords:
            if keyword.lower() in text_lower:
                fake_score += 1
        
        # Check for disaster types
        disaster_scores = {}
        for disaster_type, keywords in self.disaster_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            disaster_scores[disaster_type] = score
        
        # Additional fake news detection based on writing style
        # Check for excessive capitalization
        caps_ratio = sum(1 for char in text if char.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:  # More than 30% caps
            fake_score += 1
        
        # Check for excessive punctuation
        exclamation_count = text.count('!')
        if exclamation_count > 2:  # More than 2 exclamation marks
            fake_score += 1
        
        # Check for sensationalist language
        sensational_words = ['BREAKING', 'SHOCKING', 'INCREDIBLE', 'AMAZING', 'UNBELIEVABLE']
        if any(word.lower() in text_lower for word in sensational_words):
            fake_score += 1
        
        return fake_score, disaster_scores
    
    def analyze_image_simple(self, image_path):
        """Simple image analysis using basic color and pattern detection"""
        try:
            if not image_path or not os.path.exists(image_path):
                return None
            
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image size
            width, height = image.size
            
            # Simple color analysis
            colors = image.getcolors(maxcolors=1000)
            if not colors:
                return None
            
            # Analyze dominant colors
            dominant_colors = []
            for count, color in colors:
                if count > (width * height * 0.01):  # More than 1% of pixels
                    dominant_colors.append(color)
            
            # Simple pattern detection based on colors
            pattern_scores = {}
            for disaster_type, patterns in self.image_patterns.items():
                score = 0
                for color in dominant_colors:
                    r, g, b = color
                    if 'orange' in patterns or 'red' in patterns:
                        if r > 150 and g > 100:  # Orange/red tones
                            score += 1
                    if 'blue' in patterns:
                        if b > 150 and r < 100:  # Blue tones
                            score += 1
                    if 'gray' in patterns:
                        if abs(r - g) < 20 and abs(g - b) < 20:  # Gray tones
                            score += 1
                pattern_scores[disaster_type] = score
            
            return pattern_scores
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None
    
    def classify(self, text, image_path=None):
        """Main classification method"""
        try:
            # Analyze text
            fake_score, disaster_scores = self.analyze_text(text)
            
            # Analyze image if provided
            image_scores = None
            if image_path:
                image_scores = self.analyze_image_simple(image_path)
            
            # Enhanced fake news detection logic
            text_lower = text.lower()
            
            # Check for multiple fake news indicators
            fake_indicators = [
                'fake', 'hoax', 'conspiracy', 'government hiding', 'aliens', 'chemtrails',
                'fake news', 'false', 'misinformation', 'disinformation', 'clickbait',
                'BREAKING', 'SHOCKING', 'YOU WON\'T BELIEVE', 'VIRAL', 'MUST SHARE',
                'share this', 'everyone must know', 'truth revealed', 'hidden truth',
                'they don\'t want you to know', 'mainstream media hiding', 'real truth',
                'wake up', 'sheeple', 'deep state', 'illuminati', 'nwo', 'agenda'
            ]
            
            # Count fake indicators
            fake_indicator_count = sum(1 for indicator in fake_indicators if indicator.lower() in text_lower)
            
            # Check for suspicious patterns
            suspicious_patterns = [
                '!', '!!', '!!!',  # Excessive exclamation marks
                'ALERT', 'URGENT', 'CRITICAL',  # Alarmist language
                'BREAKING NEWS', 'JUST IN', 'DEVELOPING',  # Sensationalist headlines
                'SHARE NOW', 'PASS IT ON', 'TELL EVERYONE',  # Viral sharing requests
                'THEY DON\'T WANT YOU TO KNOW', 'HIDDEN TRUTH',  # Conspiracy language
            ]
            
            suspicious_count = sum(1 for pattern in suspicious_patterns if pattern.lower() in text_lower)
            
            # Check for credibility indicators (lack of specific details)
            credibility_issues = [
                'somewhere', 'some place', 'unknown location', 'secret location',
                'they say', 'rumors', 'allegedly', 'supposedly', 'reportedly',
                'sources say', 'anonymous sources', 'unconfirmed reports'
            ]
            
            credibility_count = sum(1 for issue in credibility_issues if issue.lower() in text_lower)
            
            # Calculate total fake score
            total_fake_score = fake_indicator_count + suspicious_count + credibility_count
            
            # Determine if this is likely fake news
            is_likely_fake = (
                total_fake_score >= 2 or  # Multiple fake indicators
                fake_indicator_count >= 1 or  # Any direct fake indicator
                suspicious_count >= 2 or  # Multiple suspicious patterns
                (fake_indicator_count >= 1 and suspicious_count >= 1)  # Combination
            )
            
            if is_likely_fake:
                prediction = 'fake'
                confidence = min(0.95, 0.6 + total_fake_score * 0.1)
                modality = 'text_only'
                explanation = f"Detected {total_fake_score} fake news indicators including suspicious language patterns"
            else:
                # Find the most likely disaster type
                max_disaster_score = max(disaster_scores.values())
                max_disaster_type = max(disaster_scores, key=disaster_scores.get)
                
                if max_disaster_score > 0:
                    # Combine text and image analysis
                    if image_scores:
                        # Multimodal analysis
                        image_score = image_scores.get(max_disaster_type, 0)
                        combined_score = max_disaster_score + image_score
                        prediction = f'real_{max_disaster_type}'
                        confidence = min(0.95, 0.6 + combined_score * 0.1)
                        modality = 'multimodal'
                        explanation = f"Detected {max_disaster_type} indicators in both text and image"
                    else:
                        # Text-only analysis
                        prediction = f'real_{max_disaster_type}'
                        confidence = min(0.85, 0.5 + max_disaster_score * 0.1)
                        modality = 'text_only'
                        explanation = f"Detected {max_disaster_type} indicators in text"
                else:
                    # Uncertain case
                    prediction = 'fake'
                    confidence = 0.5
                    modality = 'text_only'
                    explanation = "Unable to determine authenticity with confidence"
            
            # Generate probabilities
            probabilities = [0.1, 0.1, 0.1, 0.1, 0.1]  # Default equal distribution
            if prediction == 'fake':
                probabilities[0] = confidence
                probabilities[1:] = [(1 - confidence) / 4] * 4
            else:
                disaster_idx = self.classes.index(prediction)
                probabilities[disaster_idx] = confidence
                remaining = (1 - confidence) / 4
                for i in range(5):
                    if i != disaster_idx:
                        probabilities[i] = remaining
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities,
                'modality': modality,
                'explanation': explanation
            }
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return {
                'prediction': 'fake',
                'confidence': 0.5,
                'probabilities': [0.5, 0.125, 0.125, 0.125, 0.125],
                'modality': 'fallback',
                'explanation': 'Error occurred during analysis'
            }

# Initialize the simple multimodal classifier
simple_multimodal_classifier = SimpleMultimodalClassifier() 