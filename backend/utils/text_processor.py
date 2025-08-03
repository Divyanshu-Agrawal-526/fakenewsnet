import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
from typing import List, Dict, Any

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Custom stop words for disaster context
        self.disaster_stop_words = {
            'rt', 'via', 'new', 'time', 'today', 'yesterday', 'tomorrow',
            'breaking', 'update', 'latest', 'news', 'report', 'says', 'said'
        }
        
        # Disaster-related words to keep
        self.disaster_keywords = {
            'fire', 'flood', 'hurricane', 'earthquake', 'tornado', 'tsunami',
            'wildfire', 'storm', 'disaster', 'emergency', 'evacuation',
            'damage', 'destruction', 'rescue', 'help', 'urgent', 'warning',
            'alert', 'dangerous', 'severe', 'critical'
        }
    
    def preprocess(self, text: str) -> str:
        """Main preprocessing function"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = self.remove_urls(text)
            
            # Remove user mentions
            text = self.remove_mentions(text)
            
            # Remove hashtags but keep the text
            text = self.process_hashtags(text)
            
            # Remove special characters but keep important ones
            text = self.clean_special_chars(text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stop words but keep disaster keywords
            tokens = self.remove_stop_words(tokens)
            
            # Lemmatize
            tokens = self.lemmatize_tokens(tokens)
            
            # Join tokens back into text
            processed_text = ' '.join(tokens)
            
            # Remove extra whitespace
            processed_text = ' '.join(processed_text.split())
            
            return processed_text
            
        except Exception as e:
            logging.error(f"Error in text preprocessing: {e}")
            return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_mentions(self, text: str) -> str:
        """Remove user mentions from text"""
        mention_pattern = r'@\w+'
        return re.sub(mention_pattern, '', text)
    
    def process_hashtags(self, text: str) -> str:
        """Process hashtags - remove # but keep the text"""
        hashtag_pattern = r'#(\w+)'
        return re.sub(hashtag_pattern, r'\1', text)
    
    def clean_special_chars(self, text: str) -> str:
        """Clean special characters but keep important ones"""
        # Keep alphanumeric, spaces, and some punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[\.\,\!\?]{2,}', '.', text)
        
        return text
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove stop words but keep disaster keywords"""
        filtered_tokens = []
        
        for token in tokens:
            # Keep disaster keywords
            if token.lower() in self.disaster_keywords:
                filtered_tokens.append(token)
            # Remove stop words and custom stop words
            elif token.lower() not in self.stop_words and token.lower() not in self.disaster_stop_words:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        processed_text = self.preprocess(text)
        tokens = word_tokenize(processed_text)
        
        # Filter out short tokens and keep meaningful ones
        keywords = [token for token in tokens if len(token) > 2]
        
        return keywords[:10]  # Return top 10 keywords
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            'locations': [],
            'organizations': [],
            'disasters': []
        }
        
        # Simple pattern matching for entities
        # In a real implementation, you'd use NER models
        
        # Location patterns
        location_patterns = [
            r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'at\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'near\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['locations'].extend(matches)
        
        # Organization patterns
        org_patterns = [
            r'([A-Z][a-z]+\s+(?:Department|Agency|Service|Department))',
            r'([A-Z][a-z]+\s+(?:Police|Fire|Emergency))'
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['organizations'].extend(matches)
        
        # Disaster patterns
        disaster_patterns = [
            r'(wildfire|fire|blaze)',
            r'(flood|flooding|water)',
            r'(hurricane|storm|cyclone)',
            r'(earthquake|quake|tremor)'
        ]
        
        for pattern in disaster_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['disasters'].extend(matches)
        
        return entities
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        try:
            sentences = re.split(r'[.!?]+', text)
            words = text.split()
            syllables = self.count_syllables(text)
            
            if len(sentences) == 0 or len(words) == 0:
                return {
                    'flesch_reading_ease': 0.0,
                    'flesch_kincaid_grade': 0.0,
                    'avg_sentence_length': 0.0,
                    'avg_word_length': 0.0
                }
            
            # Flesch Reading Ease
            flesch_ease = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
            
            # Flesch-Kincaid Grade Level
            flesch_grade = (0.39 * (len(words) / len(sentences))) + (11.8 * (syllables / len(words))) - 15.59
            
            return {
                'flesch_reading_ease': max(0.0, min(100.0, flesch_ease)),
                'flesch_kincaid_grade': max(0.0, flesch_grade),
                'avg_sentence_length': len(words) / len(sentences),
                'avg_word_length': sum(len(word) for word in words) / len(words)
            }
            
        except Exception as e:
            logging.error(f"Error calculating readability: {e}")
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'avg_sentence_length': 0.0,
                'avg_word_length': 0.0
            }
    
    def count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified)"""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(1, count)
    
    def detect_language(self, text: str) -> str:
        """Detect language of text (simplified)"""
        # Simple language detection based on common words
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        spanish_words = {'el', 'la', 'los', 'las', 'y', 'o', 'pero', 'en', 'con', 'por', 'para'}
        
        words = set(text.lower().split())
        
        english_count = len(words.intersection(english_words))
        spanish_count = len(words.intersection(spanish_words))
        
        if english_count > spanish_count:
            return 'en'
        elif spanish_count > english_count:
            return 'es'
        else:
            return 'en'  # Default to English
    
    def extract_sentiment(self, text: str) -> Dict[str, float]:
        """Extract sentiment from text"""
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logging.error(f"Error extracting sentiment: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0
            } 