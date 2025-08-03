import requests
import json
import logging
from typing import Dict, Any, List
import re
from datetime import datetime
import os
from urllib.parse import quote_plus
import time

class FactChecker:
    def __init__(self):
        self.api_keys = {
            'newsapi': os.getenv('NEWS_API_KEY', ''),
            'factcheck': os.getenv('FACTCHECK_API_KEY', ''),
            'google': os.getenv('GOOGLE_API_KEY', '')
        }
        
        self.verified_sources = [
            'reuters.com', 'ap.org', 'bbc.com', 'cnn.com', 'nbcnews.com',
            'abcnews.go.com', 'cbsnews.com', 'foxnews.com', 'usatoday.com',
            'nytimes.com', 'washingtonpost.com', 'latimes.com', 'chicagotribune.com'
        ]
        
        self.disaster_keywords = [
            'fire', 'flood', 'hurricane', 'earthquake', 'tornado', 'tsunami',
            'wildfire', 'storm', 'disaster', 'emergency', 'evacuation'
        ]
        
        self.is_model_loaded = True  # Fact checker doesn't need model loading
    
    def verify(self, text: str, location: str = None) -> Dict[str, Any]:
        """
        Verify the authenticity of a tweet through multiple fact-checking methods
        """
        try:
            results = {
                'verified': False,
                'confidence': 0.0,
                'sources': [],
                'explanations': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Extract key information from text
            extracted_info = self.extract_information(text)
            
            # Method 1: News API verification
            news_results = self.verify_with_news_api(extracted_info, location)
            results['sources'].extend(news_results['sources'])
            results['explanations'].extend(news_results['explanations'])
            
            # Method 2: Keyword-based verification
            keyword_results = self.verify_with_keywords(text, extracted_info)
            results['explanations'].extend(keyword_results['explanations'])
            
            # Method 3: Location-based verification
            if location:
                location_results = self.verify_with_location(location, extracted_info)
                results['sources'].extend(location_results['sources'])
                results['explanations'].extend(location_results['explanations'])
            
            # Method 4: Temporal verification
            temporal_results = self.verify_temporal_consistency(extracted_info)
            results['explanations'].extend(temporal_results['explanations'])
            
            # Calculate overall confidence
            results['confidence'] = self.calculate_confidence(results)
            results['verified'] = results['confidence'] > 0.6
            
            return results
            
        except Exception as e:
            logging.error(f"Error in fact checking: {e}")
            return {
                'verified': False,
                'confidence': 0.0,
                'sources': [],
                'explanations': ['Error occurred during fact checking'],
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def extract_information(self, text: str) -> Dict[str, Any]:
        """Extract key information from tweet text"""
        info = {
            'disaster_type': None,
            'location': None,
            'time_indicators': [],
            'severity_indicators': [],
            'action_indicators': []
        }
        
        text_lower = text.lower()
        
        # Extract disaster type
        for keyword in self.disaster_keywords:
            if keyword in text_lower:
                info['disaster_type'] = keyword
                break
        
        # Extract location indicators
        location_patterns = [
            r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'at\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'near\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                info['location'] = matches[0]
                break
        
        # Extract time indicators
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)',
            r'(today|yesterday|tomorrow)',
            r'(\d{1,2}\s+(?:hours?|days?)\s+ago)'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            info['time_indicators'].extend(matches)
        
        # Extract severity indicators
        severity_words = ['severe', 'critical', 'dangerous', 'urgent', 'emergency', 'evacuation']
        for word in severity_words:
            if word in text_lower:
                info['severity_indicators'].append(word)
        
        # Extract action indicators
        action_words = ['evacuate', 'warning', 'alert', 'help', 'rescue', 'damage']
        for word in action_words:
            if word in text_lower:
                info['action_indicators'].append(word)
        
        return info
    
    def verify_with_news_api(self, extracted_info: Dict[str, Any], location: str = None) -> Dict[str, Any]:
        """Verify using News API"""
        results = {
            'sources': [],
            'explanations': []
        }
        
        if not self.api_keys['newsapi']:
            results['explanations'].append('News API key not available')
            return results
        
        try:
            # Build search query
            query_parts = []
            if extracted_info['disaster_type']:
                query_parts.append(extracted_info['disaster_type'])
            if location:
                query_parts.append(location)
            elif extracted_info['location']:
                query_parts.append(extracted_info['location'])
            
            if not query_parts:
                results['explanations'].append('Insufficient information for news verification')
                return results
            
            query = ' '.join(query_parts)
            
            # Search for recent news articles
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'apiKey': self.api_keys['newsapi'],
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                # Check for verified sources
                verified_articles = []
                for article in articles:
                    source_domain = self.extract_domain(article.get('url', ''))
                    if any(verified in source_domain for verified in self.verified_sources):
                        verified_articles.append(article)
                
                if verified_articles:
                    results['sources'].extend([
                        {
                            'title': article.get('title', ''),
                            'url': article.get('url', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'published_at': article.get('publishedAt', '')
                        }
                        for article in verified_articles[:3]  # Top 3 articles
                    ])
                    results['explanations'].append(f'Found {len(verified_articles)} verified news articles about this incident')
                else:
                    results['explanations'].append('No verified news sources found for this incident')
            else:
                results['explanations'].append('Unable to access news API')
                
        except Exception as e:
            logging.error(f"Error in news API verification: {e}")
            results['explanations'].append('Error accessing news sources')
        
        return results
    
    def verify_with_keywords(self, text: str, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Verify using keyword analysis"""
        results = {
            'explanations': []
        }
        
        # Check for disaster-specific keywords
        disaster_count = sum(1 for keyword in self.disaster_keywords if keyword in text.lower())
        
        if disaster_count > 0:
            results['explanations'].append(f'Contains {disaster_count} disaster-related keywords')
        else:
            results['explanations'].append('No disaster-related keywords found')
        
        # Check for credibility indicators
        credibility_indicators = ['official', 'confirmed', 'verified', 'authority', 'emergency']
        credibility_count = sum(1 for indicator in credibility_indicators if indicator in text.lower())
        
        if credibility_count > 0:
            results['explanations'].append(f'Contains {credibility_count} credibility indicators')
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'click here',
            r'follow me',
            r'retweet',
            r'like and share',
            r'viral'
        ]
        
        suspicious_count = 0
        for pattern in suspicious_patterns:
            if re.search(pattern, text.lower()):
                suspicious_count += 1
        
        if suspicious_count > 0:
            results['explanations'].append(f'Contains {suspicious_count} suspicious patterns')
        
        return results
    
    def verify_with_location(self, location: str, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Verify using location-based checks"""
        results = {
            'sources': [],
            'explanations': []
        }
        
        try:
            # Check if location is mentioned in the tweet
            if extracted_info['location']:
                results['explanations'].append(f'Location mentioned: {extracted_info["location"]}')
            
            # Check for location consistency
            if location and extracted_info['location']:
                if location.lower() in extracted_info['location'].lower() or extracted_info['location'].lower() in location.lower():
                    results['explanations'].append('Location information is consistent')
                else:
                    results['explanations'].append('Location information may be inconsistent')
            
            # Add location-based news search
            if location:
                location_results = self.verify_with_news_api(extracted_info, location)
                results['sources'].extend(location_results['sources'])
                results['explanations'].extend(location_results['explanations'])
                
        except Exception as e:
            logging.error(f"Error in location verification: {e}")
            results['explanations'].append('Error in location verification')
        
        return results
    
    def verify_temporal_consistency(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Verify temporal consistency of the information"""
        results = {
            'explanations': []
        }
        
        # Check for time indicators
        if extracted_info['time_indicators']:
            results['explanations'].append(f'Contains {len(extracted_info["time_indicators"])} time indicators')
        else:
            results['explanations'].append('No specific time indicators found')
        
        # Check for recent disaster patterns
        current_month = datetime.now().month
        
        # Seasonal disaster patterns
        seasonal_patterns = {
            'wildfire': [6, 7, 8, 9],  # Summer months
            'hurricane': [6, 7, 8, 9, 10],  # Hurricane season
            'flood': [3, 4, 5, 6, 7, 8],  # Spring/summer rains
            'earthquake': list(range(1, 13))  # Year-round
        }
        
        if extracted_info['disaster_type'] in seasonal_patterns:
            if current_month in seasonal_patterns[extracted_info['disaster_type']]:
                results['explanations'].append('Disaster type is consistent with current season')
            else:
                results['explanations'].append('Disaster type may be inconsistent with current season')
        
        return results
    
    def calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        confidence = 0.0
        
        # Base confidence
        base_confidence = 0.3
        
        # Source verification bonus
        source_count = len(results['sources'])
        if source_count > 0:
            base_confidence += min(source_count * 0.2, 0.4)
        
        # Explanation analysis
        positive_indicators = 0
        negative_indicators = 0
        
        for explanation in results['explanations']:
            if any(word in explanation.lower() for word in ['verified', 'confirmed', 'consistent', 'found']):
                positive_indicators += 1
            elif any(word in explanation.lower() for word in ['suspicious', 'inconsistent', 'error', 'unable']):
                negative_indicators += 1
        
        confidence = base_confidence + (positive_indicators * 0.1) - (negative_indicators * 0.1)
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ''
    
    def is_loaded(self) -> bool:
        """Check if fact checker is loaded"""
        return self.is_model_loaded 