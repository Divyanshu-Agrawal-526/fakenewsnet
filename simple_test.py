#!/usr/bin/env python3
"""
Simple test for the improved classification logic
"""

def test_fake_news_detection():
    """Test the fake news detection logic"""
    
    # Test cases
    test_tweets = [
        # The problematic tweet that was being misclassified
        {
            'text': 'Major wildfire spreading rapidly in California. Evacuations ordered for multiple communities. Stay safe everyone! #CaliforniaWildfire',
            'expected': 'fake',
            'description': 'Fake tweet with disaster keywords'
        },
        # Obvious fake news
        {
            'text': 'ALIENS CAUSED THE EARTHQUAKE! Government hiding the truth! Share this everywhere! #Conspiracy',
            'expected': 'fake',
            'description': 'Obvious fake news with conspiracy language'
        },
        # Sensationalist fake news
        {
            'text': 'BREAKING!!! SHOCKING wildfire footage you won\'t believe! MUST SHARE NOW! #Viral',
            'expected': 'fake',
            'description': 'Fake news with sensationalist language'
        },
        # Real disaster example
        {
            'text': 'Official evacuation order issued for residents in Northern California due to rapidly spreading wildfire. Follow emergency instructions.',
            'expected': 'real_wildfire',
            'description': 'Real disaster with official language'
        }
    ]
    
    print("Testing Improved Fake News Detection")
    print("=" * 50)
    
    for i, test_case in enumerate(test_tweets, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Tweet: {test_case['text']}")
        
        # Simulate the classification logic
        result = simulate_classification(test_case['text'])
        
        print(f"Prediction: {result['prediction']}")
        print(f"Expected: {test_case['expected']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Explanation: {result['explanation']}")
        
        # Check if prediction matches expected
        if result['prediction'] == test_case['expected']:
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
        
        print("-" * 50)

def simulate_classification(text):
    """Simulate the classification logic"""
    text_lower = text.lower()
    
    # Fake news indicators
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
    
    # Additional checks
    # Check for excessive capitalization
    caps_ratio = sum(1 for char in text if char.isupper()) / len(text) if text else 0
    if caps_ratio > 0.3:  # More than 30% caps
        total_fake_score += 1
    
    # Check for excessive punctuation
    exclamation_count = text.count('!')
    if exclamation_count > 2:  # More than 2 exclamation marks
        total_fake_score += 1
    
    # Check for sensationalist language
    sensational_words = ['BREAKING', 'SHOCKING', 'INCREDIBLE', 'AMAZING', 'UNBELIEVABLE']
    if any(word.lower() in text_lower for word in sensational_words):
        total_fake_score += 1
    
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
        explanation = f"Detected {total_fake_score} fake news indicators including suspicious language patterns"
    else:
        # Check for disaster keywords
        disaster_keywords = {
            'wildfire': ['fire', 'wildfire', 'burning', 'flame', 'smoke', 'blaze', 'forest fire'],
            'flood': ['flood', 'water', 'rain', 'overflow', 'drowning', 'submerged', 'water level'],
            'hurricane': ['hurricane', 'storm', 'wind', 'tropical', 'cyclone', 'typhoon', 'gale'],
            'earthquake': ['earthquake', 'quake', 'shaking', 'tremor', 'seismic', 'magnitude', 'epicenter']
        }
        
        disaster_scores = {}
        for disaster_type, keywords in disaster_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            disaster_scores[disaster_type] = score
        
        max_disaster_score = max(disaster_scores.values())
        max_disaster_type = max(disaster_scores, key=disaster_scores.get)
        
        if max_disaster_score > 0:
            prediction = f'real_{max_disaster_type}'
            confidence = min(0.85, 0.5 + max_disaster_score * 0.1)
            explanation = f"Detected {max_disaster_type} indicators in text"
        else:
            prediction = 'fake'
            confidence = 0.5
            explanation = "Unable to determine authenticity with confidence"
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'explanation': explanation
    }

if __name__ == "__main__":
    test_fake_news_detection() 