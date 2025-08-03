#!/usr/bin/env python3
"""
Test script for multimodal classification
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.multimodal_classifier_simple import simple_multimodal_classifier

def test_classification():
    """Test the classification with various tweets"""
    
    test_tweets = [
        # Fake news examples
        {
            'text': 'Major wildfire spreading rapidly in California. Evacuations ordered for multiple communities. Stay safe everyone! #CaliforniaWildfire',
            'expected': 'fake',
            'description': 'Fake tweet with disaster keywords but no fake indicators'
        },
        {
            'text': 'ALIENS CAUSED THE EARTHQUAKE! Government hiding the truth! Share this everywhere! #Conspiracy',
            'expected': 'fake',
            'description': 'Obvious fake news with conspiracy language'
        },
        {
            'text': 'BREAKING!!! SHOCKING wildfire footage you won\'t believe! MUST SHARE NOW! #Viral',
            'expected': 'fake',
            'description': 'Fake news with sensationalist language'
        },
        {
            'text': 'They don\'t want you to know about this earthquake! Hidden truth revealed! #WakeUp',
            'expected': 'fake',
            'description': 'Fake news with conspiracy language'
        },
        
        # Real disaster examples
        {
            'text': 'Official evacuation order issued for residents in Northern California due to rapidly spreading wildfire. Follow emergency instructions.',
            'expected': 'real_wildfire',
            'description': 'Real disaster with official language'
        },
        {
            'text': 'Flash flood warning issued for downtown Houston. Roads are already flooding. Avoid travel if possible. #HoustonFlood',
            'expected': 'real_flood',
            'description': 'Real disaster with specific details'
        },
        {
            'text': 'Hurricane Maria makes landfall in Puerto Rico. Wind speeds reaching 155 mph. Stay indoors and follow safety protocols.',
            'expected': 'real_hurricane',
            'description': 'Real disaster with specific information'
        },
        {
            'text': 'Magnitude 6.2 earthquake strikes near Los Angeles. USGS confirms epicenter location. Minor damage reported.',
            'expected': 'real_earthquake',
            'description': 'Real disaster with official sources'
        }
    ]
    
    print("Testing Multimodal Classification System")
    print("=" * 50)
    
    for i, test_case in enumerate(test_tweets, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Tweet: {test_case['text']}")
        
        # Run classification
        result = simple_multimodal_classifier.classify(test_case['text'])
        
        print(f"Prediction: {result['prediction']}")
        print(f"Expected: {test_case['expected']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Modality: {result['modality']}")
        print(f"Explanation: {result['explanation']}")
        
        # Check if prediction matches expected
        if result['prediction'] == test_case['expected']:
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
        
        print("-" * 50)

if __name__ == "__main__":
    test_classification() 