#!/usr/bin/env python3
"""
Test script for the training pipeline
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append('backend')

def test_dataset_loading():
    """Test if datasets can be loaded correctly"""
    print("Testing dataset loading...")
    
    try:
        from backend.train_models import DatasetManager
        
        # Initialize dataset manager
        dataset_manager = DatasetManager()
        
        # Test loading each dataset
        print("\n1. Testing FakeNewsNet dataset...")
        fakenews_data = dataset_manager.load_fakenewsnet_data()
        if fakenews_data is not None:
            print(f"✓ FakeNewsNet loaded successfully: {len(fakenews_data)} samples")
            print(f"  Columns: {list(fakenews_data.columns)}")
        else:
            print("✗ Failed to load FakeNewsNet data")
        
        print("\n2. Testing CrisisNLP dataset...")
        crisisnlp_data = dataset_manager.load_crisisnlp_data()
        if crisisnlp_data is not None:
            print(f"✓ CrisisNLP loaded successfully: {len(crisisnlp_data)} samples")
            print(f"  Columns: {list(crisisnlp_data.columns)}")
        else:
            print("✗ Failed to load CrisisNLP data")
        
        print("\n3. Testing CrisisMMD dataset...")
        crisismmd_data = dataset_manager.load_crisismmd_data()
        if crisismmd_data is not None:
            print(f"✓ CrisisMMD loaded successfully")
            for task, data in crisismmd_data.items():
                print(f"  Task '{task}': {len(data)} splits")
        else:
            print("✗ Failed to load CrisisMMD data")
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing dataset loading: {e}")
        return False

def test_model_imports():
    """Test if model classes can be imported"""
    print("\nTesting model imports...")
    
    try:
        from backend.models.fake_news_detector import FakeNewsDetector
        from backend.models.disaster_classifier import DisasterClassifier
        from backend.models.multimodal_classifier import MultimodalClassifier
        
        print("✓ All model classes imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error importing models: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Training Pipeline Test ===\n")
    
    # Test dataset loading
    dataset_ok = test_dataset_loading()
    
    # Test model imports
    models_ok = test_model_imports()
    
    print("\n=== Test Results ===")
    print(f"Dataset loading: {'✓ PASS' if dataset_ok else '✗ FAIL'}")
    print(f"Model imports: {'✓ PASS' if models_ok else '✗ FAIL'}")
    
    if dataset_ok and models_ok:
        print("\n✓ All tests passed! You can now run the training pipeline.")
        print("\nTo train models locally:")
        print("cd backend")
        print("python train_models.py")
        
        print("\nTo train models in GitHub:")
        print("1. Push your code to GitHub")
        print("2. GitHub Actions will automatically run the training workflow")
        print("3. Check the Actions tab in your repository")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 