#!/usr/bin/env python3
"""
Comprehensive Training Pipeline for Fake News Detection During Natural Disasters
Handles multiple datasets: CrisisNLP, CrisisMMD, and FakeNewsNet
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fake_news_detector import FakeNewsDetector
from models.disaster_classifier import DisasterClassifier
from models.multimodal_classifier import MultimodalClassifier
from models.fact_checker import FactChecker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages loading and preprocessing of different datasets"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_fakenewsnet_data(self):
        """Load FakeNewsNet dataset for fake news detection"""
        logger.info("Loading FakeNewsNet dataset...")
        
        fake_data = []
        real_data = []
        
        # Load fake news data
        fake_files = [
            "fakenewsnet_dataset/gossipcop_fake.csv",
            "fakenewsnet_dataset/politifact_fake.csv"
        ]
        
        real_files = [
            "fakenewsnet_dataset/gossipcop_real.csv", 
            "fakenewsnet_dataset/politifact_real.csv"
        ]
        
        for file_path in fake_files:
            try:
                df = pd.read_csv(self.data_dir / file_path)
                df['label'] = 1  # Fake
                fake_data.append(df)
                logger.info(f"Loaded {len(df)} fake samples from {file_path}")
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
                
        for file_path in real_files:
            try:
                df = pd.read_csv(self.data_dir / file_path)
                df['label'] = 0  # Real
                real_data.append(df)
                logger.info(f"Loaded {len(df)} real samples from {file_path}")
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
        
        # Combine datasets
        if fake_data and real_data:
            fake_df = pd.concat(fake_data, ignore_index=True)
            real_df = pd.concat(real_data, ignore_index=True)
            combined_df = pd.concat([fake_df, real_df], ignore_index=True)
            
            logger.info(f"Total FakeNewsNet samples: {len(combined_df)}")
            logger.info(f"Fake samples: {len(fake_df)}, Real samples: {len(real_df)}")
            
            return combined_df
        else:
            logger.error("Could not load FakeNewsNet data")
            return None
    
    def load_crisisnlp_data(self):
        """Load CrisisNLP dataset for disaster classification"""
        logger.info("Loading CrisisNLP dataset...")
        
        crisisnlp_dir = self.data_dir / "crisisnlp_dataset" / "events_set2"
        all_data = []
        
        if crisisnlp_dir.exists():
            for event_dir in crisisnlp_dir.iterdir():
                if event_dir.is_dir() and event_dir.name != "__pycache__":
                    event_name = event_dir.name
                    logger.info(f"Processing event: {event_name}")
                    
                    # Load train, dev, test files
                    for split in ['train', 'dev', 'test']:
                        file_path = event_dir / f"{event_name}_{split}.tsv"
                        if file_path.exists():
                            try:
                                df = pd.read_csv(file_path, sep='\t')
                                df['event'] = event_name
                                df['split'] = split
                                all_data.append(df)
                                logger.info(f"Loaded {len(df)} samples from {file_path}")
                            except Exception as e:
                                logger.warning(f"Could not load {file_path}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total CrisisNLP samples: {len(combined_df)}")
            return combined_df
        else:
            logger.error("Could not load CrisisNLP data")
            return None
    
    def load_crisismmd_data(self):
        """Load CrisisMMD dataset for multimodal classification"""
        logger.info("Loading CrisisMMD dataset...")
        
        crisismmd_dir = self.data_dir / "CrisisMMD" / "crisismmd_datasplit_all"
        datasets = {}
        
        if crisismmd_dir.exists():
            # Load different tasks
            tasks = ['informative', 'humanitarian', 'damage']
            
            for task in tasks:
                train_file = crisismmd_dir / f"task_{task}_text_img_train.tsv"
                dev_file = crisismmd_dir / f"task_{task}_text_img_dev.tsv"
                test_file = crisismmd_dir / f"task_{task}_text_img_test.tsv"
                
                task_data = {}
                
                for split, file_path in [('train', train_file), ('dev', dev_file), ('test', test_file)]:
                    if file_path.exists():
                        try:
                            df = pd.read_csv(file_path, sep='\t')
                            task_data[split] = df
                            logger.info(f"Loaded {len(df)} samples for {task} {split}")
                        except Exception as e:
                            logger.warning(f"Could not load {file_path}: {e}")
                
                if task_data:
                    datasets[task] = task_data
        
        if datasets:
            logger.info(f"Loaded CrisisMMD data for {len(datasets)} tasks")
            return datasets
        else:
            logger.error("Could not load CrisisMMD data")
            return None

class ModelTrainer:
    """Handles training of all models"""
    
    def __init__(self, output_dir="models/saved_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_results = {}
        
    def train_fake_news_detector(self, data):
        """Train the fake news detection model"""
        logger.info("Training Fake News Detection Model...")
        
        try:
            detector = FakeNewsDetector()
            
            # Prepare data
            if data is not None and 'title' in data.columns:
                # Use title as text feature
                texts = data['title'].fillna('').tolist()
                labels = data['label'].tolist()
                
                # Train the model
                detector.train_model(texts, labels)
                
                # Save model (models are saved automatically in train_model)
                self.training_results['fake_news_detector'] = {
                    'status': 'trained',
                    'samples': len(texts),
                    'fake_samples': sum(labels),
                    'real_samples': len(labels) - sum(labels)
                }
                logger.info(f"Fake news detector trained successfully with {len(texts)} samples")
                
                return self.training_results['fake_news_detector']
            else:
                logger.error("Invalid data format for fake news detection training")
                return None
                
        except Exception as e:
            logger.error(f"Error training fake news detector: {e}")
            return None
    
    def train_disaster_classifier(self, data):
        """Train the disaster classification model"""
        logger.info("Training Disaster Classification Model...")
        
        try:
            classifier = DisasterClassifier()
            
            # Prepare data
            if data is not None and 'tweet_text' in data.columns:
                texts = data['tweet_text'].fillna('').tolist()
                events = data['event'].tolist()
                
                # Train the model
                classifier.train_model(texts, events)
                
                # Save model (models are saved automatically in train_model)
                self.training_results['disaster_classifier'] = {
                    'status': 'trained',
                    'samples': len(texts),
                    'unique_events': len(set(events))
                }
                logger.info(f"Disaster classifier trained successfully with {len(texts)} samples")
                
                return self.training_results['disaster_classifier']
            else:
                logger.error("Invalid data format for disaster classification training")
                return None
                
        except Exception as e:
            logger.error(f"Error training disaster classifier: {e}")
            return None
    
    def train_multimodal_classifier(self, data):
        """Train the multimodal classification model"""
        logger.info("Training Multimodal Classification Model...")
        
        try:
            multimodal = MultimodalClassifier()
            
            # Prepare data for different tasks
            if data and 'informative' in data:
                informative_data = data['informative']['train']
                
                if 'tweet_text' in informative_data.columns:
                    texts = informative_data['tweet_text'].fillna('').tolist()
                    labels = informative_data['text_info'].tolist()
                    
                    # For now, we'll just log that multimodal training is available
                    # since the MultimodalClassifier doesn't have a train_model method
                    self.training_results['multimodal_classifier'] = {
                        'status': 'ready_for_training',
                        'samples': len(texts),
                        'note': 'Multimodal classifier requires custom training implementation'
                    }
                    logger.info(f"Multimodal classifier data prepared with {len(texts)} samples")
                    
                    return self.training_results['multimodal_classifier']
                else:
                    logger.error("Invalid data format for multimodal classification training")
                    return None
            else:
                logger.error("No informative task data available for multimodal training")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing multimodal classifier data: {e}")
            return None
    
    def generate_training_report(self):
        """Generate a comprehensive training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(self.training_results.keys()),
            'results': self.training_results,
            'model_paths': {
                'fake_news_detector': str(self.output_dir / "fake_news_detector"),
                'disaster_classifier': str(self.output_dir / "disaster_classifier"),
                'multimodal_classifier': str(self.output_dir / "multimodal_classifier")
            }
        }
        
        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")
        return report

def main():
    """Main training pipeline"""
    logger.info("Starting comprehensive model training pipeline...")
    
    # Initialize dataset manager
    dataset_manager = DatasetManager()
    
    # Load datasets
    fakenews_data = dataset_manager.load_fakenewsnet_data()
    crisisnlp_data = dataset_manager.load_crisisnlp_data()
    crisismmd_data = dataset_manager.load_crisismmd_data()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train models
    results = {}
    
    if fakenews_data is not None:
        results['fake_news'] = trainer.train_fake_news_detector(fakenews_data)
    
    if crisisnlp_data is not None:
        results['disaster_classifier'] = trainer.train_disaster_classifier(crisisnlp_data)
    
    if crisismmd_data is not None:
        results['multimodal'] = trainer.train_multimodal_classifier(crisismmd_data)
    
    # Generate report
    report = trainer.generate_training_report()
    
    logger.info("Training pipeline completed!")
    logger.info(f"Models trained: {len(results)}")
    
    return results

if __name__ == "__main__":
    main() 