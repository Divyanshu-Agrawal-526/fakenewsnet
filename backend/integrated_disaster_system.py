#!/usr/bin/env python3
"""
Integrated Disaster Reporting and Misinformation Detection System
Based on the research paper: "ContCommRTD: A Distributed Content-Based Misinformation-Aware Community Detection System for Real-Time Disaster Reporting"

This integrates:
1. Topic modeling
2. Community detection
3. Geolocation extraction
4. Misinformation filtering
5. Real-time reporting capabilities
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Tuple, Any, Optional

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from topic_modeler import TopicModeler
from community_detector import CommunityDetector
from geolocation_extractor import GeolocationExtractor
from fn_bert_tfidf import FNBertTfidfModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class IntegratedDisasterSystem:
    """
    Complete disaster reporting and misinformation detection system
    """
    
    def __init__(self, 
                 n_topics: int = 6,
                 similarity_threshold: float = 0.3,
                 min_community_size: int = 5):
        """
        Initialize the integrated system
        """
        self.n_topics = n_topics
        self.similarity_threshold = similarity_threshold
        self.min_community_size = min_community_size
        
        # Initialize components
        self.topic_modeler = TopicModeler(n_topics=n_topics)
        self.community_detector = CommunityDetector(
            similarity_threshold=similarity_threshold,
            min_community_size=min_community_size
        )
        self.geolocation_extractor = GeolocationExtractor(use_geocoding=False)  # Disable for demo
        self.misinformation_detector = None  # Will be loaded if available
        
        # System state
        self.topic_results = None
        self.community_results = None
        self.location_results = None
        self.filtered_communities = None
        
        logger.info("Integrated disaster system initialized")
    
    def load_misinformation_detector(self, model_path: str = None):
        """
        Load the misinformation detection model
        """
        try:
            if model_path and os.path.exists(model_path):
                self.misinformation_detector = FNBertTfidfModel.load_model(model_path)
                logger.info("Misinformation detector loaded successfully")
            else:
                logger.warning("Misinformation detector not available")
        except Exception as e:
            logger.error(f"Failed to load misinformation detector: {e}")
    
    def process_dataset(self, 
                       texts: List[str], 
                       user_ids: Optional[List[str]] = None,
                       timestamps: Optional[List[str]] = None,
                       metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process the complete dataset through all system components
        """
        logger.info("Starting integrated dataset processing...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(texts),
            'processing_steps': []
        }
        
        # Step 1: Topic Modeling
        logger.info("Step 1: Topic Modeling")
        try:
            topic_results = self.topic_modeler.fit(texts)
            self.topic_results = topic_results
            results['topic_modeling'] = {
                'status': 'success',
                'n_topics': self.n_topics,
                'evaluation_metrics': topic_results['evaluation_metrics']
            }
            results['processing_steps'].append('topic_modeling')
            logger.info("Topic modeling completed successfully")
        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            results['topic_modeling'] = {'status': 'failed', 'error': str(e)}
        
        # Step 2: Geolocation Extraction
        logger.info("Step 2: Geolocation Extraction")
        try:
            location_results = self.geolocation_extractor.extract_locations_from_dataset(texts)
            self.location_results = location_results
            results['geolocation_extraction'] = {
                'status': 'success',
                'texts_with_locations': sum(1 for data in location_results.values() if data['extracted_locations']),
                'texts_with_coordinates': sum(1 for data in location_results.values() if data['has_coordinates'])
            }
            results['processing_steps'].append('geolocation_extraction')
            logger.info("Geolocation extraction completed successfully")
        except Exception as e:
            logger.error(f"Geolocation extraction failed: {e}")
            results['geolocation_extraction'] = {'status': 'failed', 'error': str(e)}
        
        # Step 3: Community Detection
        logger.info("Step 3: Community Detection")
        try:
            # Build social graph
            social_graph = self.community_detector.build_social_graph(
                texts=texts,
                user_ids=user_ids,
                timestamps=timestamps,
                locations=[data.get('extracted_locations', []) for data in location_results.values()]
            )
            
            # Detect communities
            community_results = self.community_detector.detect_communities(method='content_based')
            self.community_results = community_results
            
            # Evaluate communities
            community_metrics = self.community_detector.evaluate_communities()
            
            results['community_detection'] = {
                'status': 'success',
                'n_communities': len(community_results),
                'evaluation_metrics': community_metrics
            }
            results['processing_steps'].append('community_detection')
            logger.info("Community detection completed successfully")
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            results['community_detection'] = {'status': 'failed', 'error': str(e)}
        
        # Step 4: Misinformation Filtering (if available)
        if self.misinformation_detector:
            logger.info("Step 4: Misinformation Filtering")
            try:
                filtered_results = self._filter_misinformation(texts, community_results)
                self.filtered_communities = filtered_results
                results['misinformation_filtering'] = {
                    'status': 'success',
                    'filtered_communities': len(filtered_results),
                    'removed_samples': len(texts) - sum(len(comm) for comm in filtered_results.values())
                }
                results['processing_steps'].append('misinformation_filtering')
                logger.info("Misinformation filtering completed successfully")
            except Exception as e:
                logger.error(f"Misinformation filtering failed: {e}")
                results['misinformation_filtering'] = {'status': 'failed', 'error': str(e)}
        else:
            logger.info("Step 4: Misinformation Filtering (skipped - detector not available)")
            results['misinformation_filtering'] = {'status': 'skipped', 'reason': 'detector_not_available'}
        
        logger.info("Integrated dataset processing completed")
        return results
    
    def _filter_misinformation(self, 
                              texts: List[str], 
                              communities: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Filter out misinformation from communities
        """
        logger.info("Filtering misinformation from communities...")
        
        filtered_communities = {}
        
        for comm_id, members in communities.items():
            # Get texts for this community
            community_texts = [texts[i] for i in members if i < len(texts)]
            
            # Predict fake/real
            predictions = self.misinformation_detector.predict(community_texts)
            
            # Keep only real news (label 0)
            real_members = [members[i] for i, pred in enumerate(predictions) if pred == 0]
            
            if len(real_members) >= self.min_community_size:
                filtered_communities[comm_id] = real_members
        
        logger.info(f"Misinformation filtering: {len(communities)} -> {len(filtered_communities)} communities")
        return filtered_communities
    
    def get_topic_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected topics
        """
        if not self.topic_results:
            return {'error': 'Topic modeling not completed'}
        
        summary = {
            'n_topics': self.n_topics,
            'topics': {}
        }
        
        for topic_id in range(self.n_topics):
            if topic_id in self.topic_modeler.topic_keywords:
                summary['topics'][topic_id] = {
                    'keywords': self.topic_modeler.topic_keywords[topic_id][:10],
                    'coherence': self.topic_results['evaluation_metrics']['topic_coherence_scores'][topic_id]
                }
        
        return summary
    
    def get_community_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected communities
        """
        if not self.community_results:
            return {'error': 'Community detection not completed'}
        
        summary = {
            'n_communities': len(self.community_results),
            'communities': {}
        }
        
        for comm_id, members in self.community_results.items():
            if comm_id in self.community_detector.community_features:
                features = self.community_detector.community_features[comm_id]
                summary['communities'][comm_id] = {
                    'size': features['size'],
                    'top_keywords': features['top_keywords'][:5],
                    'avg_similarity': features['avg_similarity'],
                    'density': features['density']
                }
        
        return summary
    
    def get_geographic_summary(self) -> Dict[str, Any]:
        """
        Get summary of geographic distribution
        """
        if not self.location_results:
            return {'error': 'Geolocation extraction not completed'}
        
        return self.geolocation_extractor.analyze_geographic_distribution(self.location_results)
    
    def subscribe_to_topic(self, 
                          topic_id: int, 
                          location_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Subscribe to updates for a specific topic and optionally location
        """
        if not self.topic_results or not self.community_results:
            return [{'error': 'System not fully processed yet'}]
        
        # Get texts for this topic
        topic_distributions = self.topic_results['topic_distributions']
        topic_members = []
        
        for i, distribution in enumerate(topic_distributions):
            dominant_topic = np.argmax(distribution)
            if dominant_topic == topic_id:
                topic_members.append(i)
        
        # Filter by location if specified
        if location_filter and self.location_results:
            location_filtered = []
            for idx in topic_members:
                if idx in self.location_results:
                    locations = self.location_results[idx]['extracted_locations']
                    if any(location_filter.lower() in loc.lower() for loc in locations):
                        location_filtered.append(idx)
            topic_members = location_filtered
        
        # Get community information for these members
        results = []
        for idx in topic_members:
            result = {
                'text_id': idx,
                'text': self.topic_results['processed_texts'][idx] if idx < len(self.topic_results['processed_texts']) else f"Text {idx}",
                'topic': topic_id,
                'communities': []
            }
            
            # Find which communities this text belongs to
            for comm_id, members in self.community_results.items():
                if idx in members:
                    if comm_id in self.community_detector.community_features:
                        features = self.community_detector.community_features[comm_id]
                        result['communities'].append({
                            'community_id': comm_id,
                            'size': features['size'],
                            'top_keywords': features['top_keywords'][:5]
                        })
            
            results.append(result)
        
        return results
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive system report
        """
        logger.info("Generating comprehensive system report...")
        
        report = []
        report.append("=" * 80)
        report.append("INTEGRATED DISASTER REPORTING SYSTEM - COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Topic modeling summary
        topic_summary = self.get_topic_summary()
        if 'error' not in topic_summary:
            report.append("TOPIC MODELING SUMMARY:")
            report.append("-" * 30)
            report.append(f"Number of topics: {topic_summary['n_topics']}")
            for topic_id, topic_info in topic_summary['topics'].items():
                report.append(f"Topic {topic_id}: {', '.join(topic_info['keywords'][:5])}")
                report.append(f"  Coherence: {topic_info['coherence']:.4f}")
            report.append("")
        
        # Community detection summary
        community_summary = self.get_community_summary()
        if 'error' not in community_summary:
            report.append("COMMUNITY DETECTION SUMMARY:")
            report.append("-" * 30)
            report.append(f"Number of communities: {community_summary['n_communities']}")
            for comm_id, comm_info in community_summary['communities'].items():
                report.append(f"Community {comm_id}: {comm_info['size']} members")
                report.append(f"  Keywords: {', '.join(comm_info['top_keywords'])}")
                report.append(f"  Similarity: {comm_info['avg_similarity']:.4f}")
            report.append("")
        
        # Geographic summary
        geo_summary = self.get_geographic_summary()
        if 'error' not in geo_summary:
            report.append("GEOGRAPHIC DISTRIBUTION SUMMARY:")
            report.append("-" * 30)
            report.append(f"Texts with locations: {geo_summary['texts_with_locations']}")
            report.append(f"Texts with coordinates: {geo_summary['texts_with_coordinates']}")
            report.append(f"Location extraction rate: {geo_summary['location_extraction_rate']:.2%}")
            report.append(f"Geocoding success rate: {geo_summary['geocoding_success_rate']:.2%}")
            report.append("")
        
        # Misinformation filtering summary
        if self.filtered_communities:
            report.append("MISINFORMATION FILTERING SUMMARY:")
            report.append("-" * 30)
            report.append(f"Original communities: {len(self.community_results)}")
            report.append(f"Filtered communities: {len(self.filtered_communities)}")
            report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def save_system_state(self, output_path: str):
        """
        Save complete system state
        """
        logger.info("Saving system state...")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save topic modeler
        if self.topic_modeler:
            topic_path = os.path.join(output_path, 'topic_modeler')
            self.topic_modeler.save_model(topic_path)
        
        # Save community detector results
        if self.community_detector:
            community_path = os.path.join(output_path, 'community_detector')
            self.community_detector.save_results(community_path)
        
        # Save geolocation results
        if self.location_results:
            location_path = os.path.join(output_path, 'geolocation')
            self.geolocation_extractor.save_results(self.location_results, location_path)
        
        # Save system configuration
        config = {
            'n_topics': self.n_topics,
            'similarity_threshold': self.similarity_threshold,
            'min_community_size': self.min_community_size,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_path, 'system_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"System state saved to {output_path}")

def main():
    """
    Main function to demonstrate the integrated system
    """
    logger.info("🚀 Starting Integrated Disaster Reporting System...")
    
    # Create integrated system
    system = IntegratedDisasterSystem(
        n_topics=6,
        similarity_threshold=0.3,
        min_community_size=5
    )
    
    # Load data
    logger.info("Loading dataset...")
    try:
        # Load FakeNewsNet data
        gossipcop_fake = pd.read_csv("data/fakenewsnet_dataset/gossipcop_fake.csv")
        politifact_fake = pd.read_csv("data/fakenewsnet_dataset/politifact_fake.csv")
        gossipcop_real = pd.read_csv("data/fakenewsnet_dataset/gossipcop_real.csv")
        politifact_real = pd.read_csv("data/fakenewsnet_dataset/politifact_real.csv")
        
        # Add labels
        gossipcop_fake['label'] = 1
        politifact_fake['label'] = 1
        gossipcop_real['label'] = 0
        politifact_real['label'] = 0
        
        # Combine datasets
        all_data = pd.concat([gossipcop_fake, politifact_fake, gossipcop_real, politifact_real], ignore_index=True)
        all_data = all_data.dropna(subset=['title'])
        all_data = all_data[all_data['title'].str.len() > 10]
        
        # Use subset for demonstration
        sample_size = min(5000, len(all_data))
        sample_data = all_data.sample(n=sample_size, random_state=42)
        
        texts = sample_data['title'].fillna('').astype(str).tolist()
        labels = sample_data['label'].astype(int).tolist()
        
        logger.info(f"Loaded {len(texts)} texts for processing")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Process dataset
    logger.info("Processing dataset through integrated system...")
    results = system.process_dataset(texts)
    
    # Generate report
    logger.info("Generating comprehensive report...")
    report = system.generate_report()
    print(report)
    
    # Save system state
    output_dir = "models/saved_models/integrated_system"
    system.save_system_state(output_dir)
    
    # Demonstrate topic subscription
    logger.info("Demonstrating topic subscription...")
    topic_updates = system.subscribe_to_topic(topic_id=0, location_filter="california")
    logger.info(f"Found {len(topic_updates)} updates for topic 0 in California")
    
    logger.info("✅ Integrated system demonstration completed!")

if __name__ == "__main__":
    main()
