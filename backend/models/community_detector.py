#!/usr/bin/env python3
"""
Community Detection System for Disaster Reporting
Based on the research paper: "ContCommRTD: A Distributed Content-Based Misinformation-Aware Community Detection System for Real-Time Disaster Reporting"

This implements:
1. Social graph construction
2. Content-based community detection
3. Geolocation-aware clustering
4. Community quality evaluation
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import pickle
import os
from typing import List, Dict, Tuple, Any, Optional
import re
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)

class CommunityDetector:
    """
    Community detection system for disaster-related social media content
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.3,
                 min_community_size: int = 5,
                 max_communities: int = 20):
        self.similarity_threshold = similarity_threshold
        self.min_community_size = min_community_size
        self.max_communities = max_communities
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.social_graph = None
        self.communities = {}
        self.community_features = {}
        
        logger.info("Community detector initialized")
    
    def build_social_graph(self, 
                          texts: List[str], 
                          user_ids: Optional[List[str]] = None,
                          timestamps: Optional[List[str]] = None,
                          locations: Optional[List[str]] = None,
                          social_relations: Optional[List[Tuple[int, int]]] = None) -> nx.Graph:
        """
        Build social graph from content and social relations
        """
        logger.info("Building social graph...")
        
        # Create graph
        self.social_graph = nx.Graph()
        
        # Add nodes (tweets/posts)
        for i, text in enumerate(texts):
            node_data = {
                'text': text,
                'index': i,
                'user_id': user_ids[i] if user_ids else f"user_{i}",
                'timestamp': timestamps[i] if timestamps else None,
                'location': locations[i] if locations else None
            }
            self.social_graph.add_node(i, **node_data)
        
        # Add edges based on content similarity
        logger.info("Adding content similarity edges...")
        self._add_content_similarity_edges(texts)
        
        # Add social relation edges if provided
        if social_relations:
            logger.info("Adding social relation edges...")
            for source, target in social_relations:
                if source < len(texts) and target < len(texts):
                    self.social_graph.add_edge(source, target, relation_type='social')
        
        logger.info(f"Social graph built with {self.social_graph.number_of_nodes()} nodes and {self.social_graph.number_of_edges()} edges")
        return self.social_graph
    
    def _add_content_similarity_edges(self, texts: List[str]):
        """
        Add edges based on content similarity
        """
        # Vectorize texts
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Add edges for similar content
        edges_added = 0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = similarity_matrix[i, j]
                if similarity > self.similarity_threshold:
                    self.social_graph.add_edge(i, j, 
                                            similarity=similarity,
                                            relation_type='content')
                    edges_added += 1
        
        logger.info(f"Added {edges_added} content similarity edges")
    
    def detect_communities(self, method: str = 'content_based') -> Dict[int, List[int]]:
        """
        Detect communities using specified method
        """
        if self.social_graph is None:
            logger.error("Social graph not built yet!")
            return {}
        
        logger.info(f"Detecting communities using {method} method...")
        
        if method == 'content_based':
            communities = self._content_based_community_detection()
        elif method == 'graph_based':
            communities = self._graph_based_community_detection()
        elif method == 'hybrid':
            communities = self._hybrid_community_detection()
        else:
            logger.error(f"Unknown method: {method}")
            return {}
        
        # Filter communities by size
        filtered_communities = {}
        for comm_id, members in communities.items():
            if len(members) >= self.min_community_size:
                filtered_communities[comm_id] = members
        
        self.communities = filtered_communities
        
        # Extract community features
        self._extract_community_features()
        
        logger.info(f"Detected {len(self.communities)} communities")
        return self.communities
    
    def _content_based_community_detection(self) -> Dict[int, List[int]]:
        """
        Detect communities based on content similarity
        """
        # Get TF-IDF features
        texts = [self.social_graph.nodes[node]['text'] for node in self.social_graph.nodes()]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Use K-means clustering
        n_clusters = min(self.max_communities, max(1, len(texts) // max(1, self.min_community_size)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Group by cluster
        communities = defaultdict(list)
        for node_idx, cluster_id in enumerate(cluster_labels):
            communities[cluster_id].append(node_idx)
        
        return dict(communities)
    
    def _graph_based_community_detection(self) -> Dict[int, List[int]]:
        """
        Detect communities using graph algorithms
        """
        # Use Louvain method for community detection
        communities = nx.community.louvain_communities(self.social_graph)
        
        # Convert to our format
        result = {}
        for i, community in enumerate(communities):
            if len(community) >= self.min_community_size:
                result[i] = list(community)
        
        return result
    
    def _hybrid_community_detection(self) -> Dict[int, List[int]]:
        """
        Combine content and graph-based methods
        """
        # Get content-based communities
        content_communities = self._content_based_community_detection()
        
        # Get graph-based communities
        graph_communities = self._graph_based_community_detection()
        
        # Merge communities that have significant overlap
        merged_communities = self._merge_overlapping_communities(
            content_communities, graph_communities
        )
        
        return merged_communities
    
    def _merge_overlapping_communities(self, 
                                     comm1: Dict[int, List[int]], 
                                     comm2: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Merge communities with significant overlap
        """
        merged = {}
        used_nodes = set()
        
        # Start with first set of communities
        for comm_id, members in comm1.items():
            if not any(node in used_nodes for node in members):
                merged[comm_id] = members
                used_nodes.update(members)
        
        # Add non-overlapping communities from second set
        for comm_id, members in comm2.items():
            overlap = len(set(members) & used_nodes)
            if overlap / len(members) < 0.3:  # Less than 30% overlap
                merged[f"graph_{comm_id}"] = members
                used_nodes.update(members)
        
        return merged
    
    def _extract_community_features(self):
        """
        Extract features for each community
        """
        for comm_id, members in self.communities.items():
            features = {
                'size': len(members),
                'texts': [self.social_graph.nodes[node]['text'] for node in members],
                'users': [self.social_graph.nodes[node]['user_id'] for node in members],
                'locations': [self.social_graph.nodes[node]['location'] for node in members if self.social_graph.nodes[node]['location']],
                'avg_similarity': self._calculate_community_similarity(members),
                'density': self._calculate_community_density(members)
            }
            
            # Extract common keywords
            all_texts = ' '.join(features['texts'])
            words = re.findall(r'\b\w+\b', all_texts.lower())
            word_counts = Counter(words)
            features['top_keywords'] = [word for word, count in word_counts.most_common(10)]
            
            self.community_features[comm_id] = features
    
    def _calculate_community_similarity(self, members: List[int]) -> float:
        """
        Calculate average similarity within community
        """
        if len(members) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if self.social_graph.has_edge(members[i], members[j]):
                    edge_data = self.social_graph.get_edge_data(members[i], members[j])
                    if 'similarity' in edge_data:
                        similarities.append(edge_data['similarity'])
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_community_density(self, members: List[int]) -> float:
        """
        Calculate density of community subgraph
        """
        if len(members) < 2:
            return 0.0
        
        subgraph = self.social_graph.subgraph(members)
        return nx.density(subgraph)
    
    def evaluate_communities(self) -> Dict[str, float]:
        """
        Evaluate quality of detected communities
        """
        if not self.communities:
            logger.error("No communities detected yet!")
            return {}
        
        # Get all node features for evaluation
        all_nodes = list(self.social_graph.nodes())
        texts = [self.social_graph.nodes[node]['text'] for node in all_nodes]
        
        # Vectorize texts
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Get community labels for evaluation
        community_labels = np.zeros(len(all_nodes))
        for comm_id, members in self.communities.items():
            for node in members:
                if node < len(all_nodes):
                    community_labels[node] = comm_id
        
        # Calculate metrics
        metrics = {}
        
        # Silhouette score
        if len(set(community_labels)) > 1:
            # Convert sparse matrix to dense for sklearn metrics
            tfidf_dense = tfidf_matrix.toarray()
            metrics['silhouette'] = silhouette_score(tfidf_dense, community_labels)
        else:
            metrics['silhouette'] = 0.0
        
        # Davies-Bouldin score
        if len(set(community_labels)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(tfidf_dense, community_labels)
        else:
            metrics['davies_bouldin'] = 0.0
        
        # Calinski-Harabasz score
        if len(set(community_labels)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(tfidf_dense, community_labels)
        else:
            metrics['calinski_harabasz'] = 0.0
        
        # Community statistics
        metrics['num_communities'] = len(self.communities)
        metrics['avg_community_size'] = np.mean([len(members) for members in self.communities.values()])
        metrics['avg_similarity'] = np.mean([features['avg_similarity'] for features in self.community_features.values()])
        metrics['avg_density'] = np.mean([features['density'] for features in self.community_features.values()])
        
        logger.info("Community evaluation completed:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def plot_communities(self, save_path: str = None):
        """
        Visualize communities
        """
        if not self.communities:
            logger.error("No communities to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Community size distribution
        sizes = [len(members) for members in self.communities.values()]
        axes[0, 0].hist(sizes, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Community Size Distribution')
        axes[0, 0].set_xlabel('Community Size')
        axes[0, 0].set_ylabel('Frequency')
        
        # Community similarity vs density
        similarities = [features['avg_similarity'] for features in self.community_features.values()]
        densities = [features['density'] for features in self.community_features.values()]
        axes[0, 1].scatter(similarities, densities, alpha=0.6)
        axes[0, 1].set_title('Community Similarity vs Density')
        axes[0, 1].set_xlabel('Average Similarity')
        axes[0, 1].set_ylabel('Density')
        
        # Top keywords per community (first 4 communities)
        for i, (comm_id, features) in enumerate(list(self.community_features.items())[:4]):
            keywords = features['top_keywords'][:5]
            axes[1, 0].barh(range(len(keywords)), [1] * len(keywords), 
                           label=f'Community {comm_id}')
            axes[1, 0].set_yticks(range(len(keywords)))
            axes[1, 0].set_yticklabels(keywords)
            axes[1, 0].set_title('Top Keywords by Community')
        
        # Network visualization (simplified)
        if self.social_graph:
            pos = nx.spring_layout(self.social_graph, k=1, iterations=50)
            nx.draw(self.social_graph, pos, 
                   node_size=20, 
                   node_color='lightblue',
                   with_labels=False,
                   alpha=0.6,
                   ax=axes[1, 1])
            axes[1, 1].set_title('Social Graph Structure')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Community visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, output_path: str):
        """
        Save community detection results
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save communities - convert numpy types to native Python types
        serializable_communities = {}
        for comm_id, members in self.communities.items():
            serializable_communities[str(comm_id)] = [int(member) for member in members]
        
        with open(os.path.join(output_path, 'communities.json'), 'w') as f:
            json.dump(serializable_communities, f, indent=2)
        
        # Save community features
        with open(os.path.join(output_path, 'community_features.json'), 'w') as f:
            # Convert numpy types to native Python types
            serializable_features = {}
            for comm_id, features in self.community_features.items():
                serializable_features[str(comm_id)] = {
                    'size': int(features['size']),
                    'texts': features['texts'],
                    'users': features['users'],
                    'locations': features['locations'],
                    'avg_similarity': float(features['avg_similarity']),
                    'density': float(features['density']),
                    'top_keywords': features['top_keywords']
                }
            json.dump(serializable_features, f, indent=2)
        
        # Save evaluation metrics
        metrics = self.evaluate_communities()
        with open(os.path.join(output_path, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create community detector
    detector = CommunityDetector()
    
    # Sample data
    sample_texts = [
        "Hurricane warning issued for coastal areas",
        "Earthquake hits region with magnitude 6.5",
        "Flooding reported in downtown area",
        "Wildfire spreads rapidly through forest",
        "Tornado warning for multiple counties",
        "Heavy rainfall causes flash floods"
    ]
    
    # Build social graph
    graph = detector.build_social_graph(sample_texts)
    
    # Detect communities
    communities = detector.detect_communities(method='content_based')
    
    # Evaluate and plot
    metrics = detector.evaluate_communities()
    detector.plot_communities()
    
    print("Community detection example completed!")
