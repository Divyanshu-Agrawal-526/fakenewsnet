#!/usr/bin/env python3
"""
Geolocation Extraction System for Disaster Reporting
Based on the research paper: "ContCommRTD: A Distributed Content-Based Misinformation-Aware Community Detection System for Real-Time Disaster Reporting"

This implements:
1. Named Entity Recognition (NER) for location extraction
2. Geocoding and coordinate mapping
3. Location-based filtering and clustering
4. Geographic community analysis
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Tuple, Any, Optional
import json
import os
from datetime import datetime
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)

class GeolocationExtractor:
    """
    Extract and process geolocation information from text and metadata
    """
    
    def __init__(self, 
                 use_geocoding: bool = True,
                 cache_locations: bool = True,
                 max_retries: int = 3):
        self.use_geocoding = use_geocoding
        self.cache_locations = cache_locations
        self.max_retries = max_retries
        
        # Location cache
        self.location_cache = {}
        self.coordinate_cache = {}
        
        # Common disaster-related location patterns
        self.location_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:County|City|State|Province|Region|Area)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Beach|Harbor|Bay|River|Lake|Mountain|Forest)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Airport|Hospital|University|Center|Park)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|Avenue|Road|Boulevard|Drive)\b'
        ]
        
        # US States and common abbreviations
        self.us_states = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming'
        }
        
        # Initialize geocoder if needed
        self.geocoder = None
        if self.use_geocoding:
            try:
                self.geocoder = Nominatim(user_agent="disaster_reporter")
                logger.info("Geocoder initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize geocoder: {e}")
                self.use_geocoding = False
        
        logger.info("Geolocation extractor initialized")
    
    def extract_locations_from_text(self, text: str) -> List[str]:
        """
        Extract location mentions from text using regex patterns
        """
        locations = []
        
        # Extract locations using patterns
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            locations.extend(matches)
        
        # Extract US state abbreviations
        state_matches = re.findall(r'\b[A-Z]{2}\b', text)
        for state_abbr in state_matches:
            if state_abbr in self.us_states:
                locations.append(self.us_states[state_abbr])
        
        # Extract city names (simple heuristic)
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Check if it's followed by common location words
                if i + 1 < len(words):
                    next_word = words[i + 1].lower()
                    if next_word in ['city', 'town', 'village', 'county', 'state']:
                        locations.append(word)
        
        # Remove duplicates and clean
        unique_locations = list(set(locations))
        cleaned_locations = [loc.strip() for loc in unique_locations if len(loc.strip()) > 2]
        
        return cleaned_locations
    
    def geocode_location(self, location: str) -> Optional[Tuple[float, float]]:
        """
        Convert location name to coordinates
        """
        if not self.use_geocoding or not self.geocoder:
            return None
        
        # Check cache first
        if location in self.coordinate_cache:
            return self.coordinate_cache[location]
        
        # Try to geocode
        for attempt in range(self.max_retries):
            try:
                # Add some context for better geocoding
                search_query = f"{location}, USA"
                result = self.geocoder.geocode(search_query, timeout=10)
                
                if result:
                    coordinates = (result.latitude, result.longitude)
                    self.coordinate_cache[location] = coordinates
                    logger.debug(f"Geocoded {location} -> {coordinates}")
                    return coordinates
                
                time.sleep(1)  # Rate limiting
                
            except (GeocoderTimedOut, GeocoderUnavailable) as e:
                logger.warning(f"Geocoding attempt {attempt + 1} failed for {location}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
            except Exception as e:
                logger.error(f"Unexpected error geocoding {location}: {e}")
                break
        
        logger.warning(f"Failed to geocode {location} after {self.max_retries} attempts")
        return None
    
    def extract_locations_from_dataset(self, 
                                     texts: List[str], 
                                     metadata: Optional[Dict] = None) -> Dict[int, Dict]:
        """
        Extract locations from a dataset of texts
        """
        logger.info(f"Extracting locations from {len(texts)} texts...")
        
        results = {}
        successful_geocoding = 0
        
        for i, text in enumerate(texts):
            # Extract location mentions
            locations = self.extract_locations_from_text(text)
            
            # Geocode locations
            coordinates = []
            for location in locations:
                coord = self.geocode_location(location)
                if coord:
                    coordinates.append({
                        'location': location,
                        'latitude': coord[0],
                        'longitude': coord[1]
                    })
            
            # Store results
            results[i] = {
                'text': text,
                'extracted_locations': locations,
                'coordinates': coordinates,
                'has_coordinates': len(coordinates) > 0
            }
            
            if coordinates:
                successful_geocoding += 1
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(texts)} texts, {successful_geocoding} geocoded")
        
        logger.info(f"Location extraction completed: {successful_geocoding}/{len(texts)} texts geocoded")
        return results
    
    def create_location_clusters(self, 
                               location_data: Dict[int, Dict], 
                               eps: float = 0.1,
                               min_samples: int = 3) -> Dict[int, List[int]]:
        """
        Create geographic clusters based on coordinates
        """
        logger.info("Creating geographic clusters...")
        
        # Collect all coordinates
        coordinates = []
        text_indices = []
        
        for idx, data in location_data.items():
            for coord_data in data['coordinates']:
                coordinates.append([coord_data['latitude'], coord_data['longitude']])
                text_indices.append(idx)
        
        if len(coordinates) < min_samples:
            logger.warning(f"Not enough coordinates ({len(coordinates)}) for clustering")
            return {}
        
        # Convert to numpy array
        coords_array = np.array(coordinates)
        
        # Create clusters using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_array)
        cluster_labels = clustering.labels_
        
        # Group by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # -1 means noise in DBSCAN
                clusters[label].append(text_indices[i])
        
        logger.info(f"Created {len(clusters)} geographic clusters")
        return dict(clusters)
    
    def analyze_geographic_distribution(self, 
                                      location_data: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Analyze geographic distribution of locations
        """
        logger.info("Analyzing geographic distribution...")
        
        # Collect statistics
        total_texts = len(location_data)
        texts_with_locations = sum(1 for data in location_data.values() if data['extracted_locations'])
        texts_with_coordinates = sum(1 for data in location_data.values() if data['has_coordinates'])
        
        # Location frequency analysis
        all_locations = []
        for data in location_data.values():
            all_locations.extend(data['extracted_locations'])
        
        location_counts = Counter(all_locations)
        top_locations = location_counts.most_common(20)
        
        # Coordinate analysis
        all_coordinates = []
        for data in location_data.values():
            for coord_data in data['coordinates']:
                all_coordinates.append([coord_data['latitude'], coord_data['longitude']])
        
        if all_coordinates:
            coords_array = np.array(all_coordinates)
            lat_range = (coords_array[:, 0].min(), coords_array[:, 0].max())
            lon_range = (coords_array[:, 1].min(), coords_array[:, 1].max())
            
            # Calculate geographic center
            center_lat = coords_array[:, 0].mean()
            center_lon = coords_array[:, 1].mean()
        else:
            lat_range = lon_range = (0, 0)
            center_lat = center_lon = 0
        
        analysis = {
            'total_texts': total_texts,
            'texts_with_locations': texts_with_locations,
            'texts_with_coordinates': texts_with_coordinates,
            'location_extraction_rate': texts_with_locations / total_texts if total_texts > 0 else 0,
            'geocoding_success_rate': texts_with_coordinates / total_texts if total_texts > 0 else 0,
            'top_locations': top_locations,
            'total_unique_locations': len(location_counts),
            'geographic_bounds': {
                'latitude_range': lat_range,
                'longitude_range': lon_range,
                'center': (center_lat, center_lon)
            }
        }
        
        logger.info("Geographic analysis completed")
        return analysis
    
    def plot_geographic_distribution(self, 
                                   location_data: Dict[int, Dict], 
                                   save_path: str = None):
        """
        Visualize geographic distribution
        """
        # Collect coordinates
        coordinates = []
        texts = []
        
        for idx, data in location_data.items():
            for coord_data in data['coordinates']:
                coordinates.append([coord_data['latitude'], coord_data['longitude']])
                texts.append(data['text'][:50] + "..." if len(data['text']) > 50 else data['text'])
        
        if not coordinates:
            logger.warning("No coordinates to plot")
            return
        
        coords_array = np.array(coordinates)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot of coordinates
        axes[0, 0].scatter(coords_array[:, 1], coords_array[:, 0], alpha=0.6, s=50)
        axes[0, 0].set_title('Geographic Distribution of Locations')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Location frequency histogram
        all_locations = []
        for data in location_data.values():
            all_locations.extend(data['extracted_locations'])
        
        location_counts = Counter(all_locations)
        top_10_locations = location_counts.most_common(10)
        
        locations, counts = zip(*top_10_locations)
        axes[0, 1].barh(range(len(locations)), counts)
        axes[0, 1].set_yticks(range(len(locations)))
        axes[0, 1].set_yticklabels(locations)
        axes[0, 1].set_title('Top 10 Most Mentioned Locations')
        axes[0, 1].set_xlabel('Frequency')
        
        # Coordinate density heatmap
        if len(coordinates) > 10:
            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(
                coords_array[:, 0], coords_array[:, 1], bins=20
            )
            im = axes[1, 0].imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            axes[1, 0].set_title('Location Density Heatmap')
            axes[1, 0].set_xlabel('Latitude')
            axes[1, 0].set_ylabel('Longitude')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Success rate pie chart
        total_texts = len(location_data)
        texts_with_coords = sum(1 for data in location_data.values() if data['has_coordinates'])
        texts_without_coords = total_texts - texts_with_coords
        
        axes[1, 1].pie([texts_with_coords, texts_without_coords], 
                       labels=['With Coordinates', 'Without Coordinates'],
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Geocoding Success Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Geographic visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, 
                    location_data: Dict[int, Dict], 
                    output_path: str):
        """
        Save geolocation extraction results
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save location data
        with open(os.path.join(output_path, 'location_data.json'), 'w') as f:
            # Convert numpy types to native Python types
            serializable_data = {}
            for idx, data in location_data.items():
                serializable_data[str(idx)] = {
                    'text': data['text'],
                    'extracted_locations': data['extracted_locations'],
                    'coordinates': [
                        {
                            'location': coord['location'],
                            'latitude': float(coord['latitude']),
                            'longitude': float(coord['longitude'])
                        }
                        for coord in data['coordinates']
                    ],
                    'has_coordinates': data['has_coordinates']
                }
            json.dump(serializable_data, f, indent=2)
        
        # Save analysis
        analysis = self.analyze_geographic_distribution(location_data)
        with open(os.path.join(output_path, 'geographic_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save coordinate cache
        with open(os.path.join(output_path, 'coordinate_cache.json'), 'w') as f:
            json.dump(self.coordinate_cache, f, indent=2)
        
        logger.info(f"Geolocation results saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create geolocation extractor
    extractor = GeolocationExtractor(use_geocoding=False)  # Disable geocoding for demo
    
    # Sample texts
    sample_texts = [
        "Hurricane warning issued for Miami Beach, Florida",
        "Earthquake hits Los Angeles County, California",
        "Flooding reported in downtown Houston, Texas",
        "Wildfire spreads rapidly through Yellowstone National Park",
        "Tornado warning for multiple counties in Oklahoma",
        "Heavy rainfall causes flash floods in New York City"
    ]
    
    # Extract locations
    location_data = extractor.extract_locations_from_dataset(sample_texts)
    
    # Analyze distribution
    analysis = extractor.analyze_geographic_distribution(location_data)
    
    # Plot results
    extractor.plot_geographic_distribution(location_data)
    
    print("Geolocation extraction example completed!")
