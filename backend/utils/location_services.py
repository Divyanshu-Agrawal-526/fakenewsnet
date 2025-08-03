import requests
import json
import logging
from typing import Dict, Any, List
import os
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

class LocationServices:
    def __init__(self):
        self.geocoder = Nominatim(user_agent="fakenews_detector")
        self.api_keys = {
            'google': os.getenv('GOOGLE_API_KEY', ''),
            'openweather': os.getenv('OPENWEATHER_API_KEY', '')
        }
        
        # Authority database (simplified - in real implementation, this would be a proper database)
        self.authority_database = {
            'wildfire': {
                'fire_department': {
                    'name': 'Local Fire Department',
                    'phone': '911',
                    'email': 'emergency@firedept.gov',
                    'response_time': '5-10 minutes'
                },
                'forest_service': {
                    'name': 'US Forest Service',
                    'phone': '1-800-832-1355',
                    'email': 'fire@fs.fed.us',
                    'response_time': '15-30 minutes'
                }
            },
            'flood': {
                'emergency_management': {
                    'name': 'Emergency Management Agency',
                    'phone': '911',
                    'email': 'emergency@ema.gov',
                    'response_time': '10-15 minutes'
                },
                'water_resources': {
                    'name': 'Water Resources Department',
                    'phone': '1-800-555-0123',
                    'email': 'flood@water.gov',
                    'response_time': '20-30 minutes'
                }
            },
            'hurricane': {
                'national_weather_service': {
                    'name': 'National Weather Service',
                    'phone': '1-800-555-0124',
                    'email': 'hurricane@weather.gov',
                    'response_time': 'Immediate'
                },
                'emergency_management': {
                    'name': 'Emergency Management Agency',
                    'phone': '911',
                    'email': 'emergency@ema.gov',
                    'response_time': '10-15 minutes'
                }
            },
            'earthquake': {
                'usgs': {
                    'name': 'US Geological Survey',
                    'phone': '1-800-555-0125',
                    'email': 'earthquake@usgs.gov',
                    'response_time': 'Immediate'
                },
                'emergency_management': {
                    'name': 'Emergency Management Agency',
                    'phone': '911',
                    'email': 'emergency@ema.gov',
                    'response_time': '10-15 minutes'
                }
            }
        }
    
    def get_location_info(self, location: str) -> Dict[str, Any]:
        """Get detailed information about a location"""
        try:
            # Geocode the location
            location_data = self.geocode_location(location)
            
            if not location_data:
                return {
                    'error': 'Unable to geocode location',
                    'location': location
                }
            
            # Get weather information
            weather_info = self.get_weather_info(location_data['coordinates'])
            
            # Get nearby emergency services
            emergency_services = self.get_emergency_services(location_data['coordinates'])
            
            return {
                'location': location,
                'coordinates': location_data['coordinates'],
                'address': location_data['address'],
                'weather': weather_info,
                'emergency_services': emergency_services,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logging.error(f"Error getting location info: {e}")
            return {
                'error': 'Error processing location',
                'location': location
            }
    
    def geocode_location(self, location: str) -> Dict[str, Any]:
        """Convert location string to coordinates"""
        try:
            # Add a small delay to respect rate limits
            time.sleep(1)
            
            # Geocode the location
            location_obj = self.geocoder.geocode(location)
            
            if location_obj:
                return {
                    'coordinates': (location_obj.latitude, location_obj.longitude),
                    'address': location_obj.address,
                    'raw': location_obj.raw
                }
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error geocoding location {location}: {e}")
            return None
    
    def get_weather_info(self, coordinates: tuple) -> Dict[str, Any]:
        """Get current weather information for coordinates"""
        try:
            if not self.api_keys['openweather']:
                return {'error': 'Weather API key not available'}
            
            lat, lon = coordinates
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_keys['openweather'],
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'description': data['weather'][0]['description'],
                    'wind_speed': data['wind']['speed'] if 'wind' in data else 0,
                    'conditions': data['weather'][0]['main']
                }
            else:
                return {'error': 'Unable to fetch weather data'}
                
        except Exception as e:
            logging.error(f"Error getting weather info: {e}")
            return {'error': 'Error fetching weather data'}
    
    def get_emergency_services(self, coordinates: tuple) -> List[Dict[str, Any]]:
        """Get nearby emergency services"""
        try:
            # In a real implementation, this would query a database of emergency services
            # For now, return a simplified list
            services = [
                {
                    'type': 'Police Department',
                    'distance': '0.5 miles',
                    'phone': '911',
                    'address': 'Local Police Station'
                },
                {
                    'type': 'Fire Department',
                    'distance': '1.2 miles',
                    'phone': '911',
                    'address': 'Local Fire Station'
                },
                {
                    'type': 'Hospital',
                    'distance': '2.1 miles',
                    'phone': '911',
                    'address': 'Local Hospital'
                }
            ]
            
            return services
            
        except Exception as e:
            logging.error(f"Error getting emergency services: {e}")
            return []
    
    def get_authorities(self, location: str, disaster_type: str = None) -> List[Dict[str, Any]]:
        """Get relevant authorities for a location and disaster type"""
        try:
            # Get location coordinates
            location_data = self.geocode_location(location)
            
            if not location_data:
                return []
            
            authorities = []
            
            # Get general emergency authorities
            general_authorities = [
                {
                    'id': 'local_police',
                    'name': 'Local Police Department',
                    'type': 'Police',
                    'phone': '911',
                    'email': 'emergency@police.gov',
                    'response_time': '5-10 minutes',
                    'distance': '0.5 miles'
                },
                {
                    'id': 'local_fire',
                    'name': 'Local Fire Department',
                    'type': 'Fire',
                    'phone': '911',
                    'email': 'emergency@fire.gov',
                    'response_time': '5-10 minutes',
                    'distance': '1.0 miles'
                },
                {
                    'id': 'emergency_management',
                    'name': 'Emergency Management Agency',
                    'type': 'Emergency Management',
                    'phone': '911',
                    'email': 'emergency@ema.gov',
                    'response_time': '10-15 minutes',
                    'distance': '2.0 miles'
                }
            ]
            
            authorities.extend(general_authorities)
            
            # Add disaster-specific authorities
            if disaster_type and disaster_type in self.authority_database:
                disaster_authorities = self.authority_database[disaster_type]
                
                for authority_id, authority_info in disaster_authorities.items():
                    authority_info['id'] = authority_id
                    authority_info['type'] = disaster_type.title()
                    authority_info['distance'] = 'Varies'
                    authorities.append(authority_info)
            
            return authorities
            
        except Exception as e:
            logging.error(f"Error getting authorities: {e}")
            return []
    
    def calculate_distance(self, coord1: tuple, coord2: tuple) -> float:
        """Calculate distance between two coordinates in miles"""
        try:
            return geodesic(coord1, coord2).miles
        except Exception as e:
            logging.error(f"Error calculating distance: {e}")
            return 0.0
    
    def is_location_valid(self, location: str) -> bool:
        """Check if a location string is valid"""
        try:
            location_data = self.geocode_location(location)
            return location_data is not None
        except Exception as e:
            logging.error(f"Error validating location: {e}")
            return False
    
    def get_nearby_locations(self, coordinates: tuple, radius_miles: float = 10) -> List[Dict[str, Any]]:
        """Get nearby locations within a radius"""
        try:
            # This would typically query a database of known locations
            # For now, return a simplified list
            nearby = [
                {
                    'name': 'Downtown Area',
                    'coordinates': (coordinates[0] + 0.01, coordinates[1] + 0.01),
                    'distance': 1.2,
                    'type': 'Urban'
                },
                {
                    'name': 'Suburban Area',
                    'coordinates': (coordinates[0] - 0.01, coordinates[1] - 0.01),
                    'distance': 2.5,
                    'type': 'Residential'
                }
            ]
            
            return [loc for loc in nearby if loc['distance'] <= radius_miles]
            
        except Exception as e:
            logging.error(f"Error getting nearby locations: {e}")
            return [] 