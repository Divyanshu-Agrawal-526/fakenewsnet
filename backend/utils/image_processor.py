import cv2
import numpy as np
from PIL import Image
import requests
import logging
from typing import Dict, Any, List
import os
from io import BytesIO
import base64

class ImageProcessor:
    def __init__(self):
        self.disaster_keywords = {
            'fire': ['fire', 'flame', 'smoke', 'burning', 'blaze'],
            'flood': ['water', 'flood', 'overflow', 'river', 'stream'],
            'hurricane': ['storm', 'wind', 'rain', 'clouds', 'destruction'],
            'earthquake': ['damage', 'destruction', 'building', 'crack', 'debris']
        }
        
        # Load pre-trained models for image analysis
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for disaster-related content"""
        try:
            # Load image
            image = cv2.imread(image_path)
            
            if image is None:
                return {
                    'error': 'Unable to load image',
                    'analysis': {}
                }
            
            # Convert to RGB for analysis
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            analysis = {
                'basic_info': self.get_basic_info(image),
                'disaster_indicators': self.detect_disaster_indicators(image_rgb),
                'image_quality': self.assess_image_quality(image),
                'metadata': self.extract_metadata(image_path),
                'face_detection': self.detect_faces(image),
                'text_detection': self.detect_text(image)
            }
            
            return {
                'success': True,
                'analysis': analysis
            }
            
        except Exception as e:
            logging.error(f"Error analyzing image: {e}")
            return {
                'error': f'Error analyzing image: {str(e)}',
                'analysis': {}
            }
    
    def analyze_image_from_url(self, image_url: str) -> Dict[str, Any]:
        """Analyze image from URL"""
        try:
            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(BytesIO(response.content))
            
            # Convert to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Save temporarily for analysis
            temp_path = f"temp_image_{hash(image_url)}.jpg"
            cv2.imwrite(temp_path, image_cv)
            
            # Analyze
            result = self.analyze_image(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            logging.error(f"Error analyzing image from URL: {e}")
            return {
                'error': f'Error analyzing image from URL: {str(e)}',
                'analysis': {}
            }
    
    def get_basic_info(self, image) -> Dict[str, Any]:
        """Get basic image information"""
        height, width = image.shape[:2]
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': width / height if height > 0 else 0,
            'total_pixels': width * height,
            'channels': image.shape[2] if len(image.shape) > 2 else 1
        }
    
    def detect_disaster_indicators(self, image_rgb) -> Dict[str, Any]:
        """Detect disaster-related indicators in image"""
        indicators = {
            'fire_detected': False,
            'smoke_detected': False,
            'water_detected': False,
            'damage_detected': False,
            'confidence_scores': {}
        }
        
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Fire detection (red/orange colors)
            lower_fire = np.array([0, 50, 50])
            upper_fire = np.array([20, 255, 255])
            fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
            fire_pixels = cv2.countNonZero(fire_mask)
            fire_ratio = fire_pixels / (image_rgb.shape[0] * image_rgb.shape[1])
            
            if fire_ratio > 0.05:  # 5% threshold
                indicators['fire_detected'] = True
                indicators['confidence_scores']['fire'] = min(fire_ratio * 2, 1.0)
            
            # Smoke detection (gray colors)
            lower_smoke = np.array([0, 0, 100])
            upper_smoke = np.array([180, 30, 200])
            smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
            smoke_pixels = cv2.countNonZero(smoke_mask)
            smoke_ratio = smoke_pixels / (image_rgb.shape[0] * image_rgb.shape[1])
            
            if smoke_ratio > 0.1:  # 10% threshold
                indicators['smoke_detected'] = True
                indicators['confidence_scores']['smoke'] = min(smoke_ratio * 1.5, 1.0)
            
            # Water detection (blue colors)
            lower_water = np.array([100, 50, 50])
            upper_water = np.array([130, 255, 255])
            water_mask = cv2.inRange(hsv, lower_water, upper_water)
            water_pixels = cv2.countNonZero(water_mask)
            water_ratio = water_pixels / (image_rgb.shape[0] * image_rgb.shape[1])
            
            if water_ratio > 0.15:  # 15% threshold
                indicators['water_detected'] = True
                indicators['confidence_scores']['water'] = min(water_ratio * 1.3, 1.0)
            
            # Damage detection (edge detection)
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = cv2.countNonZero(edges)
            edge_ratio = edge_pixels / (image_rgb.shape[0] * image_rgb.shape[1])
            
            if edge_ratio > 0.2:  # 20% threshold
                indicators['damage_detected'] = True
                indicators['confidence_scores']['damage'] = min(edge_ratio * 1.2, 1.0)
            
        except Exception as e:
            logging.error(f"Error detecting disaster indicators: {e}")
        
        return indicators
    
    def assess_image_quality(self, image) -> Dict[str, Any]:
        """Assess image quality metrics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = 1.0 / (1.0 + laplacian_var / 1000.0)
            
            # Calculate brightness
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # Calculate contrast
            contrast = np.std(gray)
            contrast_score = min(contrast / 50.0, 1.0)
            
            # Overall quality score
            quality_score = (blur_score + brightness_score + contrast_score) / 3
            
            return {
                'blur_score': blur_score,
                'brightness_score': brightness_score,
                'contrast_score': contrast_score,
                'overall_quality': quality_score,
                'is_blurry': blur_score < 0.5,
                'is_dark': brightness_score < 0.3,
                'is_low_contrast': contrast_score < 0.3
            }
            
        except Exception as e:
            logging.error(f"Error assessing image quality: {e}")
            return {
                'blur_score': 0.0,
                'brightness_score': 0.0,
                'contrast_score': 0.0,
                'overall_quality': 0.0,
                'is_blurry': True,
                'is_dark': True,
                'is_low_contrast': True
            }
    
    def extract_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract image metadata"""
        try:
            image = Image.open(image_path)
            
            metadata = {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'info': image.info
            }
            
            # Extract EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                metadata['exif'] = {
                    'datetime': exif.get(36867, 'Unknown'),  # DateTime
                    'make': exif.get(271, 'Unknown'),        # Make
                    'model': exif.get(272, 'Unknown'),       # Model
                    'software': exif.get(305, 'Unknown')     # Software
                }
            
            return metadata
            
        except Exception as e:
            logging.error(f"Error extracting metadata: {e}")
            return {
                'format': 'Unknown',
                'mode': 'Unknown',
                'size': (0, 0),
                'info': {},
                'exif': {}
            }
    
    def detect_faces(self, image) -> Dict[str, Any]:
        """Detect faces in image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return {
                'face_count': len(faces),
                'faces_detected': len(faces) > 0,
                'face_locations': faces.tolist() if len(faces) > 0 else []
            }
            
        except Exception as e:
            logging.error(f"Error detecting faces: {e}")
            return {
                'face_count': 0,
                'faces_detected': False,
                'face_locations': []
            }
    
    def detect_text(self, image) -> Dict[str, Any]:
        """Detect text in image (simplified)"""
        try:
            # This is a simplified text detection
            # In a real implementation, you'd use OCR libraries like Tesseract
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection for text regions
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be text
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text typically has aspect ratio > 1 and reasonable size
                if aspect_ratio > 1 and w > 20 and h > 10:
                    text_regions.append((x, y, w, h))
            
            return {
                'text_regions_detected': len(text_regions),
                'text_regions': text_regions,
                'has_text': len(text_regions) > 0
            }
            
        except Exception as e:
            logging.error(f"Error detecting text: {e}")
            return {
                'text_regions_detected': 0,
                'text_regions': [],
                'has_text': False
            }
    
    def compare_images(self, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """Compare two images for similarity"""
        try:
            # Load images
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1 is None or img2 is None:
                return {
                    'error': 'Unable to load one or both images',
                    'similarity_score': 0.0
                }
            
            # Resize images to same size for comparison
            height = min(img1.shape[0], img2.shape[0])
            width = min(img1.shape[1], img2.shape[1])
            
            img1_resized = cv2.resize(img1, (width, height))
            img2_resized = cv2.resize(img2, (width, height))
            
            # Calculate structural similarity
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to grayscale for SSIM
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            
            similarity_score = ssim(gray1, gray2)
            
            return {
                'similarity_score': similarity_score,
                'is_similar': similarity_score > 0.8,
                'comparison_metrics': {
                    'ssim': similarity_score,
                    'size_difference': abs(img1.shape[0] * img1.shape[1] - img2.shape[0] * img2.shape[1])
                }
            }
            
        except Exception as e:
            logging.error(f"Error comparing images: {e}")
            return {
                'error': f'Error comparing images: {str(e)}',
                'similarity_score': 0.0
            } 