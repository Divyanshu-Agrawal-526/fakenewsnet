from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
import uuid

# Import models and utilities
from models.fake_news_detector import FakeNewsDetector
from models.disaster_classifier import DisasterClassifier
from models.fact_checker import FactChecker
from models.multimodal_classifier_simple import simple_multimodal_classifier
from utils.location_services import LocationServices
from utils.authority_contact import AuthorityContact
from utils.image_processor import ImageProcessor
from utils.text_processor import TextProcessor

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize components
fake_news_detector = FakeNewsDetector()
disaster_classifier = DisasterClassifier()
fact_checker = FactChecker()
location_services = LocationServices()
authority_contact = AuthorityContact()
image_processor = ImageProcessor()
text_processor = TextProcessor()

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route('/analyze', methods=['POST'])
def analyze_tweet():
    """
    Analyze a tweet for fake news detection using multimodal classification
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text content is required'}), 400
        
        text = data['text']
        image_path = data.get('image_path', None)
        location = data.get('location', None)
        user_id = data.get('user_id', str(uuid.uuid4()))
        
        # Process text
        processed_text = text_processor.preprocess(text)
        
        # Use simple multimodal classifier for comprehensive analysis
        classification_result = simple_multimodal_classifier.classify(processed_text, image_path)
        
        # Extract results
        prediction = classification_result['prediction']
        confidence = classification_result['confidence']
        modality = classification_result['modality']
        explanation = classification_result['explanation']
        probabilities = classification_result['probabilities']
        
        # Determine if it's fake or real
        is_fake = prediction == 'fake'
        disaster_type = prediction if not is_fake else None
        
        # Fact check if real
        fact_check_result = None
        if not is_fake:
            fact_check_result = fact_checker.verify(text, location)
        else:
            fact_check_result = {'verified': False, 'sources': []}
        
        # Get location information
        location_info = None
        if location:
            location_info = location_services.get_location_info(location)
        
        # Prepare response
        response = {
            'tweet_id': str(uuid.uuid4()),
            'user_id': user_id,
            'text': text,
            'processed_text': processed_text,
            'image_path': image_path,
            'multimodal_analysis': {
                'prediction': prediction,
                'is_fake': is_fake,
                'confidence': confidence,
                'modality': modality,
                'explanation': explanation,
                'probabilities': probabilities
            },
            'disaster_classification': {
                'type': disaster_type,
                'confidence': confidence
            },
            'fact_checking': fact_check_result,
            'location_info': location_info,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error in analyze_tweet: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/classify', methods=['POST'])
def classify_disaster():
    """
    Classify disaster type for real tweets
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text content is required'}), 400
        
        text = data['text']
        processed_text = text_processor.preprocess(text)
        
        result = disaster_classifier.predict(processed_text)
        
        return jsonify({
            'disaster_type': result['prediction'],
            'confidence': result['confidence'],
            'explanation': result['explanation']
        }), 200
        
    except Exception as e:
        logging.error(f"Error in classify_disaster: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/fact-check', methods=['POST'])
def fact_check():
    """
    Verify tweet authenticity through fact checking
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text content is required'}), 400
        
        text = data['text']
        location = data.get('location', None)
        
        result = fact_checker.verify(text, location)
        
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"Error in fact_check: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/authorities', methods=['GET'])
def get_authorities():
    """
    Get relevant authorities based on location
    """
    try:
        location = request.args.get('location')
        disaster_type = request.args.get('disaster_type')
        
        if not location:
            return jsonify({'error': 'Location parameter is required'}), 400
        
        authorities = location_services.get_authorities(location, disaster_type)
        
        return jsonify({
            'location': location,
            'disaster_type': disaster_type,
            'authorities': authorities
        }), 200
        
    except Exception as e:
        logging.error(f"Error in get_authorities: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/contact-authority', methods=['POST'])
def contact_authority():
    """
    Send alert to relevant authorities
    """
    try:
        data = request.get_json()
        
        if not data or 'authority_id' not in data or 'message' not in data:
            return jsonify({'error': 'Authority ID and message are required'}), 400
        
        authority_id = data['authority_id']
        message = data['message']
        location = data.get('location', None)
        disaster_type = data.get('disaster_type', None)
        
        result = authority_contact.send_alert(authority_id, message, location, disaster_type)
        
        return jsonify({
            'success': result['success'],
            'message': result['message'],
            'alert_id': result.get('alert_id')
        }), 200
        
    except Exception as e:
        logging.error(f"Error in contact_authority: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/upload-image', methods=['POST'])
def upload_image():
    """
    Upload and analyze image from tweet
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Create upload folder if it doesn't exist
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            file.save(filepath)
            
            # Analyze image
            image_analysis = image_processor.analyze_image(filepath)
            
            return jsonify({
                'filename': filename,
                'analysis': image_analysis
            }), 200
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        logging.error(f"Error in upload_image: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple tweets in batch
    """
    try:
        data = request.get_json()
        
        if not data or 'tweets' not in data:
            return jsonify({'error': 'Tweets array is required'}), 400
        
        tweets = data['tweets']
        results = []
        
        for tweet in tweets:
            try:
                text = tweet.get('text', '')
                location = tweet.get('location', None)
                
                processed_text = text_processor.preprocess(text)
                fake_news_result = fake_news_detector.predict(processed_text)
                
                result = {
                    'tweet_id': tweet.get('id', str(uuid.uuid4())),
                    'text': text,
                    'fake_news_detection': fake_news_result
                }
                
                if fake_news_result['prediction'] == 'real':
                    disaster_result = disaster_classifier.predict(processed_text)
                    result['disaster_classification'] = disaster_result
                
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing tweet: {str(e)}")
                results.append({
                    'tweet_id': tweet.get('id', str(uuid.uuid4())),
                    'error': 'Failed to process tweet'
                })
        
        return jsonify({
            'total_tweets': len(tweets),
            'processed_tweets': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        logging.error(f"Error in batch_analyze: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """
    Get system statistics and performance metrics
    """
    try:
        stats = {
            'total_analyses': fake_news_detector.get_total_analyses(),
            'accuracy': fake_news_detector.get_accuracy(),
            'disaster_distribution': disaster_classifier.get_disaster_distribution(),
            'system_uptime': datetime.utcnow().isoformat(),
            'models_status': {
                'fake_news_detector': fake_news_detector.is_loaded(),
                'disaster_classifier': disaster_classifier.is_loaded(),
                'fact_checker': fact_checker.is_loaded()
            }
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logging.error(f"Error in get_statistics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500 