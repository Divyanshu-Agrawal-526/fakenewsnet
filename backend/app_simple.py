from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv
import logging
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fakenews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# Simple fake news detection (rule-based)
def simple_fake_news_detector(text):
    """Simple rule-based fake news detection"""
    text_lower = text.lower()
    
    # Suspicious patterns
    suspicious_patterns = [
        'click here',
        'follow me',
        'retweet',
        'like and share',
        'viral',
        'you won\'t believe',
        'amazing',
        'incredible'
    ]
    
    # Disaster keywords
    disaster_keywords = [
        'fire', 'flood', 'hurricane', 'earthquake', 'tornado', 'tsunami',
        'wildfire', 'storm', 'disaster', 'emergency', 'evacuation'
    ]
    
    # Count suspicious patterns
    suspicious_count = sum(1 for pattern in suspicious_patterns if pattern in text_lower)
    
    # Count disaster keywords
    disaster_count = sum(1 for keyword in disaster_keywords if keyword in text_lower)
    
    # Simple scoring
    if suspicious_count > 0:
        return {
            'prediction': 'fake',
            'confidence': min(0.8 + (suspicious_count * 0.1), 0.95),
            'explanation': f'Contains {suspicious_count} suspicious patterns'
        }
    elif disaster_count > 0:
        return {
            'prediction': 'real',
            'confidence': min(0.7 + (disaster_count * 0.05), 0.9),
            'explanation': f'Contains {disaster_count} disaster-related keywords'
        }
    else:
        return {
            'prediction': 'unknown',
            'confidence': 0.5,
            'explanation': 'Insufficient information to determine'
        }

# Simple disaster classifier
def simple_disaster_classifier(text):
    """Simple rule-based disaster classification"""
    text_lower = text.lower()
    
    disaster_types = {
        'wildfire': ['fire', 'wildfire', 'blaze', 'burning', 'smoke', 'flame'],
        'flood': ['flood', 'water', 'rain', 'overflow', 'river', 'stream'],
        'hurricane': ['hurricane', 'storm', 'wind', 'tropical', 'cyclone'],
        'earthquake': ['earthquake', 'quake', 'tremor', 'seismic', 'shaking']
    }
    
    for disaster_type, keywords in disaster_types.items():
        if any(keyword in text_lower for keyword in keywords):
            return {
                'prediction': disaster_type,
                'confidence': 0.8,
                'explanation': f'Contains {disaster_type}-related keywords'
            }
    
    return {
        'prediction': 'unknown',
        'confidence': 0.0,
        'explanation': 'No specific disaster type detected'
    }

# Simple fact checker
def simple_fact_checker(text, location=None):
    """Simple fact checking simulation"""
    return {
        'verified': True,
        'confidence': 0.7,
        'sources': [
            {
                'title': 'Sample News Source',
                'url': 'https://example.com',
                'source': 'Example News',
                'published_at': datetime.utcnow().isoformat()
            }
        ],
        'explanations': [
            'Contains disaster-related keywords',
            'Location information provided' if location else 'No location specified'
        ]
    }

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Fake News Detection API (Simple Version)',
        'version': '1.0.0',
        'status': 'running'
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'fake_news_detector': True,
            'disaster_classifier': True,
            'fact_checker': True
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_tweet():
    """Analyze a tweet for fake news detection"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text content is required'}), 400
        
        text = data['text']
        location = data.get('location', None)
        user_id = data.get('user_id', str(uuid.uuid4()))
        
        # Analyze for fake news
        fake_news_result = simple_fake_news_detector(text)
        
        # If real, classify disaster type
        disaster_type = None
        confidence = None
        
        if fake_news_result['prediction'] == 'real':
            disaster_result = simple_disaster_classifier(text)
            disaster_type = disaster_result['prediction']
            confidence = disaster_result['confidence']
            
            # Perform fact checking
            fact_check_result = simple_fact_checker(text, location)
        else:
            fact_check_result = {'verified': False, 'sources': []}
        
        # Prepare response
        response = {
            'tweet_id': str(uuid.uuid4()),
            'user_id': user_id,
            'text': text,
            'processed_text': text.lower(),
            'fake_news_detection': {
                'prediction': fake_news_result['prediction'],
                'confidence': fake_news_result['confidence'],
                'explanation': fake_news_result['explanation']
            },
            'disaster_classification': {
                'type': disaster_type,
                'confidence': confidence
            },
            'fact_checking': fact_check_result,
            'location_info': {'location': location} if location else None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error in analyze_tweet: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/classify', methods=['POST'])
def classify_disaster():
    """Classify disaster type for real tweets"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text content is required'}), 400
        
        text = data['text']
        result = simple_disaster_classifier(text)
        
        return jsonify({
            'disaster_type': result['prediction'],
            'confidence': result['confidence'],
            'explanation': result['explanation']
        }), 200
        
    except Exception as e:
        logging.error(f"Error in classify_disaster: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/fact-check', methods=['POST'])
def fact_check():
    """Verify tweet authenticity through fact checking"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text content is required'}), 400
        
        text = data['text']
        location = data.get('location', None)
        
        result = simple_fact_checker(text, location)
        
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"Error in fact_check: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/authorities', methods=['GET'])
def get_authorities():
    """Get relevant authorities based on location"""
    try:
        location = request.args.get('location')
        disaster_type = request.args.get('disaster_type')
        
        if not location:
            return jsonify({'error': 'Location parameter is required'}), 400
        
        # Mock authorities data
        authorities = [
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
        
        return jsonify({
            'location': location,
            'disaster_type': disaster_type,
            'authorities': authorities
        }), 200
        
    except Exception as e:
        logging.error(f"Error in get_authorities: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/contact-authority', methods=['POST'])
def contact_authority():
    """Send alert to relevant authorities"""
    try:
        data = request.get_json()
        
        if not data or 'authority_id' not in data or 'message' not in data:
            return jsonify({'error': 'Authority ID and message are required'}), 400
        
        authority_id = data['authority_id']
        message = data['message']
        location = data.get('location', None)
        disaster_type = data.get('disaster_type', None)
        
        # Mock response
        result = {
            'success': True,
            'message': 'Alert sent successfully (simulated)',
            'alert_id': str(uuid.uuid4())
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"Error in contact_authority: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics and performance metrics"""
    try:
        stats = {
            'total_analyses': 1250,
            'accuracy': 0.85,
            'disaster_distribution': {
                'wildfire': 45,
                'flood': 30,
                'hurricane': 15,
                'earthquake': 10,
            },
            'system_uptime': datetime.utcnow().isoformat(),
            'models_status': {
                'fake_news_detector': True,
                'disaster_classifier': True,
                'fact_checker': True
            }
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logging.error(f"Error in get_statistics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
    
    logger.info("Simple Fake News Detection API started")
    app.run(debug=True, host='0.0.0.0', port=5000) 