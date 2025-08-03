from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv
import logging

# Import our modules
from api.routes import api_bp
from models.fake_news_detector import FakeNewsDetector
from models.disaster_classifier import DisasterClassifier
from models.fact_checker import FactChecker
from utils.location_services import LocationServices
from utils.authority_contact import AuthorityContact

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

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')

# Initialize ML models
fake_news_detector = FakeNewsDetector()
disaster_classifier = DisasterClassifier()
fact_checker = FactChecker()
location_services = LocationServices()
authority_contact = AuthorityContact()

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Fake News Detection API',
        'version': '1.0.0',
        'status': 'running'
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'fake_news_detector': fake_news_detector.is_loaded(),
            'disaster_classifier': disaster_classifier.is_loaded(),
            'fact_checker': fact_checker.is_loaded()
        }
    })

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
    
    # Load models
    try:
        fake_news_detector.load_model()
        disaster_classifier.load_model()
        fact_checker.load_model()
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 