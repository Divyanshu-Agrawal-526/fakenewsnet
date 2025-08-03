# Fake News Detection During Natural Disasters

## Project Overview
A comprehensive multimodal fake news detection system specifically designed for natural disasters. The system analyzes tweets to determine authenticity and classify real disasters into categories (wildfire, floods, hurricane, earthquake), with integrated fact-checking and authority contact features.

## Features
- **Fake News Detection**: AI-powered classification of real vs fake disaster tweets
- **Disaster Classification**: Categorizes real disasters into wildfire, floods, hurricane, earthquake
- **Multimodal Analysis**: Processes both text and images from tweets
- **Fact Checking**: Verifies authenticity through multiple sources
- **Location-Based Services**: Identifies relevant authorities based on tweet location
- **Authority Contact System**: Direct communication with emergency services
- **Real-time Processing**: Live analysis of incoming disaster reports

## Technology Stack
- **Backend**: Python Flask, TensorFlow/PyTorch
- **Frontend**: React.js, Material-UI
- **Database**: SQLite
- **ML Models**: BERT, CNN for images, Ensemble methods
- **APIs**: Twitter API, Fact-checking APIs, Emergency services APIs

## Project Structure
```
fakenews/
├── backend/                 # Flask API server
│   ├── models/             # ML models and training
│   ├── api/               # API endpoints
│   ├── utils/             # Utility functions
│   └── data/              # Dataset and preprocessing
├── frontend/              # React web application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   └── services/      # API services
├── ml_pipeline/           # ML training and evaluation
├── docs/                  # Documentation
└── tests/                 # Unit and integration tests
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- Git

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## Usage
1. Start the backend server (Flask API)
2. Start the frontend development server
3. Access the web application at `http://localhost:3000`
4. Upload or input disaster-related tweets for analysis
5. View classification results and contact relevant authorities

## API Endpoints
- `POST /api/analyze` - Analyze tweet for fake news detection
- `POST /api/classify` - Classify disaster type
- `POST /api/fact-check` - Verify tweet authenticity
- `GET /api/authorities` - Get relevant authorities by location
- `POST /api/contact-authority` - Send alert to authorities

## Contributing
This is a capstone project for final year studies. For academic purposes only.

## License
Academic project - not for commercial use. 