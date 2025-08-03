# Technical Documentation

## Fake News Detection During Natural Disasters
### Final Year Capstone Project - Multimodal Classification System

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Machine Learning Models](#machine-learning-models)
4. [API Documentation](#api-documentation)
5. [Database Schema](#database-schema)
6. [Frontend Architecture](#frontend-architecture)
7. [Deployment Guide](#deployment-guide)
8. [Testing Strategy](#testing-strategy)
9. [Security Considerations](#security-considerations)
10. [Performance Optimization](#performance-optimization)

---

## Project Overview

### Problem Statement
During natural disasters, the rapid spread of misinformation through social media can have severe consequences for public safety and emergency response efforts. This project addresses this critical challenge by developing a comprehensive fake news detection system specifically designed for disaster-related content.

### Solution
A multimodal classification system that:
- Detects fake news in disaster-related tweets
- Classifies real disasters into specific types (wildfire, flood, hurricane, earthquake)
- Verifies information authenticity through fact-checking
- Enables direct communication with emergency authorities
- Provides real-time analysis and alerts

### Key Features
- **Fake News Detection**: AI-powered classification using BERT and ensemble methods
- **Disaster Classification**: Multi-class classification for disaster types
- **Fact Checking**: Multi-source verification system
- **Location Services**: Geocoding and authority identification
- **Authority Contact**: Direct alert system for emergency services
- **Real-time Processing**: Live analysis with immediate results

---

## System Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   ML Models     │
│   (React.js)    │◄──►│   (Flask API)   │◄──►│   (TensorFlow)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Database      │
                       │   (SQLite)      │
                       └─────────────────┘
```

### Component Breakdown

#### Backend (Flask API)
- **Framework**: Flask with Flask-CORS
- **Database**: SQLite with SQLAlchemy ORM
- **Authentication**: JWT-based (planned)
- **File Upload**: Image processing for multimodal analysis
- **Caching**: Redis (planned for production)

#### Frontend (React.js)
- **Framework**: React 18 with hooks
- **UI Library**: Material-UI (MUI)
- **State Management**: React Query for server state
- **Routing**: React Router v6
- **Charts**: Chart.js with react-chartjs-2

#### Machine Learning Pipeline
- **Text Processing**: NLTK, spaCy, TextBlob
- **Deep Learning**: TensorFlow, PyTorch
- **NLP Models**: BERT, Transformers
- **Computer Vision**: OpenCV, PIL
- **Ensemble Methods**: Scikit-learn

---

## Machine Learning Models

### 1. Fake News Detector

#### Architecture
- **Primary Model**: BERT (bert-base-uncased)
- **Ensemble Model**: Random Forest + TF-IDF
- **Combination**: Weighted ensemble (70% BERT, 30% RF)

#### Features
```python
# Text Features
- Text length and word count
- Sentiment analysis (polarity, subjectivity)
- Language patterns (uppercase ratio, digit ratio)
- Disaster keywords count
- Credibility indicators

# TF-IDF Features
- 5000 most common words
- Stop words removal
- Lemmatization
```

#### Training Process
1. **Data Preprocessing**: Text cleaning, tokenization
2. **Feature Extraction**: TF-IDF vectorization
3. **Model Training**: BERT fine-tuning + RF training
4. **Ensemble Combination**: Weighted voting
5. **Evaluation**: Cross-validation, accuracy metrics

### 2. Disaster Classifier

#### Architecture
- **Primary Model**: BERT with 4-class classification
- **Ensemble Model**: Random Forest with disaster-specific features
- **Classes**: wildfire, flood, hurricane, earthquake

#### Features
```python
# Disaster-Specific Features
- Disaster keywords per category
- Urgency indicators
- Location mentions
- Temporal patterns
- Severity indicators
```

### 3. Fact Checker

#### Architecture
- **News API Integration**: Real-time news verification
- **Keyword Analysis**: Credibility indicators
- **Location Verification**: Geocoding and consistency checks
- **Temporal Analysis**: Seasonal patterns and time consistency

#### Verification Methods
1. **News Source Verification**: Check against verified news sources
2. **Keyword Analysis**: Identify suspicious patterns
3. **Location Consistency**: Verify location information
4. **Temporal Consistency**: Check against seasonal patterns

---

## API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Analyze Tweet
```http
POST /analyze
Content-Type: application/json

{
  "text": "BREAKING: Major wildfire in California",
  "location": "California",
  "image_url": "optional_image_url"
}

Response:
{
  "tweet_id": "uuid",
  "fake_news_detection": {
    "prediction": "real",
    "confidence": 0.85,
    "explanation": "High confidence real disaster report"
  },
  "disaster_classification": {
    "type": "wildfire",
    "confidence": 0.92
  },
  "fact_checking": {
    "verified": true,
    "confidence": 0.78,
    "sources": [...]
  }
}
```

#### 2. Classify Disaster
```http
POST /classify
Content-Type: application/json

{
  "text": "Flooding reported in downtown area"
}

Response:
{
  "disaster_type": "flood",
  "confidence": 0.89,
  "explanation": "Contains flood-related keywords"
}
```

#### 3. Fact Check
```http
POST /fact-check
Content-Type: application/json

{
  "text": "Earthquake reported in San Francisco",
  "location": "San Francisco"
}

Response:
{
  "verified": true,
  "confidence": 0.82,
  "sources": [...],
  "explanations": [...]
}
```

#### 4. Get Authorities
```http
GET /authorities?location=California&disaster_type=wildfire

Response:
{
  "location": "California",
  "disaster_type": "wildfire",
  "authorities": [
    {
      "id": "fire_department",
      "name": "Local Fire Department",
      "phone": "911",
      "email": "emergency@fire.gov",
      "response_time": "5-10 minutes"
    }
  ]
}
```

#### 5. Contact Authority
```http
POST /contact-authority
Content-Type: application/json

{
  "authority_id": "fire_department",
  "message": "Major wildfire spreading rapidly",
  "location": "California",
  "disaster_type": "wildfire"
}

Response:
{
  "success": true,
  "message": "Alert sent successfully",
  "alert_id": "uuid"
}
```

---

## Database Schema

### Tables

#### 1. Analysis Results
```sql
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id TEXT UNIQUE,
    user_id TEXT,
    text TEXT,
    processed_text TEXT,
    fake_news_prediction TEXT,
    fake_news_confidence REAL,
    disaster_type TEXT,
    disaster_confidence REAL,
    fact_check_verified BOOLEAN,
    fact_check_confidence REAL,
    location TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 2. Authorities
```sql
CREATE TABLE authorities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    authority_id TEXT UNIQUE,
    name TEXT,
    type TEXT,
    phone TEXT,
    email TEXT,
    response_time TEXT,
    location_coverage TEXT
);
```

#### 3. Alerts
```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT UNIQUE,
    authority_id TEXT,
    message TEXT,
    location TEXT,
    disaster_type TEXT,
    status TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## Frontend Architecture

### Component Structure
```
src/
├── components/
│   ├── Navbar.js
│   ├── AnalysisForm.js
│   ├── ResultsDisplay.js
│   └── AuthorityContact.js
├── pages/
│   ├── Home.js
│   ├── Analyze.js
│   ├── Dashboard.js
│   ├── Authorities.js
│   └── About.js
├── services/
│   ├── api.js
│   └── utils.js
└── App.js
```

### State Management
- **Local State**: React hooks (useState, useEffect)
- **Server State**: React Query for API calls
- **Global State**: Context API (if needed)

### Key Features
- **Responsive Design**: Mobile-first approach
- **Real-time Updates**: WebSocket integration (planned)
- **Error Handling**: Comprehensive error boundaries
- **Loading States**: Skeleton loaders and progress indicators
- **Accessibility**: ARIA labels and keyboard navigation

---

## Deployment Guide

### Development Setup

#### Prerequisites
- Python 3.8+
- Node.js 14+
- Git

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Production Deployment

#### Environment Variables
```bash
# Backend
SECRET_KEY=your-production-secret-key
FLASK_ENV=production
DATABASE_URL=postgresql://user:pass@host:port/db

# API Keys
NEWS_API_KEY=your-news-api-key
TWITTER_API_KEY=your-twitter-api-key
GOOGLE_API_KEY=your-google-api-key

# Email/SMS
SMTP_SERVER=smtp.gmail.com
TWILIO_ACCOUNT_SID=your-twilio-sid
```

#### Docker Deployment
```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
```

---

## Testing Strategy

### Backend Testing
- **Unit Tests**: pytest for individual functions
- **Integration Tests**: API endpoint testing
- **Model Testing**: ML model accuracy validation
- **Performance Tests**: Load testing with locust

### Frontend Testing
- **Unit Tests**: Jest for component testing
- **Integration Tests**: React Testing Library
- **E2E Tests**: Cypress for user workflows

### Test Coverage Goals
- Backend: 80%+ coverage
- Frontend: 70%+ coverage
- Critical paths: 100% coverage

---

## Security Considerations

### Data Protection
- **Input Validation**: Comprehensive sanitization
- **SQL Injection**: Parameterized queries
- **XSS Prevention**: Content Security Policy
- **CSRF Protection**: Token-based validation

### API Security
- **Rate Limiting**: Prevent abuse
- **Authentication**: JWT tokens
- **Authorization**: Role-based access
- **HTTPS**: SSL/TLS encryption

### ML Model Security
- **Model Poisoning**: Input validation
- **Adversarial Attacks**: Robust model training
- **Privacy**: Data anonymization
- **Bias Detection**: Fairness metrics

---

## Performance Optimization

### Backend Optimization
- **Caching**: Redis for frequently accessed data
- **Database**: Connection pooling and indexing
- **Async Processing**: Celery for background tasks
- **Load Balancing**: Multiple server instances

### Frontend Optimization
- **Code Splitting**: Lazy loading of components
- **Bundle Optimization**: Tree shaking and minification
- **Caching**: Service workers for offline support
- **CDN**: Static asset delivery

### ML Model Optimization
- **Model Quantization**: Reduced model size
- **Batch Processing**: Efficient inference
- **GPU Acceleration**: CUDA support
- **Model Serving**: TensorFlow Serving

---

## Future Enhancements

### Planned Features
1. **Real-time Streaming**: WebSocket integration
2. **Mobile App**: React Native version
3. **Advanced Analytics**: Detailed reporting dashboard
4. **Multi-language Support**: Internationalization
5. **Advanced ML**: Transformer models and attention mechanisms

### Scalability Improvements
1. **Microservices**: Service-oriented architecture
2. **Cloud Deployment**: AWS/Azure/GCP integration
3. **Auto-scaling**: Kubernetes orchestration
4. **Global Distribution**: Multi-region deployment

---

## Conclusion

This comprehensive fake news detection system demonstrates the application of modern AI/ML technologies to solve real-world problems. The multimodal approach, combining text and image analysis with fact-checking, provides a robust solution for disaster-related misinformation detection.

The system's modular architecture allows for easy extension and maintenance, while the comprehensive testing strategy ensures reliability and accuracy. The focus on security and performance optimization makes it suitable for production deployment.

This project serves as an excellent capstone demonstration of full-stack development, machine learning, and system design skills. 