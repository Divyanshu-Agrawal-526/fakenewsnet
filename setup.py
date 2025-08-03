#!/usr/bin/env python3
"""
Setup script for Fake News Detection During Natural Disasters
A comprehensive capstone project with multimodal classification
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print project banner"""
    print("=" * 70)
    print("  FAKE NEWS DETECTION DURING NATURAL DISASTERS")
    print("  Final Year Capstone Project")
    print("  Multimodal Classification System")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_node_version():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js version: {result.stdout.strip()}")
            return True
        else:
            print("❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("❌ Node.js not found")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'backend/models/saved',
        'backend/uploads',
        'backend/data',
        'backend/logs',
        'frontend/src/components',
        'frontend/src/pages',
        'frontend/src/services',
        'docs',
        'tests',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_backend():
    """Setup Python backend"""
    print("\n🔧 Setting up Python Backend...")
    
    # Create virtual environment
    if not os.path.exists('backend/venv'):
        print("Creating Python virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'backend/venv'])
    
    # Activate virtual environment and install dependencies
    if platform.system() == "Windows":
        pip_path = "backend/venv/Scripts/pip"
        python_path = "backend/venv/Scripts/python"
    else:
        pip_path = "backend/venv/bin/pip"
        python_path = "backend/venv/bin/python"
    
    print("Installing Python dependencies...")
    subprocess.run([pip_path, 'install', '--upgrade', 'pip'])
    subprocess.run([pip_path, 'install', '-r', 'backend/requirements.txt'])
    
    print("✅ Backend setup completed")

def setup_frontend():
    """Setup React frontend"""
    print("\n🔧 Setting up React Frontend...")
    
    # Install Node.js dependencies
    print("Installing Node.js dependencies...")
    subprocess.run(['npm', 'install'], cwd='frontend')
    
    print("✅ Frontend setup completed")

def create_env_file():
    """Create .env file with sample configuration"""
    env_content = """# Backend Configuration
SECRET_KEY=your-secret-key-here-change-in-production
FLASK_ENV=development
FLASK_DEBUG=True

# Database Configuration
DATABASE_URL=sqlite:///fakenews.db

# API Keys (Get these from respective services)
NEWS_API_KEY=your-news-api-key
TWITTER_API_KEY=your-twitter-api-key
TWITTER_API_SECRET=your-twitter-api-secret
GOOGLE_API_KEY=your-google-api-key
OPENWEATHER_API_KEY=your-openweather-api-key

# Email Configuration (for authority alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Twilio Configuration (for SMS alerts)
TWILIO_ACCOUNT_SID=your-twilio-account-sid
TWILIO_AUTH_TOKEN=your-twilio-auth-token
TWILIO_PHONE_NUMBER=your-twilio-phone-number

# SendGrid Configuration (for email alerts)
SENDGRID_API_KEY=your-sendgrid-api-key
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with sample configuration")

def create_sample_data():
    """Create sample data for testing"""
    sample_tweets = [
        {
            "text": "BREAKING: Major wildfire spreading rapidly in California. Evacuation orders issued for multiple communities. #wildfire #emergency",
            "location": "California",
            "expected_result": "real"
        },
        {
            "text": "URGENT: Flooding reported in downtown area. Water levels rising rapidly. Please evacuate immediately. #flood #emergency",
            "location": "New York",
            "expected_result": "real"
        },
        {
            "text": "CLICK HERE to see the most amazing disaster photos! You won't believe what happened! #viral #share",
            "location": "Unknown",
            "expected_result": "fake"
        }
    ]
    
    import json
    with open('backend/data/sample_tweets.json', 'w') as f:
        json.dump(sample_tweets, f, indent=2)
    
    print("✅ Created sample data for testing")

def run_tests():
    """Run basic tests"""
    print("\n🧪 Running basic tests...")
    
    # Test backend
    try:
        result = subprocess.run([sys.executable, '-c', 'import flask; print("Flask imported successfully")'])
        if result.returncode == 0:
            print("✅ Backend dependencies test passed")
        else:
            print("❌ Backend dependencies test failed")
    except Exception as e:
        print(f"❌ Backend test error: {e}")
    
    # Test frontend
    try:
        result = subprocess.run(['npm', 'test', '--', '--watchAll=false'], cwd='frontend', capture_output=True)
        if result.returncode == 0:
            print("✅ Frontend tests passed")
        else:
            print("❌ Frontend tests failed")
    except Exception as e:
        print(f"❌ Frontend test error: {e}")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 70)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📋 Next Steps:")
    print("1. Configure API keys in the .env file")
    print("2. Start the backend server:")
    print("   cd backend")
    print("   python app.py")
    print("3. Start the frontend development server:")
    print("   cd frontend")
    print("   npm start")
    print("4. Open http://localhost:3000 in your browser")
    print("\n📚 Documentation:")
    print("- README.md contains detailed setup instructions")
    print("- API documentation available at /api/health when server is running")
    print("\n🔧 Development:")
    print("- Backend API runs on http://localhost:5000")
    print("- Frontend runs on http://localhost:3000")
    print("- Database file: backend/fakenews.db")
    print("\n⚠️  Important Notes:")
    print("- This is an academic project, not for commercial use")
    print("- API keys are required for full functionality")
    print("- Models will be downloaded on first run")
    print("=" * 70)

def main():
    """Main setup function"""
    print_banner()
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    check_python_version()
    if not check_node_version():
        print("❌ Node.js is required for the frontend")
        print("Please install Node.js from https://nodejs.org/")
        sys.exit(1)
    
    # Create project structure
    print("\n📁 Creating project structure...")
    create_directories()
    
    # Setup backend
    setup_backend()
    
    # Setup frontend
    setup_frontend()
    
    # Create configuration files
    print("\n⚙️  Creating configuration files...")
    create_env_file()
    create_sample_data()
    
    # Run tests
    run_tests()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 