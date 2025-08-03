#!/usr/bin/env python3
"""
Windows-specific setup script for Fake News Detection During Natural Disasters
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
    print("  Windows Setup Script")
    print("=" * 70)
    print()

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def find_npm():
    """Find npm executable on Windows"""
    possible_paths = [
        r"C:\Program Files\nodejs\npm.cmd",
        r"C:\Program Files (x86)\nodejs\npm.cmd",
        "npm.cmd",
        "npm"
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Found npm: {path}")
                return path
        except FileNotFoundError:
            continue
    
    print("❌ npm not found. Please install Node.js from https://nodejs.org/")
    return None

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
    
    # Install dependencies
    pip_path = "backend/venv/Scripts/pip"
    print("Installing Python dependencies...")
    
    try:
        subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([pip_path, 'install', '-r', 'backend/requirements.txt'], check=True)
        print("✅ Backend setup completed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Python dependencies: {e}")
        print("Trying with individual packages...")
        
        # Install packages individually to handle compatibility issues
        packages = [
            'Flask==2.3.3',
            'Flask-CORS==4.0.0',
            'Flask-SQLAlchemy==3.0.5',
            'requests>=2.31.0',
            'python-dotenv>=1.0.0',
            'numpy>=1.24.3',
            'pandas>=2.0.3',
            'scikit-learn>=1.3.0',
            'nltk>=3.8.1',
            'textblob>=0.17.1',
            'geopy>=2.3.0',
            'Pillow>=10.0.0',
            'matplotlib>=3.7.2',
            'seaborn>=0.12.2',
            'plotly>=5.16.1',
            'beautifulsoup4>=4.12.2',
            'lxml>=4.9.3',
            'pytest>=7.4.2',
            'pytest-flask>=1.2.0',
        ]
        
        for package in packages:
            try:
                subprocess.run([pip_path, 'install', package], check=True)
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"⚠️  Failed to install {package} - continuing...")
        
        # Try to install TensorFlow separately
        try:
            subprocess.run([pip_path, 'install', 'tensorflow>=2.15.0'], check=True)
            print("✅ Installed TensorFlow")
        except subprocess.CalledProcessError:
            print("⚠️  TensorFlow installation failed - will use fallback models")
        
        try:
            subprocess.run([pip_path, 'install', 'torch>=2.0.1'], check=True)
            print("✅ Installed PyTorch")
        except subprocess.CalledProcessError:
            print("⚠️  PyTorch installation failed - will use fallback models")

def setup_frontend():
    """Setup React frontend"""
    print("\n🔧 Setting up React Frontend...")
    
    npm_path = find_npm()
    if not npm_path:
        print("❌ Cannot proceed without npm")
        return False
    
    try:
        print("Installing Node.js dependencies...")
        subprocess.run([npm_path, 'install'], cwd='frontend', check=True)
        print("✅ Frontend setup completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Node.js dependencies: {e}")
        return False

def create_env_file():
    """Create .env file"""
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
    """Create sample data"""
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

def print_next_steps():
    """Print next steps"""
    print("\n" + "=" * 70)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📋 Next Steps:")
    print("1. Configure API keys in the .env file")
    print("2. Start the backend server:")
    print("   cd backend")
    print("   venv\\Scripts\\python app.py")
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
    
    # Create project structure
    print("\n📁 Creating project structure...")
    create_directories()
    
    # Setup backend
    setup_backend()
    
    # Setup frontend
    frontend_success = setup_frontend()
    
    # Create configuration files
    print("\n⚙️  Creating configuration files...")
    create_env_file()
    create_sample_data()
    
    # Print next steps
    print_next_steps()
    
    if not frontend_success:
        print("\n⚠️  Frontend setup failed. Please install Node.js and run:")
        print("   cd frontend")
        print("   npm install")

if __name__ == "__main__":
    main() 