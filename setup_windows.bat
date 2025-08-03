@echo off
echo ======================================================================
echo   FAKE NEWS DETECTION DURING NATURAL DISASTERS
echo   Windows Setup Script
echo ======================================================================
echo.

echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Checking Node.js version...
node --version
if %errorlevel% neq 0 (
    echo Error: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

echo.
echo Creating project structure...
if not exist "backend\models\saved" mkdir "backend\models\saved"
if not exist "backend\uploads" mkdir "backend\uploads"
if not exist "backend\data" mkdir "backend\data"
if not exist "backend\logs" mkdir "backend\logs"
if not exist "frontend\src\components" mkdir "frontend\src\components"
if not exist "frontend\src\pages" mkdir "frontend\src\pages"
if not exist "frontend\src\services" mkdir "frontend\src\services"
if not exist "docs" mkdir "docs"
if not exist "tests" mkdir "tests"

echo.
echo Setting up Python Backend...
cd backend
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install Flask==2.3.3 Flask-CORS==4.0.0 Flask-SQLAlchemy==3.0.5
python -m pip install requests python-dotenv numpy pandas scikit-learn
python -m pip install nltk textblob geopy Pillow matplotlib seaborn plotly
python -m pip install beautifulsoup4 lxml pytest pytest-flask

echo Attempting to install TensorFlow...
python -m pip install tensorflow
if %errorlevel% neq 0 (
    echo Warning: TensorFlow installation failed - will use fallback models
)

echo Attempting to install PyTorch...
python -m pip install torch
if %errorlevel% neq 0 (
    echo Warning: PyTorch installation failed - will use fallback models
)

cd ..

echo.
echo Setting up React Frontend...
cd frontend
echo Installing Node.js dependencies...
npm install
if %errorlevel% neq 0 (
    echo Error: npm install failed
    echo Please check your Node.js installation
    pause
    exit /b 1
)
cd ..

echo.
echo Creating configuration files...
echo # Backend Configuration > .env
echo SECRET_KEY=your-secret-key-here-change-in-production >> .env
echo FLASK_ENV=development >> .env
echo FLASK_DEBUG=True >> .env
echo DATABASE_URL=sqlite:///fakenews.db >> .env
echo NEWS_API_KEY=your-news-api-key >> .env
echo TWITTER_API_KEY=your-twitter-api-key >> .env
echo GOOGLE_API_KEY=your-google-api-key >> .env

echo.
echo ======================================================================
echo 🎉 SETUP COMPLETED SUCCESSFULLY!
echo ======================================================================
echo.
echo 📋 Next Steps:
echo 1. Configure API keys in the .env file
echo 2. Start the backend server:
echo    cd backend
echo    venv\Scripts\python app.py
echo 3. Start the frontend development server:
echo    cd frontend
echo    npm start
echo 4. Open http://localhost:3000 in your browser
echo.
echo ======================================================================
pause 