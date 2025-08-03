# Quick Start Guide - Windows

## 🚀 Fast Setup for Windows Users

### Option 1: Automated Setup (Recommended)

1. **Run the Windows batch file**:
   ```cmd
   setup_windows.bat
   ```

2. **Start the backend** (in a new command prompt):
   ```cmd
   cd backend
   venv\Scripts\python app_simple.py
   ```

3. **Start the frontend** (in another command prompt):
   ```cmd
   cd frontend
   npm start
   ```

4. **Open your browser** and go to: http://localhost:3000

### Option 2: Manual Setup

#### Step 1: Backend Setup
```cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install Flask==2.3.3 Flask-CORS==4.0.0 Flask-SQLAlchemy==3.0.5
pip install requests python-dotenv numpy pandas scikit-learn
pip install nltk textblob geopy Pillow matplotlib seaborn plotly
pip install beautifulsoup4 lxml pytest pytest-flask
```

#### Step 2: Frontend Setup
```cmd
cd frontend
npm install
```

#### Step 3: Start the Application
```cmd
# Terminal 1 - Backend
cd backend
venv\Scripts\python app_simple.py

# Terminal 2 - Frontend
cd frontend
npm start
```

## 🔧 Troubleshooting

### Common Issues:

1. **"npm not found"**
   - Install Node.js from https://nodejs.org/
   - Restart your command prompt after installation

2. **"python not found"**
   - Install Python from https://python.org/
   - Make sure to check "Add Python to PATH" during installation

3. **TensorFlow/PyTorch installation fails**
   - The simple version (`app_simple.py`) works without these
   - Use the simple version for testing

4. **Port already in use**
   - Backend: Change port in `app_simple.py` (line 200)
   - Frontend: Use `npm start -- --port 3001`

## 📋 What's Included

### Simple Version Features:
- ✅ Fake news detection (rule-based)
- ✅ Disaster classification (rule-based)
- ✅ Fact checking (simulated)
- ✅ Authority contact (simulated)
- ✅ Dashboard with statistics
- ✅ Modern React UI

### Full Version Features (requires ML libraries):
- 🤖 Advanced ML models (BERT, CNN)
- 🔍 Real fact checking with news APIs
- 📍 Real geocoding and location services
- 📧 Real email/SMS alerts

## 🎯 Testing the System

### Sample Tweets to Test:

**Real Disaster Tweet:**
```
"BREAKING: Major wildfire spreading rapidly in California. Evacuation orders issued for multiple communities. #wildfire #emergency"
```

**Fake News Tweet:**
```
"CLICK HERE to see the most amazing disaster photos! You won't believe what happened! #viral #share"
```

**Flood Tweet:**
```
"URGENT: Flooding reported in downtown area. Water levels rising rapidly. Please evacuate immediately. #flood #emergency"
```

## 📁 Project Structure

```
fakenews/
├── backend/
│   ├── app_simple.py          # Simple version (no ML dependencies)
│   ├── app.py                 # Full version (with ML)
│   ├── requirements.txt       # Python dependencies
│   └── venv/                 # Virtual environment
├── frontend/
│   ├── src/                  # React source code
│   ├── public/               # Static files
│   └── package.json          # Node.js dependencies
├── setup_windows.bat         # Windows setup script
└── QUICK_START.md           # This file
```

## 🔗 API Endpoints

When the backend is running, you can test these endpoints:

- `GET http://localhost:5000/` - Home page
- `GET http://localhost:5000/health` - Health check
- `POST http://localhost:5000/api/analyze` - Analyze tweet
- `GET http://localhost:5000/api/statistics` - System stats

## 🎓 Academic Project

This is a **final year capstone project** demonstrating:
- Full-stack web development
- Machine learning implementation
- API design and integration
- Modern UI/UX design
- System architecture

**Note**: This is for academic purposes only, not for commercial use.

## 🆘 Need Help?

If you encounter issues:

1. **Check the console** for error messages
2. **Verify Python and Node.js** are installed correctly
3. **Try the simple version** first (`app_simple.py`)
4. **Check ports** are not in use (5000, 3000)

## 🚀 Next Steps

Once the basic system is running:

1. **Configure API keys** in `.env` file for full functionality
2. **Install ML libraries** for advanced features
3. **Customize the models** for your specific needs
4. **Add real data sources** for production use

---

**Happy Coding! 🎉** 