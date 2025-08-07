# Model Training Guide

This guide explains how to train your fake news detection models using the datasets in your repository.

## 📊 Available Datasets

Your repository contains three major datasets:

### 1. **FakeNewsNet Dataset** (`data/fakenewsnet_dataset/`)
- **Purpose**: Fake news detection training
- **Files**: 
  - `gossipcop_fake.csv` - Fake news from GossipCop
  - `gossipcop_real.csv` - Real news from GossipCop
  - `politifact_fake.csv` - Fake news from PolitiFact
  - `politifact_real.csv` - Real news from PolitiFact
- **Features**: `id`, `news_url`, `title`, `tweet_ids`
- **Labels**: Binary (0 = Real, 1 = Fake)

### 2. **CrisisNLP Dataset** (`data/crisisnlp_dataset/`)
- **Purpose**: Disaster classification training
- **Events**: 
  - California Wildfires 2018
  - Hurricane Dorian 2019
  - Hurricane Florence 2018
  - Kerala Floods 2018
  - Midwestern US Floods 2019
  - Pakistan Earthquake 2019
- **Features**: Tweet text, disaster type, location
- **Labels**: Disaster categories (wildfire, hurricane, flood, earthquake)

### 3. **CrisisMMD Dataset** (`data/CrisisMMD/`)
- **Purpose**: Multimodal disaster analysis
- **Tasks**:
  - **Informative vs Not Informative**: Classify if content is relevant
  - **Humanitarian Categories**: 8 categories (affected individuals, infrastructure damage, etc.)
  - **Damage Severity**: 3 levels (severe, mild, little/no damage)
- **Features**: Text + Images
- **Format**: TSV files with train/dev/test splits

## 🚀 Training Options

### Option 1: Local Training

1. **Test the setup**:
   ```bash
   python test_training.py
   ```

2. **Run training locally**:
   ```bash
   cd backend
   python train_models.py
   ```

3. **Check results**:
   - Models saved in: `backend/models/saved_models/`
   - Training log: `backend/training.log`
   - Report: `backend/models/saved_models/training_report.json`

### Option 2: GitHub Actions (Automated)

1. **Push to GitHub**: The training workflow runs automatically
2. **Schedule**: Runs every Sunday at 2 AM UTC
3. **Manual trigger**: Available in GitHub Actions tab
4. **Artifacts**: Trained models uploaded as artifacts

### Option 3: Cloud Training

You can also train on cloud platforms:

- **Google Colab**: Upload datasets and run training script
- **AWS/GCP**: Use larger compute resources
- **Hugging Face**: Use their training infrastructure

## 📋 Training Pipeline

The training script (`backend/train_models.py`) handles:

1. **Dataset Loading**: Automatically loads all three datasets
2. **Data Preprocessing**: Cleans and prepares data for training
3. **Model Training**: Trains three models:
   - **Fake News Detector**: Binary classification
   - **Disaster Classifier**: Multi-class disaster classification
   - **Multimodal Classifier**: Text + image analysis
4. **Model Saving**: Saves trained models with metadata
5. **Report Generation**: Creates comprehensive training reports

## 🔧 Configuration

### Environment Setup
```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Dataset Paths
The training script expects datasets in:
- `data/fakenewsnet_dataset/`
- `data/crisisnlp_dataset/`
- `data/CrisisMMD/`

### Model Output
Trained models are saved to:
- `backend/models/saved_models/fake_news_detector/`
- `backend/models/saved_models/disaster_classifier/`
- `backend/models/saved_models/multimodal_classifier/`

## 📊 Expected Results

### Fake News Detection
- **Dataset**: ~10,000+ samples from FakeNewsNet
- **Accuracy**: 85-95% (depending on model)
- **Use Case**: Detect fake news during disasters

### Disaster Classification
- **Dataset**: ~5,000+ samples from CrisisNLP
- **Categories**: wildfire, hurricane, flood, earthquake
- **Accuracy**: 80-90%
- **Use Case**: Classify disaster types from tweets

### Multimodal Analysis
- **Dataset**: ~2,000+ samples from CrisisMMD
- **Tasks**: Informative classification, humanitarian categories, damage assessment
- **Accuracy**: 75-85%
- **Use Case**: Analyze both text and images from disaster tweets

## 🐛 Troubleshooting

### Common Issues

1. **Dataset not found**:
   - Check file paths in `data/` directory
   - Ensure CSV/TSV files are properly formatted

2. **Memory issues**:
   - Reduce batch size in model configurations
   - Use smaller model architectures
   - Train on cloud with more RAM

3. **Import errors**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

4. **Training failures**:
   - Check training logs in `training.log`
   - Verify dataset quality and format
   - Ensure sufficient training data

### Debug Commands

```bash
# Test dataset loading
python test_training.py

# Check dataset sizes
python -c "import pandas as pd; print(pd.read_csv('data/fakenewsnet_dataset/gossipcop_fake.csv').shape)"

# Verify model imports
python -c "from backend.models.fake_news_detector import FakeNewsDetector; print('OK')"
```

## 📈 Monitoring Training

### Local Monitoring
- **Logs**: Check `backend/training.log`
- **Progress**: Real-time console output
- **Results**: JSON report in `backend/models/saved_models/training_report.json`

### GitHub Actions Monitoring
- **Workflow**: Check Actions tab in GitHub
- **Artifacts**: Download trained models from workflow runs
- **Logs**: View detailed logs in Actions

## 🔄 Continuous Training

The GitHub Actions workflow enables:

1. **Scheduled Training**: Weekly automatic retraining
2. **Triggered Training**: Manual runs when needed
3. **Model Versioning**: Each run creates new model versions
4. **Performance Tracking**: Compare results across runs

## 📝 Next Steps

After training:

1. **Test Models**: Use the trained models in your Flask API
2. **Deploy**: Integrate models into your web application
3. **Monitor**: Track model performance in production
4. **Iterate**: Improve models based on real-world usage

## 🤝 Contributing

To improve the training pipeline:

1. **Add new datasets**: Update `DatasetManager` class
2. **Improve models**: Modify model architectures
3. **Add metrics**: Include additional evaluation metrics
4. **Optimize**: Improve training efficiency and speed

---

**Note**: This training setup is designed for academic/research purposes. For production use, consider additional security, validation, and monitoring measures. 