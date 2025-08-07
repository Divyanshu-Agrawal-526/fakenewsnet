import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os

def load_data():
    # Get the current directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of CSV files
    csv_files = [
        'preprocessed_tweets_florence.csv',
        'preprocessed_tweets.csv',
        'preprocessed_tweets_dorian.csv',
        'preprocessed_tweets_california.csv'
    ]
    
    all_texts = []
    all_labels = []
    label_map = {}
    
    # Load and process each CSV file
    for i, file in enumerate(csv_files):
        # Construct full path to the CSV file
        file_path = os.path.join(current_dir, file)
        
        if os.path.exists(file_path):
            print(f"\nLoading {file}...")
            try:
                # Read the full file
                df = pd.read_csv(file_path)
                
                # Use tweet_text column
                if 'tweet_text' not in df.columns:
                    print(f"Warning: 'tweet_text' column not found in {file}. Available columns: {df.columns.tolist()}")
                    continue
                
                # Add texts and labels
                all_texts.extend(df['tweet_text'].values)
                all_labels.extend([i] * len(df))
                
                # Store label mapping
                disaster_name = file.replace('preprocessed_tweets_', '').replace('.csv', '')
                label_map[i] = disaster_name
                
                print(f"Successfully loaded {len(df)} tweets from {file}")
                
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
        else:
            print(f"\nWarning: File not found: {file_path}")
    
    if not all_texts:
        raise ValueError("No data was loaded from any of the CSV files")
    
    return all_texts, all_labels, label_map

def main():
    # Load data
    texts, labels, label_map = load_data()
    
    print(f"\nNumber of classes: {len(label_map)}")
    print("Label mapping:")
    for idx, name in label_map.items():
        print(f"{idx}: {name}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train classifier
    print("\nTraining classifier...")
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test_tfidf)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(label_map.values())))
    
    print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.2f}")

if __name__ == "__main__":
    main() 