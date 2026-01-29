import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, 'data', 'test_file.tsv')
model_path = os.path.join(base_dir, 'tfidf_model.joblib')

def train_and_save():
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return

    df = pd.read_csv(data_path, sep='\t')
    documents = df['text'].astype(str) # Adjust column name as needed

    # 2. Train Model
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(documents)
    
    # 3. Save Model
    # This serializes the object to a file
    joblib.dump(vectorizer, model_path)
    print(f"✅ Model successfully saved to: {model_path}")

if __name__ == "__main__":
    train_and_save()