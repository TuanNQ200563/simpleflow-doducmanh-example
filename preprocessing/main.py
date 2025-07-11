import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text) 
    text = re.sub(r"#\w+", "", text) 
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def main(data: pd.DataFrame):
    data_raw = data.copy()
    
    # Clean the text data
    data_raw["cleaned_text"] = data_raw["text"].astype(str).apply(clean_text)
    
    # Remove rows where cleaned_text is empty or only whitespace
    data_raw = data_raw[data_raw["cleaned_text"].str.strip() != ""]
    
    data_raw["label"] = data_raw["label"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        data_raw["cleaned_text"], data_raw["label"], test_size=0.2, random_state=42
    )
    
    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english"
    )
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    transformer = {
        "features": vectorizer,
        "target": None,
    }
    
    return X_train_vectorized, X_test_vectorized, y_train, y_test, transformer
