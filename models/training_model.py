import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import joblib
import os

# Download stopwords jika belum ada
nltk.download("stopwords")

print("Memulai pelatihan model...")

# Fungsi pembersihan teks
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Baca dataset
file = "tweets-data/gabungan_saham.csv"
df = pd.read_csv(file)

if 'full_text' in df.columns:
    # Hapus baris kosong
    df = df[['full_text']].dropna()

    # Tambahkan label dummy (karena tidak ada kolom 'sentiment')
    df['clean_text'] = df['full_text'].apply(clean_text)
    df['sentiment'] = ['positive' if i % 2 == 0 else 'negative' for i in range(len(df))]

    # Mapping label
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X = df['clean_text']
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Buat pipeline TF-IDF + SVM
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))

    print("Melatih model...")
    model.fit(X_train, y_train)

    # Simpan model ke folder models
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "sentiment_model.pkl")
    joblib.dump(model, model_path)

    print(f"Model berhasil disimpan di: {model_path}")
else:
    print("Kolom 'full_text' tidak ditemukan dalam dataset.")