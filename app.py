# app.py

import streamlit as st
import pandas as pd
import re
import yfinance as yf
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

# Unduh dataset stopwords dari NLTK
nltk.download("stopwords")

# -------------------------------
# 1. Load CSS External
# -------------------------------
def local_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# -------------------------------
# 2. Fungsi Pembersihan Teks
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):  # Pastikan input string
        return ""
    text = text.lower()  # Huruf kecil semua
    text = re.sub(r'@\w+', '', text)  # Hapus mention Twitter
    text = re.sub(r'http\S+|www.\S+', '', text)  # Hapus URL
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    words = text.split()
    stop_words = set(stopwords.words("english"))  # Kata-kata yang diabaikan
    words = [word for word in words if word not in stop_words]  # Filter stop words
    return " ".join(words)  # Gabung kembali menjadi kalimat

# -------------------------------
# 3. Fungsi Analisis Sentimen
# -------------------------------
def get_sentiment(text):
    analysis = TextBlob(text)  # Analisis menggunakan TextBlob
    score = analysis.sentiment.polarity  # Dapatkan skor polaritas
    return "positive" if score > 0.1 else "negative"  # Jika lebih besar dari 0.1 → positif

# -------------------------------
# 4. Aplikasi Streamlit
# -------------------------------
st.markdown("<h1 style='text-align: center;'>📈 Aplikasi Analisis Sentimen & Prediksi Saham</h1>", unsafe_allow_html=True)

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV dengan kolom 'full_text'", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Pastikan kolom full_text tersedia
    if 'full_text' in df.columns:
        df = df[['full_text']].dropna()  # Hanya ambil kolom yang dibutuhkan
        df['clean_text'] = df['full_text'].apply(clean_text)  # Bersihkan teks
        df['sentiment'] = df['clean_text'].apply(get_sentiment)  # Analisis sentimen

        st.success("Analisis selesai!")
        
        # Tampilkan hasil dalam kartu
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Hasil Analisis Sentimen")
        st.write(df[['full_text', 'sentiment']].head())
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------------
        # 5. Visualisasi Prediksi Harga Saham Berdasarkan Waktu
        # -------------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📉 Prediksi Harga Saham Berdasarkan Waktu")

        ticker_input = st.text_input("Masukkan Ticker Saham (misal: AAPL)", value="AAPL")
        if st.button("Tampilkan Harga Saham"):
            with st.spinner("Mengambil data harga saham..."):
                try:
                    # Ambil data historis dari Yahoo Finance
                    stock_df = yf.download(ticker_input, start="2020-01-01", end="2025-01-01")

                    if stock_df.empty:
                        st.error("Tidak ada data untuk ticker ini. Coba ticker lain.")
                    else:
                        # Plot grafik harga penutupan
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(stock_df.index, stock_df['Close'], label='Harga Penutupan', color="#2980b9")
                        ax.set_title(f'Harga Saham {ticker_input} Sejak 2020')
                        ax.set_xlabel('Tanggal')
                        ax.set_ylabel('Harga (USD)')
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.5)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Gagal mengambil data saham: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Kolom 'full_text' tidak ditemukan dalam dataset.")

else:
    st.info("Silakan upload file CSV untuk memulai analisis.")