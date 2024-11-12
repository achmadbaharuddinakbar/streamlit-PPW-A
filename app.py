import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify, render_template

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    # 1. Case Folding: Mengubah semua huruf menjadi huruf kecil
    text = text.lower()
    
    # 2. Menghilangkan angka dan karakter khusus
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca dan karakter khusus
    
    # 3. Tokenisasi: Memecah teks menjadi kata-kata
    words = word_tokenize(text)
    
    # 4. Menghapus stopwords: kata-kata umum yang tidak membawa banyak informasi
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]
    
    # Menggabungkan kata-kata kembali menjadi satu kalimat
    processed_text = ' '.join(words)
    
    return processed_text

# Memuat model dan TF-IDF vectorizer yang telah dilatih
model = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Rute untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')  # Pastikan Anda membuat file index.html di folder 'templates'

# Rute untuk memproses input teks
@app.route('/classify', methods=['POST'])
def classify_text():
    # Ambil input dari form
    user_input = request.form['text']
    
    if user_input:
        # Preprocessing input
        preprocessed_text = preprocess_text(user_input)

        # Transformasi teks menggunakan TF-IDF
        text_tfidf = tfidf.transform([preprocessed_text])

        # Melakukan prediksi
        prediction = model.predict(text_tfidf)
        predicted_category = "Kesehatan" if prediction[0] == "Kesehatan" else "Kuliner"

        # Mengembalikan hasil prediksi
        return jsonify({'result': f'Hasil Klasifikasi: {predicted_category}'})
    else:
        return jsonify({'error': 'Silakan masukkan teks berita terlebih dahulu.'})

# Jalankan aplikasi
if __name__ == "__main__":
    app.run(debug=True)