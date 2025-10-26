import streamlit as st
import re, pickle, joblib, json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==========================================
# 🔹 LOAD MODEL DAN TOKENIZER
# ==========================================
@st.cache_resource
def load_artefacts():
    model = load_model('Model/bilstm_best.keras')
    with open('Data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    le = joblib.load('Data/label_encoder.pkl')
    with open('Data/metadata.json', 'r') as f:
        meta = json.load(f)
    return model, tokenizer, le, meta['maxlen']

model, tokenizer, le, maxlen = load_artefacts()

# ==========================================
# 🔹 PREPROCESSING SAMA DENGAN DI COLAB
# ==========================================
normalization_dict = {
    "gk": "tidak", "nggak": "tidak", "ga": "tidak", "bgt": "banget",
    "mbg": "makan bergizi gratis", "programnya": "program"
}
stopwords = set(["yang", "dan", "di", "ke", "dari", "untuk", "itu", "ini", "saya", "kami"])
stemmer = StemmerFactory().create_stemmer()

def full_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Normalisasi slang
    for slang, formal in normalization_dict.items():
        text = text.replace(slang, formal)

    # Deteksi negasi (sama seperti Colab)
    neg_patterns = [
        r'tidak\s+(berhasil|membantu|puas|signifikan|berdampak|efektif|berguna|positif)',
        r'tidak\s+ada\s+(perubahan|manfaat|hasil)',
        r'tidak\s+(terasa|terlihat|terbukti|bermakna)',
        r'tidak\s+(merasakan|melihat|mendapatkan)\s+(dampak|manfaat|hasil)\s+(positif|nyata|signifikan)',
        r'tidak\s+(memberikan|menunjukkan)\s+(hasil|perubahan|efek)',
        r'tidak\s+(membawa|menghasilkan)\s+(kemajuan|manfaat|perbaikan)'
    ]
    for pattern in neg_patterns:
        text = re.sub(pattern, lambda m: 'neg_' + '_'.join(m.group(0).split()), text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens

# ==========================================
# 🔹 PREDIKSI (disamakan juga dengan Colab)
# ==========================================
def predict_sentiment(text, threshold=0.5):
    processed = [full_preprocess(text)]
    sequences = tokenizer.texts_to_sequences(processed)
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post')
    prob = model.predict(padded, verbose=0)[0][0]
    label = le.inverse_transform([int(prob > threshold)])[0]
    confidence = round(prob if label == le.classes_[1] else 1 - prob, 4)
    return label, confidence, prob

# ==========================================
# 🔹 STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Prediksi Sentimen MBG", page_icon="🔮")
st.title("🔮 Prediksi Sentimen Program Makan Bergizi Gratis")
st.write("Masukkan kalimat di bawah untuk melihat prediksi sentimen menggunakan model BiLSTM.")

# Sidebar threshold
st.sidebar.header("⚙️ Pengaturan")
threshold = st.sidebar.slider("Threshold Prediksi", 0.1, 0.9, 0.5, 0.05)
st.sidebar.write(f"Threshold aktif: `{threshold}`")

# Input teks
user_input = st.text_area("📝 Masukkan teks:", height=150)

if st.button("Prediksi"):
    if user_input.strip():
        label, conf, prob = predict_sentiment(user_input, threshold)

        if label.lower() == "positif":
            st.success(f"✅ Prediksi: **{label}** ({conf*100:.2f}% confidence)")
        else:
            st.error(f"❌ Prediksi: **{label}** ({conf*100:.2f}% confidence)")

        st.markdown(f"**Probabilitas positif:** `{prob:.4f}`")
        st.markdown(f"**Threshold:** `{threshold}`")
    else:
        st.warning("Masukkan teks terlebih dahulu.")

st.caption("Model BiLSTM – Analisis Sentimen Program Makan Bergizi Gratis")
