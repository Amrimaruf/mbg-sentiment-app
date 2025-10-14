import streamlit as st
import re, pickle, joblib, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

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

normalization_dict = {
    "gk": "tidak", "nggak": "tidak", "ga": "tidak", "bgt": "banget"
}
stopwords = set(["yang", "dan", "di", "ke", "dari", "untuk", "itu", "ini", "saya", "kami"])
stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    for slang, formal in normalization_dict.items():
        text = text.replace(slang, formal)
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
    return ' '.join(tokens)

def predict_sentiment(text):
    processed = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=maxlen, padding='post')
    prob = model.predict(padded)[0][0]
    label = le.inverse_transform([int(prob > 0.5)])[0]
    confidence = round(prob if label == le.classes_[1] else 1 - prob, 4)
    return label, confidence

st.title("🔮 Prediksi Sentimen Program MBG")
st.write("Masukkan kalimat di bawah untuk melihat prediksi sentimen:")

user_input = st.text_area("📝 Teks", height=150)

if st.button("Prediksi"):
    if user_input.strip():
        label, conf = predict_sentiment(user_input)
        st.markdown(f"### 📌 Prediksi: **{label}**")
        st.markdown(f"Confidence: `{conf * 100:.2f}%`")
    else:
        st.warning("Masukkan teks terlebih dahulu.")

