import streamlit as st
import re, pickle, joblib, json, base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==========================================
# 🔹 FUNGSI TAMBAHAN: BACKGROUND IMAGE
# ==========================================
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    /* Optional: buat efek gelap biar textnya lebih jelas */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.40); /* Atur opacity di sini */
        z-index: 0;
    }}

    /* Biar semua konten tetap di atas overlay */
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ==========================================
# 🔹 PAGE CONFIG DAN BACKGROUND
# ==========================================
st.set_page_config(page_title="Prediksi Sentimen MBG", page_icon="🔮")

# Ganti path sesuai lokasi gambar kamu
add_bg_from_local("assets/mbg.jpg")


# ==========================================
# 🔹 LOAD MODEL DAN TOKENIZER
# ==========================================
@st.cache_resource
def load_artefacts():
    model = load_model('Model/bilstm_model.keras')
    with open('Data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    le = joblib.load('Data/label_encoder.pkl')
    with open('Data/metadata.json', 'r') as f:
        meta = json.load(f)
    return model, tokenizer, le, meta['maxlen']

model, tokenizer, le, maxlen = load_artefacts()


# ==========================================
# 🔹 PREPROCESSING (SAMA DENGAN COLAB)
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

    return tokens


# ==========================================
# 🔹 PREDIKSI
# ==========================================
def predict_sentiment(text):
    processed = [full_preprocess(text)]
    sequences = tokenizer.texts_to_sequences(processed)
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post')
    prob = model.predict(padded, verbose=0)[0][0]
    label = le.inverse_transform([int(prob > 0.5)])[0]
    return label


# ==========================================
# 🔹 STREAMLIT UI
# ==========================================
st.title("🔮 Prediksi Sentimen Program Makan Bergizi Gratis")
st.write("Masukkan kalimat di bawah untuk melihat prediksi sentimen menggunakan model BiLSTM:")

user_input = st.text_area("📝 Teks", height=150)

if st.button("Prediksi"):
    if user_input.strip():
        label = predict_sentiment(user_input)
        if label.lower() == "positif":
            st.success("✅ Prediksi: **Positif**")
        else:
            st.error("❌ Prediksi: **Negatif**")
    else:
        st.warning("Masukkan teks terlebih dahulu.")

st.caption("Model BiLSTM – Analisis Sentimen Program Makan Bergizi Gratis")



