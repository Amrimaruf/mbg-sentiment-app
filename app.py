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
    /* Background image */
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    /* Overlay gelap biar teks lebih jelas */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.55); /* atur opacity */
        z-index: 0;
    }}

    /* Semua elemen di atas overlay */
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}

    /* Biar header (title) putih */
    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: white !important;
    }}

    /* Style untuk text area (card translucent) */
    .stTextArea textarea {{
        background: rgba(255, 255, 255, 0.25) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 15px !important;
        backdrop-filter: blur(6px) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }}

    /* Button styling lebih clean */
    .stButton button {{
        background-color: rgba(255, 255, 255, 0.85) !important;
        color: black !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        border: none !important;
    }}

    .stButton button:hover {{
        background-color: white !important;
        color: black !important;
        transform: scale(1.02);
        transition: 0.2s;
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
    probs = model.predict(padded, verbose=0).flatten()
    labels = le.inverse_transform((probs > 0.5).astype("int32"))
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







