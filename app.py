import streamlit as st
import re, pickle, joblib, json, base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import TweetTokenizer

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
        background: rgba(0,0,0,0.55);
        z-index: 0;
    }}

    .stApp > * {{
        position: relative;
        z-index: 1;
    }}

    /* Header putih */
    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
    }}

    /* FIX: Text area → background putih solid, teks hitam */
    .stTextArea textarea {{
        background: rgba(255,255,255,0.95) !important;
        color: black !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        border: 2px solid rgba(255,255,255,0.5) !important;
    }}

    /* Caption (Model BiLSTM...) diperjelas */
    .stMarkdown small, .stCaption, .stFooter {{
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
    }}

    /* Tombol */
    .stButton button {{
        background-color: rgba(255, 255, 255, 0.85) !important;
        color: black !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        border: none !important;
    }}

    .stButton button:hover {{
        background-color: black !important;
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
# === Load kamus normalisasi ===
kamus = pd.read_csv("Data/kamusnormalisasi.csv")
kamus_dict = dict(zip(kamus["salah"], kamus["benar"]))

# === Tokenizer sama seperti training ===
tweet_tokenizer = TweetTokenizer(
    preserve_case=False,
    strip_handles=False,
    reduce_len=True
)

# === Preprocessing FINAL (tanpa stopword & tanpa stemming) ===
def full_preprocess(text):
    if pd.isnull(text):
        return []

    # 1. Cleansing
    text = text.strip()
    text = re.sub(r'@\w+', '', text)                      # username
    text = re.sub(r'http\S+|www\S+', '', text)            # URL
    text = re.sub(r'#\w+', '', text)                      # hashtag
    text = re.sub(r'\d+', '', text)                       # angka
    text = re.sub(r'[^\w\s]', '', text)                   # tanda baca
    text = re.sub(r'\b[b-hj-zB-HJ-Z]\b', '', text)        # huruf tunggal
    text = re.sub(r'[^a-zA-Z\s]', '', text)               # non-huruf
    text = re.sub(r'\s+', ' ', text).strip()              # spasi berlebih

    # 2. Case folding
    text = text.lower()

    # 3. Tokenizing
    tokens = tweet_tokenizer.tokenize(text)

    # 4. Normalisasi slang
    tokens = [kamus_dict.get(t, t) for t in tokens]

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












