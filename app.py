# app.py

import streamlit as st
import pickle
import string
import dill
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Download NLTK resources (only first run)
# -------------------------------
nltk.download('punkt')
nltk.download('stopwords')

# -------------------------------
# Setup Preprocessing
# -------------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def transform_text(text):
    # 1. Lowercase + tokenize
    tokens = nltk.word_tokenize(text.lower())

    # 2. Keep only alphanumeric
    tokens = [t for t in tokens if t.isalnum()]

    # 3. Remove stopwords + punctuation
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]

    # 4. Apply stemming
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)

# -------------------------------
# Load Tokenizer
# -------------------------------
# Load the tokenizer (no need to define it again)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------------
# Best Thresholds
# -------------------------------
BEST_THRESHOLDS = {
    "GRU": 0.76414174,
    "LSTM": 0.3130121,
    "RNN": 0.36876982
}

# -------------------------------
# Load Models
# -------------------------------
models = {
    "GRU": load_model("gru_model.keras"),
    "LSTM": load_model("lstm_model.keras"),
    "RNN": load_model("rnn_model.keras")
}

# -------------------------------
# Prediction Function
# -------------------------------
def predict_raw_auto(text, model_name, max_len=400):
    model = models[model_name]
    threshold = BEST_THRESHOLDS[model_name]

    # Preprocess with your NLTK function
    cleaned = transform_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Predict
    prob = model.predict(pad, verbose=0)[0][0]
    pred = 1 if prob >= threshold else 0

    return pred, float(prob), threshold

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📩 Spam Detection App")
st.write("Enter a message below, select a model, and check if it’s **Spam (0)** or **Ham (1)**.")

# Text input
user_text = st.text_area("✍️ Enter your message:", "")

# Model selector
model_choice = st.selectbox("🧠 Choose Model", ["GRU", "LSTM", "RNN"])

# Predict button
if st.button("🔮 Predict"):
    if user_text.strip():
        pred, prob, th = predict_raw_auto(user_text, model_choice)
        label = "✅ Ham (1)" if pred == 1 else "🚨 Spam (0)"

        st.markdown(f"### Prediction: {label}")
        st.write(f"**Model:** {model_choice}")
        st.write(f"**Probability:** {prob:.4f}")
        st.write(f"**Threshold:** {th:.3f}")
    else:
        st.warning("⚠️ Please enter some text before predicting.")



