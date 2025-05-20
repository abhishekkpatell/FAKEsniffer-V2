import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fakesniffer_model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

# Constants
max_length = 54

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="🧠")
st.title("📰 Fake News Detector")
st.markdown("Enter a news headline below to check if it is **fake or real** using a Keras model trained on GloVe embeddings.")

input_text = st.text_area("News Headline:", height=100)

if st.button("Predict"):
    if input_text.strip():
        seq = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]
        label = "🟢 Real" if prediction < 0.5 else "🔴 Fake"
        confidence = 1 - prediction if prediction < 0.5 else prediction
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"Confidence: `{confidence:.2f}`")
    else:
        st.warning("Please enter a valid headline.")
