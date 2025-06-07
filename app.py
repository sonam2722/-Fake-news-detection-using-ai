import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load model
model = load_model("models/model.h5")

# Preprocessing
def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
    return " ".join(tokens)

st.title("📰 Fake News Detection App")
input_text = st.text_area("Enter a news article text")

if st.button("Predict"):
    if input_text:
        cleaned = clean_text(input_text)
        # Convert to same format as training (example placeholder)
        # In real use, apply the same tokenizer and sequence padding as in training
        st.warning("This is a placeholder. Update with tokenizer/padding logic.")
        prediction = model.predict(np.zeros((1, 100)))  # Placeholder
        label = "FAKE" if prediction[0][0] > 0.5 else "REAL"
        st.success(f"The news is predicted to be: **{label}**")
    else:
        st.error("Please enter some text.")
