
import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model_2025.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-image: url("    https://images.app.goo.gl/KvjPB");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
# 📰 Fake News Detection App
Welcome! Enter a news article below to check if it's *Real* or *Fake*.
""")
st.markdown("---")

user_input = st.text_area("Enter a news article:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Vectorize and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.success("✅ This news article is *Real*.")
        else:
            st.error("❌ This news article is *Fake*.")
