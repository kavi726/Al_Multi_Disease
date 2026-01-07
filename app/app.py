import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="AI Multi Disease Detection", layout="centered")
st.title("🧠 AI-Based Multi Disease Detection System")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models
models = {
    "heart": joblib.load(os.path.join(BASE_DIR, "models", "heart.pkl")),
    "diabetes": joblib.load(os.path.join(BASE_DIR, "models", "diabetes.pkl")),
    "liver": joblib.load(os.path.join(BASE_DIR, "models", "liver.pkl")),
    "cancer": joblib.load(os.path.join(BASE_DIR, "models", "cancer.pkl")),
}

def gen_ai_explanation(disease, prediction, confidence):
    if prediction == 1:
        return (
            f"The AI system predicts a HIGH risk of {disease} with {confidence}% confidence. "
            f"This suggests abnormal clinical indicators. Medical consultation and lifestyle "
            f"modifications are strongly advised."
        )
    else:
        return (
            f"The AI system predicts a LOW risk of {disease} with {confidence}% confidence. "
            f"Most indicators are within normal range. Maintaining a healthy lifestyle is recommended."
        )

# UI
disease = st.selectbox("Select Disease", list(models.keys()))
input_data = st.text_input("Enter feature values (comma separated)")

if st.button("Predict"):
    try:
        values = [float(x.strip()) for x in input_data.split(",")]
        model = models[disease]
        X = np.array(values).reshape(1, -1)

        prediction = int(model.predict(X)[0])
        confidence = round(float(model.predict_proba(X).max() * 100), 2)

        explanation = gen_ai_explanation(disease, prediction, confidence)

        st.success("Prediction Completed")
        st.write(f"**Prediction:** {'Disease Detected' if prediction else 'No Disease'}")
        st.write(f"**Confidence:** {confidence}%")
        st.write(f"**Explanation:** {explanation}")

    except Exception as e:
        st.error(f"Input error: {e}")
