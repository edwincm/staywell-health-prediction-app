# app.py
import streamlit as st
import numpy as np
import pickle
import random
import requests
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Setup
st.set_page_config(page_title="StayWell - Health Status Predictor", page_icon="🏥", layout="centered")
load_dotenv()

# Load model and assets
model = load_model("models/best_model.keras")
with open("models/health_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

GROQ_URL = os.getenv("GROQ_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

FEATURES = [
    'steps', 'distance_km', 'calories', 
    'very_active_minutes', 'moderately_active_minutes',
    'lightly_active_minutes', 'sedentary_minutes'
]
NUM_DAYS = 7

# Title
st.title("🏥 StayWell - 7-Day Health Status Predictor")

# Session State initialization
if "week_data" not in st.session_state:
    st.session_state.week_data = np.zeros((NUM_DAYS, len(FEATURES)))

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "recommendation" not in st.session_state:
    st.session_state.recommendation = None

# Function to randomize reasonable data
def randomize_week_data():
    random_week = []
    for _ in range(NUM_DAYS):
        day = [
            random.randint(2000, 18000),          # steps
            round(random.uniform(1.0, 15.0), 2),  # distance_km
            round(random.uniform(1800, 4000), 2), # calories
            random.randint(0, 120),               # very_active_minutes
            random.randint(0, 100),               # moderately_active_minutes
            random.randint(50, 400),              # lightly_active_minutes
            random.randint(300, 1000)              # sedentary_minutes
        ]
        random_week.append(day)
    st.session_state.week_data = np.array(random_week)

# Function to clear data
def clear_week_data():
    st.session_state.week_data = np.zeros((NUM_DAYS, len(FEATURES)))
    st.session_state.prediction = None
    st.session_state.recommendation = None

# Function to predict
def predict_health():
    X_input = st.session_state.week_data.reshape(1, NUM_DAYS, len(FEATURES))
    pred_probs = model.predict(X_input)
    pred_idx = np.argmax(pred_probs, axis=1)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    st.session_state.prediction = pred_label

# Function to get AI recommendation
def get_recommendation():
    week_summary = ""
    for day in range(NUM_DAYS):
        week_summary += f"Day {day + 1}:\n"
        for idx, feature in enumerate(FEATURES):
            value = st.session_state.week_data[day, idx]
            week_summary += f"  {feature.replace('_', ' ').title()}: {value}\n"
        week_summary += "\n"

    user_message = f"""
    Based on the following 7-day fitness data, provide personalized health recommendations to prevent chronic diseases.

    Fitness Data:
    {week_summary}

    Focus on specific improvements:
    - Increasing steps
    - Reducing sedentary minutes
    - Improving very active minutes
    - Balancing calories
    - Any other helpful suggestions
    """

    payload = {
        "model": "llama-3.3-70b-versatile",  # Make sure this is the right model name!
        "messages": [
            {"role": "system", "content": "You are a healthcare advisor AI."},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.2,
        "max_tokens": 500,
        "top_p": 1,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload)
        data = response.json()

        if 'choices' in data:
            reply = data['choices'][0]['message']['content']
            st.session_state.recommendation = reply
        else:
            error_message = data.get('error', {}).get('message', 'Unknown error from AI service.')
            st.error(f"❌ AI Error: {error_message}")
            st.session_state.recommendation = None

    except Exception as e:
        st.error(f"❌ Failed to contact AI service: {e}")
        st.session_state.recommendation = None

# Main App Logic
st.subheader("📝 Fill or Randomize 7-Day Activity Data")

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Form-like input
for day in range(NUM_DAYS):
    cols = st.columns(len(FEATURES) + 1)
    cols[0].markdown(f"**{days_of_week[day]}**")
    for idx, feature in enumerate(FEATURES):
        st.session_state.week_data[day, idx] = cols[idx + 1].number_input(
            label=feature,
            value=float(st.session_state.week_data[day, idx]),
            step=1.0,
            key=f"{feature}_{day}"
        )

# Action buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Randomize"):
        randomize_week_data()
        st.rerun()

with col2:
    if st.button("🧹 Clear Form"):
        clear_week_data()
        st.rerun()

with col3:
    if st.button("✅ Predict Health Status"):
        predict_health()

# Prediction Result
if st.session_state.prediction:
    st.subheader(f"🧬 Prediction: :green[{st.session_state.prediction}]")

    # Get AI recommendation
    if st.button("💬 Get Lifestyle Recommendations"):
        with st.spinner("Generating advice..."):
            get_recommendation()

# AI Recommendation Output
if st.session_state.recommendation:
    st.subheader("📋 AI Recommendations")
    st.markdown(st.session_state.recommendation)