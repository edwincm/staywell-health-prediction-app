# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pickle
import random
import requests
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

app = Flask(__name__)
load_dotenv()

GROQ_URL = os.getenv("GROQ_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = load_model("models/best_model.keras")
with open("models/health_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

FEATURES = [
    'steps', 'distance_km', 'calories', 
    'very_active_minutes', 'moderately_active_minutes',
    'lightly_active_minutes', 'sedentary_minutes'
]
NUM_DAYS = 7

# Route for recommendation
@app.route("/recommend", methods=["POST"])
def recommend():
    input_data = request.json  # Full POST JSON body

    # A readable week summary
    week_summary = ""
    for day in range(7):
        week_summary += f"Day {day + 1}:\n"
        for feature in ["steps", "distance_km", "calories", "very_active_minutes", "moderately_active_minutes", "lightly_active_minutes", "sedentary_minutes"]:
            key = f"{feature}_{day}"
            value = input_data.get(key, "N/A")
            week_summary += f"  {feature.replace('_', ' ').title()}: {value}\n"
        week_summary += "\n"

    # Create a detailed user prompt
    user_message = f"""
        Based on the following 7-day fitness data, provide personalized health recommendations to prevent chronic diseases.

        Fitness Data:
        {week_summary}

        Please suggest specific improvements in areas like:
        - Increasing steps
        - Reducing sedentary minutes
        - Improving very active minutes
        - Balancing calories
        - Any other lifestyle habits you notice

        Be positive, supportive, and practical.
        """
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a healthcare advisor AI. Based on a user's 7-day fitness data (steps, calories, active minutes, sedentary time), suggest personalized health improvements and lifestyle habits to prevent chronic diseases."},
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

    response = requests.post(GROQ_URL, headers=headers, json=payload)
    data = response.json()
    
    try:
        reply = data['choices'][0]['message']['content']
    except KeyError:
        print("❌ Error from Groq API:", data)
        return jsonify({"error": "Failed to get a valid AI response. Check logs."}), 500

    return jsonify({"reply": reply})


# Function to generate random but reasonable activity data
def generate_random_week():
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
    return random_week

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form.get("action")

        if action == "randomize":
            # Generate random data
            random_data = generate_random_week()
            return render_template("index.html", features=FEATURES, prediction=None, random_data=random_data)

        elif action == "predict":
            # Read data from form
            data = []
            for day in range(NUM_DAYS):
                day_data = []
                for feature in FEATURES:
                    key = f"{feature}_{day}"
                    value = request.form.get(key, type=float)
                    day_data.append(value)
                data.append(day_data)

            X_input = np.array(data).reshape(1, NUM_DAYS, len(FEATURES))

            # Predict
            pred_probs = model.predict(X_input)
            pred_label_idx = np.argmax(pred_probs, axis=1)[0]
            pred_label = label_encoder.inverse_transform([pred_label_idx])[0]

            return render_template("index.html", features=FEATURES, prediction=pred_label, random_data=data)

    # Default GET request
    return render_template("index.html", features=FEATURES, prediction=None, random_data=None)

if __name__ == "__main__":
    app.run(debug=True)