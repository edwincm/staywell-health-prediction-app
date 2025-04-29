# 🏥 StayWell - Smart Health Status Prediction App

StayWell is a web app that predicts a person's health status ("Healthy" or "At Risk") based on their 7-day fitness activity data using a deep learning model (CNN-LSTM). It also provides personalized AI lifestyle recommendations to prevent chronic diseases.

---

## ✨ Features

- 📝 Input 7 days of fitness data
- 🔄 Randomize data with realistic ranges
- ✅ Predict health risk using CNN-LSTM model
- 🤖 Get personalized AI health tips via Groq API
- 📈 Easy-to-use, interactive interface (Streamlit)

---

## 🛠 Technologies Used

- Python
- Streamlit
- TensorFlow (Keras)
- Scikit-learn
- Groq API (for AI recommendations)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/your-username/StayWell.git
cd StayWell
pip install -r requirements.txt
streamlit run app.py