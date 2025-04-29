# preprocess_training.py
import pandas as pd
import numpy as np
import os
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_training(labeled_csv_path, output_dir="models"):
    # Load labeled data
    df = pd.read_csv(labeled_csv_path)

    # Relevant features
    features = [
        'steps', 'distance_km', 'calories', 'very_active_minutes',
        'moderately_active_minutes', 'lightly_active_minutes', 'sedentary_minutes'
    ]

    # Normalize
    scalers = {feature: MinMaxScaler() for feature in features}
    for feature in features:
        df[feature] = scalers[feature].fit_transform(df[[feature]])

    # Save scalers
    os.makedirs(f"{output_dir}/scalers", exist_ok=True)
    for feature, scaler in scalers.items():
        joblib.dump(scaler, f"{output_dir}/scalers/{feature}_scaler.pkl")

    # Create sequences and labels
    sequences, labels = [], []
    participants = df['participant_id'].unique()

    for pid in participants:
        p_data = df[df['participant_id'] == pid].sort_values(by='date_time')
        num_segments = len(p_data) // 7

        for i in range(num_segments):
            segment = p_data.iloc[i*7:(i+1)*7]
            feature_segment = segment[features].values
            label = segment['health_status'].mode()[0]  # majority label (safe)

            sequences.append(feature_segment)
            labels.append(label)

    X_raw = np.array(sequences)
    y_raw = np.array(labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Save label encoder
    with open(f"{output_dir}/health_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    return X_raw, y

if __name__ == "__main__":
    X_raw, y = preprocess_training("data/pmdata_labeled.csv")
    
    print(f"Training data ready: X shape = {X_raw.shape}, y shape = {y.shape}")