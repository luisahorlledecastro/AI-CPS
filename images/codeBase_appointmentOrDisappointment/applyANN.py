import json
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Paths
MODEL_PATH = os.path.join(script_dir,"../knowledgeBase/currentAiSolution.keras")
FEATURE_METADATA_PATH = os.path.join(script_dir,"feature_columns.json")
ACTIVATION_DATA_PATH = os.path.join(script_dir,"../activationBase/activation_data.csv")

# Load the saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Load saved feature column names
with open(FEATURE_METADATA_PATH, "r") as f:
    feature_columns = json.load(f)

# Load activation data
activation_data = pd.read_csv(ACTIVATION_DATA_PATH)

# Preprocessing function
def handle_numeric_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# Preprocess activation data
activation_data = handle_numeric_data(activation_data)

activation_data_y = activation_data["arrival_delay_m"]

# Ensure activation data contains the same features as training data
activation_data = activation_data[feature_columns]

# Scale features using training data scaler
scaler = joblib.load("../knowledgeBase/scaler.pkl")
print("Scaler loaded successfully.")
X_activation_scaled = scaler.transform(activation_data[feature_columns])

# Make prediction
predicted_value = model.predict(X_activation_scaled).flatten()[0]

# Print actual vs. predicted
TARGET_COLUMN = 'arrival_delay_m'
print(f"Actual Value: {activation_data_y.values[0]:.2f}")
print(f"Predicted Value: {predicted_value:.2f}")