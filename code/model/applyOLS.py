import joblib
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
MODEL_PATH = "./code/model/OLS/ols_model.pkl"
SCALER_PATH = "./code/model/OLS/scaler.pkl"
ACTIVATION_DATA_PATH = "../../data/cleaned/activation_data.csv"

# Load the saved OLS model
ols_model = joblib.load(MODEL_PATH)

# Load activation data
activation_data = pd.read_csv(ACTIVATION_DATA_PATH)

# Preprocessing function
def handle_numeric_data(df):
    """Encodes categorical variables as numeric values."""
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# Preprocess activation data
activation_data = handle_numeric_data(activation_data)

# Extract the actual target value before removing it
TARGET_COLUMN = 'arrival_delay_m'
actual_value = activation_data[TARGET_COLUMN].values[0]

# Ensure activation data contains only feature columns (drop target column)
activation_features = activation_data.drop(columns=[TARGET_COLUMN])

# Load the trained scaler
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded successfully.")

# Scale activation data
X_activation_scaled = scaler.transform(activation_features)  # Do NOT fit again, only transform

# Add constant (as required by OLS)
X_activation_scaled = sm.add_constant(X_activation_scaled, has_constant='add')

# Make prediction
predicted_value = ols_model.predict(X_activation_scaled).flatten()[0]

# Print actual vs. predicted values
print(f"Actual Value: {actual_value:.2f}")
print(f"Predicted Value: {predicted_value:.2f}")