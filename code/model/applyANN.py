import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model
model_path = "model_metrics/ANN/currentAiSolution.h5"
model = tf.keras.models.load_model(model_path)

# Load new data
activation_data_path = "../../data/clean/activation_data.csv"
activation_data = pd.read_csv(activation_data_path)

# Make predictions
predictions = model.predict(activation_data)

# Save predictions
activation_data["Predicted Delay"] = predictions
activation_data.to_csv("activation_data_with_predictions.csv", index=False)

print("Predictions saved to activation_data_with_predictions.csv")