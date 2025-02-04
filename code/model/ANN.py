import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving scalers
import feature_selection as sf

# Load data
TRAINING_DATA_PATH = '../../data/cleaned/training_data.csv'
TEST_DATA_PATH = '../../data/cleaned/test_data.csv'

training_data_pre_selection = pd.read_csv(TRAINING_DATA_PATH)
test_data_pre_selection = pd.read_csv(TEST_DATA_PATH)

# Preprocessing function
def handle_numeric_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# Build regression model function
def build_model(train, test, target_column, epochs=50, batch_size=64):
    X_train = train.drop(target_column, axis=1)
    y_train = train[target_column]  # Keep the actual delay in minutes

    X_test = test.drop(target_column, axis=1)
    y_test = test[target_column]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "scaler.pkl")

    # Build Regression Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation=None)  # Regression: No activation function
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',  # MSE for regression
                  metrics=['mae', 'mse'])  # Regression metrics

    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_mae, test_mse = model.evaluate(X_test_scaled, y_test, verbose=1)
    print(f"Test MAE: {test_mae:.2f}, Test MSE: {test_mse:.2f}")

    return model, history, X_test_scaled, y_test


# Save model function
def save_model(model):
    model.save("model_metrics/ANN/currentAiSolution.h5")  # Saved as .h5 format
    print(f"Model saved at currentAiSolution.h5")


# Save training metrics
def save_training_metrics(history):
    metrics_path = "model_metrics/ANN/training_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Final Training MAE: {history.history['mae'][-1]}\n")
        f.write(f"Final Validation MAE: {history.history['val_mae'][-1]}\n")
        f.write(f"Final Training MSE: {history.history['mse'][-1]}\n")
        f.write(f"Final Validation MSE: {history.history['val_mse'][-1]}\n")

    print(f"Training metrics saved at {metrics_path}")


# Plot and save training history
def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Model Mean Absolute Error')
    plt.savefig("model_metrics/ANN/training_mae.png")
    plt.show()

    plt.figure()
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Model Mean Squared Error')
    plt.savefig("model_metrics/ANN/training_mse.png")
    plt.show()

    print("Training history plots saved.")

# Scatter plot for regression predictions
def plot_regression_results(y_true, y_pred):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Delay (minutes)")
    plt.ylabel("Predicted Delay (minutes)")
    plt.title("Actual vs. Predicted Delay")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='dashed')  # Perfect fit line
    plt.savefig("model_metrics/ANN/regression_scatter.png")
    plt.show()
    print("Regression scatter plot saved.")


if __name__ == '__main__':
    # Apply feature selection
    training_data = sf.feature_selection(
        handle_numeric_data(training_data_pre_selection),
        target_column="arrival_delay_m", k=5
    )
    training_data["rained"] = training_data_pre_selection["rained"]
    test_data = handle_numeric_data(test_data_pre_selection[training_data.columns])

    model, history, X_test_scaled, y_test = build_model(training_data, test_data, 'arrival_delay_m', epochs=50,
                                                        batch_size=32)

    # Evaluate on the test set
    test_loss, test_mae, test_mse = model.evaluate(X_test_scaled, y_test, verbose=1)
    print(f"\nFinal Test MAE: {test_mae:.2f}")
    print(f"Final Test MSE: {test_mse:.4f}")

    # Store test performance
    test_metrics_path = "model_metrics/ANN/test_metrics.txt"
    with open(test_metrics_path, "w") as f:
        f.write(f"Test MAE: {test_mae:.2f}\n")
        f.write(f"Test MSE: {test_mse:.4f}\n")

    print(f"Test metrics saved at {test_metrics_path}")

    # Make Predictions on Test Data
    predictions = model.predict(X_test_scaled)

    # Save Regression Scatter Plot
    plot_regression_results(y_test, predictions)

    # Save Model & Metrics
    save_model(model)
    save_training_metrics(history)
    plot_training_history(history)
