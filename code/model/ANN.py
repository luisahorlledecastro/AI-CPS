import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import feature_selection as sf
import scipy.stats as stats

# Load data
TRAINING_DATA_PATH = '../../data/cleaned/training_data.csv'
TEST_DATA_PATH = '../../data/cleaned/test_data.csv'
ACTIVATION_DATA_PATH = '../../data/cleaned/activation_data.csv'

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
        tf.keras.layers.Dense(1, activation=None)
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

def plot_residuals_ann(y_true, y_pred, save_path):
    '''
    Creates and saves a histogram plot of the model residuals.
    Args:
    y_true (array-like): True target values
    y_pred (array-like): Predicted target values
    save_path (str): Path to save the plot
    '''
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig(os.path.join(save_path, "residuals_distribution.png"))
    plt.close()

def plot_regression_scatter_ann(y_true, y_pred, save_path):
    '''
    Creates and saves a scatter plot of actual vs predicted values.
    Args:
    y_true (array-like): True target values
    y_pred (array-like): Predicted target values
    save_path (str): Path to save the plot
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Regression Scatter: Actual vs Predicted")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='dashed')
    plt.show()
    plt.savefig(os.path.join(save_path, "regression_scatter.png"))
    plt.close()

def plot_residuals_vs_fitted_ann(model, X, y_true, save_path):
    '''
    Creates and saves a residuals vs fitted values plot for an ANN model.
    Args:
    model: Trained ANN model
    X: Feature matrix (scaled as per the training process)
    y_true (array-like): True target values
    save_path (str): Path to save the plot
    '''
    y_pred = model.predict(X).flatten()
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='dashed')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted Values")
    plt.show()
    plt.savefig(os.path.join(save_path, "residuals_vs_fitted.png"))
    plt.close()

def plot_qq_ann(model, X, y_true, save_path):
    '''
    Creates and saves a Q-Q plot of the model residuals for an ANN model.
    Args:
    model: Trained ANN model
    X: Feature matrix (scaled as per the training process)
    y_true (array-like): True target values
    save_path (str): Path to save the plot
    '''
    y_pred = model.predict(X).flatten()
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Normal Q-Q Plot")
    plt.show()
    plt.savefig(os.path.join(save_path, "qq_plot.png"))
    plt.close()


if __name__ == '__main__':
    # Apply feature selection
    training_data = sf.feature_selection(
        handle_numeric_data(training_data_pre_selection),
        target_column="arrival_delay_m", k=5
    )
    training_data["rained"] = training_data_pre_selection["rained"]
    test_data = handle_numeric_data(test_data_pre_selection[training_data.columns])
    activation_data = handle_numeric_data(test_data[training_data.columns])


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

    y_pred = model.predict(X_test_scaled).flatten()

    save_path = "model_metrics/ANN/"

    plot_residuals_ann(y_test, y_pred, save_path)
    plot_regression_scatter_ann(y_test, y_pred, save_path)
    plot_residuals_vs_fitted_ann(model, X_test_scaled, y_test, save_path)
    plot_qq_ann(model, X_test_scaled, y_test, save_path)