import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving scalers
import feature_selection as sf


# Load data
training_data_pre_selection = pd.read_csv('../../data/cleaned/training_data.csv')
test_data_pre_selection = pd.read_csv('../../data/cleaned/test_data.csv')

# Custom EarlyStopping Callback
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=5):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if current_loss < self.best:
            self.best = current_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

# Preprocessing function
def handle_numeric_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# Build model function
def build_model(train, test, target_column, epochs=50, batch_size=64):
    X_train = train.drop(target_column, axis=1)
    y_train = train[target_column]

    X_test = test.drop(target_column, axis=1)
    y_test = test[target_column]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "scaler.pkl")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stopping = CustomEarlyStopping(patience=5)

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    return model, history, X_test_scaled, y_test

# Save model function
def save_model(model):
    model.save("currentAiSolution.h5")  # Saved as .h5 format
    print(f"Model saved at currentAiSolution.h5")

# Save training metrics
def save_training_metrics(history):
    metrics_path = "training_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Final Training Loss: {history.history['loss'][-1]}\n")
        f.write(f"Final Validation Loss: {history.history['val_loss'][-1]}\n")
        f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]}\n")
        f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]}\n")

    print(f"Training metrics saved at {metrics_path}")

# Plot and save training history
def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.savefig("training_accuracy.png")

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.savefig("training_loss.png")

    print("Training history plots saved.")

# Plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved.")

# Scatter plot
def plot_scatter(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Scatter Plot")
    plt.savefig("scatter_plot.png")
    print("Scatter plot saved.")

if __name__ == '__main__':
    training_data = sf.feature_selection(
        handle_numeric_data(training_data_pre_selection),
        target_column="arrival_delay_m", k=5
    )
    training_data["rained"] = training_data_pre_selection["rained"]
    test_data = handle_numeric_data(test_data_pre_selection[training_data.columns])

    model, history, X_test_scaled, y_test = build_model(training_data, test_data, 'arrival_delay_m', epochs=200, batch_size=8)

    save_model(model)
    save_training_metrics(history)
    plot_training_history(history)

    # Predictions
    predictions = model.predict(X_test_scaled)
    predicted_classes = (predictions > 0.5).astype(int)

    # Save additional plots
    plot_confusion_matrix(y_test, predicted_classes)
    plot_scatter(y_test, predictions)

    
#  TODO confusion matrix makes no sense
