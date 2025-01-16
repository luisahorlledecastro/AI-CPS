import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import pandas as pd
import matplotlib.pyplot as plt


# load data
training_data = pd.read_csv('../../data/cleaned/training_data.csv')
test_data = pd.read_csv('../../data/cleaned/test_data.csv')


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

# Custom ReduceLROnPlateau Callback
class CustomReduceLROnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, factor=0.5, patience=3, min_lr=1e-6):
        super(CustomReduceLROnPlateau, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                    print(f"\nEpoch {epoch + 1}: Reducing learning rate to {new_lr}.")
                self.wait = 0


def handle_numeric_data(df):
    # Handle non-numeric data
    for col in df.columns:
        if df[col].dtype == 'object':
            # Label encoding for simplicity
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


from imblearn.over_sampling import SMOTE

def handle_numeric_data_with_smote(df, target_column):
    """
    Handle non-numeric data and apply SMOTE for oversampling the minority class.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): The name of the target column.

    Returns:
        X_resampled (pd.DataFrame): Resampled features.
        y_resampled (pd.Series): Resampled target.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle non-numeric data
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled


def build_model_with_smote(train, test, target_column, epochs=50, batch_size=64):
    """
    Build, train, and evaluate an ANN model using SMOTE for balancing the dataset.

    Parameters:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Testing data.
        target_column (str): The name of the target column.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        model: Trained Keras model.
        history: Training history.
    """
    # Handle non-numeric data and apply SMOTE
    X_train_resampled, y_train_resampled = handle_numeric_data_with_smote(train, target_column)

    # Preprocess test data
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]

    for col in X_test.columns:
        if X_test[col].dtype == 'object':
            le = LabelEncoder()
            X_test[col] = le.fit_transform(X_test[col].astype(str))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Build the ANN model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification output
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train_scaled, y_train_resampled,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    return model, history


def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()


if __name__ == '__main__':
    print(training_data[training_data["state"] == "Brandenburg"]['arrival_delay_m'].value_counts())

    model, history = build_model_with_smote(training_data, test_data, 'arrival_delay_m')
    plot_training_history(history)
    predictions = model.predict(test_data.drop(columns=['arrival_delay_m']))

    predicted_classes = (predictions > 0.5).astype(int)
