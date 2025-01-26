import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2


import feature_selection as sf

# load data
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


def build_model(train, test, target_column, epochs=50, batch_size=64):
    # Preprocessing
    X_train = train.drop(target_column, axis=1)
    y_train = train[target_column]

    X_test = test.drop(target_column, axis=1)
    y_test = test[target_column]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the ANN model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='linear'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation='sigmoid')  # Binary classification output
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Custom callbacks
    early_stopping = CustomEarlyStopping(patience=5)
    reduce_lr = CustomReduceLROnPlateau(factor=0.5, patience=3)

    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
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
    training_data = sf.feature_selection(handle_numeric_data(training_data_pre_selection), target_column="arrival_delay_m", k=5)
    training_data["rained"] = training_data_pre_selection["rained"]
    test_data = handle_numeric_data(test_data_pre_selection[training_data.columns])

    """    training_data = handle_numeric_data(training_data)
    test_data = handle_numeric_data(test_data)

    columns_to_drop_based_on_corr_matrix = ['state',
                                            'departure_plan_hour', 'departure_plan_datetime']

    training_data = training_data.drop(columns=columns_to_drop_based_on_corr_matrix)
    test_data = test_data.drop(columns=columns_to_drop_based_on_corr_matrix)"""

    print(training_data.columns)
    print(test_data.columns)

    model, history = build_model(training_data, test_data, 'arrival_delay_m', epochs=50, batch_size=64)
    plot_training_history(history)
    predictions = model.predict(test_data.drop(columns=['arrival_delay_m']))

    predicted_classes = (predictions > 0.5).astype(int)
