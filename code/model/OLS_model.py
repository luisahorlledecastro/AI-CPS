import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as stats

TRAINING_DATA_PATH = '../../data/cleaned/training_data.csv'
TEST_DATA_PATH = '../../data/cleaned/test_data.csv'
SAVE_PATH = "./code/model/OLS/"
TARGET_COLUMN = 'arrival_delay_m'

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def load_data(path):
    return pd.read_csv(path)

def handle_categorical_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_ols_model(X, y):
    '''
    Trains an Ordinary Least Squares (OLS) regression model.
    with:
    X (array-like): Feature matrix
    y (array-like): Target variable
    using fitted OLS model
    '''
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()

def evaluate_model(model, X, y_true):
    '''
    Evaluates the performance of the model using various metrics, such as: MAE, MSE, RMSE
    '''
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse, y_pred

def save_metrics(path, mae, mse, rmse):
    '''
    Saves the evaluation metrics to a text file.
    '''
    with open(os.path.join(path, "ols_metrics.txt"), "w") as f:
        f.write(f"Test MAE: {mae:.2f} minutes\n")
        f.write(f"Test MSE: {mse:.2f} minutes²\n")
        f.write(f"Test RMSE: {rmse:.2f} minutes\n")

def save_model_summary(path, model):
    '''
    '''
    with open(os.path.join(path, "ols_summary.txt"), "w") as f:
        f.write(str(model.summary()))

def plot_residuals(y_true, y_pred, save_path):
    '''
    Creates and saves a histogram plot of the model residuals.
    where:
    y_true (array-like): True target values
    y_pred (array-like): Predicted target values
    '''
    plt.figure(figsize=(8, 5))
    sns.histplot(y_true - y_pred, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_path, "residuals_distribution.png"))
    plt.close()

def plot_regression_scatter(y_true, y_pred, save_path):
    '''
    Creates and saves a scatter plot of actual vs predicted values.
    with arugements:
    y_true (array-like): True target values
    y_pred (array-like): Predicted target values
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Delay (minutes)")
    plt.ylabel("Predicted Delay (minutes)")
    plt.title("OLS Regression: Actual vs. Predicted Delay")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='dashed')
    plt.savefig(os.path.join(save_path, "regression_scatter.png"))
    plt.close()
    
def plot_residuals_vs_fitted(model, X, y_true, save_path):
    '''
    Creates and saves a residuals vs. fitted values plot.
    with arguments:
    model: Fitted OLS model
    X-Feature matrix
    y_true- True target values
    This plot helps to check for homoscedasticity and linearity assumptions.
    '''
    y_pred = model.predict(X)
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='dashed')
    plt.xlabel("Predicted Delay (minutes)")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Fitted Values")
    plt.savefig(os.path.join(save_path, "residuals_vs_fitted.png"))
    plt.close()

def plot_qq(model, X, y_true, save_path):
    '''
    Creates and saves a Q-Q(Quantile-Quantile)  plot of the model residuals.
    Helping to check if the residuals follow a normal distribution
    where:
    model: Fitted OLS model
    X - Feature matrix
    y_true- true target values
    '''
    y_pred = model.predict(X)
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")
    plt.savefig(os.path.join(save_path, "qq_plot.png"))
    plt.close()


def main():
    create_directory(SAVE_PATH)

    # Load and preprocess data
    training_data = load_data(TRAINING_DATA_PATH)
    test_data = load_data(TEST_DATA_PATH)

    training_data = handle_categorical_data(training_data)
    test_data = handle_categorical_data(test_data)

    X_train, y_train = preprocess_data(training_data, TARGET_COLUMN)
    X_test, y_test = preprocess_data(test_data, TARGET_COLUMN)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    joblib.dump(scaler, os.path.join(SAVE_PATH, "scaler.pkl"))

    # Train model
    ols_model = train_ols_model(X_train_scaled, y_train)
    joblib.dump(ols_model, os.path.join(SAVE_PATH, "ols_model.pkl"))

    # Evaluate model
    X_test_scaled = sm.add_constant(X_test_scaled)
    mae, mse, rmse, y_test_pred = evaluate_model(ols_model, X_test_scaled, y_test)

    # Save results
    save_metrics(SAVE_PATH, mae, mse, rmse)
    save_model_summary(SAVE_PATH, ols_model)
    plot_residuals(y_test, y_test_pred, SAVE_PATH)
    plot_regression_scatter(y_test, y_test_pred, SAVE_PATH)
    plot_residuals_vs_fitted(ols_model, X_test_scaled, y_test, SAVE_PATH)
    plot_qq(ols_model, X_test_scaled, y_test, SAVE_PATH)

    print(f"Test MAE: {mae:.2f}, \nTest MSE: {mse:.2f}, \nTest RMSE: {rmse:.2f}")
    print(f"Model and results saved in {SAVE_PATH}")

if __name__ == "__main__":
    main()
