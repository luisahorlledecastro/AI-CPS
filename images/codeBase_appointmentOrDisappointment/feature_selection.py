# load libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


# load data
df = pd.read_csv('../../data/cleaned/training_data.csv')


def handle_numeric_data(df):
    # Handle non-numeric data
    for col in df.columns:
        if df[col].dtype == 'object':
            # Label encoding for simplicity
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def engineer_columns(df):
    # weekend yes no
    # average delay based on hour and whether it's weekend or not
    # month

    df['is_weekend'] = df['day_of_week'] > 5
    df['month'] = pd.to_datetime(df['date']).dt.month

    average_departure_delay = df.groupby(['hour', 'is_weekend'])['departure_delay_m'].mean().reset_index()

    # Step 3: Merge the average delay back into the original dataset
    data = df.merge(average_departure_delay, on=['hour', 'is_weekend'], how='left')

    # Preview the result
    print(data)

    return df


def feature_selection(data, target_column, k=10):
    """
    Preprocess the data and perform feature selection.

    Parameters:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the target column.
        k (int): The number of top features to select (for univariate and mutual_info).

    Returns:
        pd.DataFrame: A dataset with only the selected features.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    print(f"Selected features using univariate method: {list(selected_features)}")
    return data[selected_features.to_list() + [target_column]]


def plot_correlation_matrix(data):
    # Compute the correlation matrix
    corr_matrix = data.corr()
    print(corr_matrix.columns)

    # Create a heatmap with labels
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm"
    )

    # Add axis labels and a title
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.title("Correlation Matrix")
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)   # Keep y-axis labels horizontal
    plt.show()


if __name__ == '__main__':
    df = handle_numeric_data(df)
    df = engineer_columns(df)
    print(df.head())
    selected_data_univariate = feature_selection(df, target_column="arrival_delay_m", k=5)
    plot_correlation_matrix(df)
