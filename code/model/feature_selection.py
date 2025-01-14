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


# load data
df = pd.read_csv('../../data/cleaned/training_data.csv')

print(df.head())


def handle_numeric_data(df):
    # Handle non-numeric data
    for col in df.columns:
        if df[col].dtype == 'object':
            # Label encoding for simplicity
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def feature_selection(data, target_column, method="univariate", k=10):
    """
    Preprocess the data and perform feature selection.

    Parameters:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the target column.
        method (str): The method for feature selection ('univariate', 'tree_based', 'mutual_info').
        k (int): The number of top features to select (for univariate and mutual_info).

    Returns:
        pd.DataFrame: A dataset with only the selected features.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Feature selection
    if method == "univariate":
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
    elif method == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
    elif method == "tree_based":
        model = ExtraTreesClassifier()
        model.fit(X, y)
        importance = model.feature_importances_
        selected_features = X.columns[np.argsort(importance)[-k:]]
        X_new = X[selected_features]
    else:
        raise ValueError("Invalid method. Choose 'univariate', 'tree_based', or 'mutual_info'.")

    print(f"Selected features using {method} method: {list(selected_features)}")
    return data[selected_features.to_list() + [target_column]]


def plot_correlation_matrix(data):
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()


if __name__ == '__main__':
    df = handle_numeric_data(df)
    selected_data = feature_selection(df, target_column="arrival_delay_m", method="univariate", k=5)
    plot_correlation_matrix(df)
    print(df.columns)
