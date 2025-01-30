import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class TrainDelayOLS:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.regularization_model = None

    def preprocess_data(self, df):
        '''Preprocess data for regression analysis.'''
        df_processed = df.copy()

        # Convert categorical variables using label encoding
        categorical_columns = ['station', 'state', 'city']
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                else:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])

        # Create time-based features
        df_processed['month'] = pd.to_datetime(df_processed['date']).dt.month
        df_processed['peak_hour'] = df_processed['departure_plan_hour'].apply(lambda x: 1 if 7 <= x <= 9 or 17 <= x <= 19 else 0)

        # Convert boolean to int
        df_processed['rained'] = df_processed['rained'].astype(int)

        # Create interaction terms and polynomial features
        df_processed['temp_rain_interaction'] = df_processed['temperature'] * df_processed['rained']
        df_processed['temperature_squared'] = df_processed['temperature'] ** 2
        df_processed['category_temperature'] = df_processed['category'] * df_processed['temperature']

        # Standardize numerical features to reduce numerical instability
        numeric_features = ['temperature', 'temperature_squared', 'temp_rain_interaction', 'category_temperature']
        df_processed[numeric_features] = self.scaler.fit_transform(df_processed[numeric_features])

        # Select features for the model
        features = ['temperature', 'rained', 'month', 'departure_plan_hour', 'category', 'day_of_week',
                    'temp_rain_interaction', 'temperature_squared', 'peak_hour', 'category_temperature']

        X = df_processed[features]
        return X, features

    def fit(self, X, y, use_regularization=False, regularization_type='ridge'):
        if use_regularization:
            if regularization_type == 'ridge':
                alphas = np.logspace(-6, 6, 13)
                self.regularization_model = RidgeCV(alphas=alphas, cv=5)
            

            elif regularization_type == 'lasso':
                alphas = np.logspace(-6, 6, 13)
                self.regularization_model = LassoCV(alphas=alphas, cv=5)
            elif regularization_type == 'elasticnet':
                alphas = np.logspace(-6, 6, 13)
                self.regularization_model = ElasticNetCV(alphas=alphas, cv=5)

            self.regularization_model.fit(X, y)
            coef = self.regularization_model.coef_
            intercept = self.regularization_model.intercept_
            print(f"{regularization_type.capitalize()} regression coefficients:", coef)
            print("Intercept:", intercept)
            return {
            'summary': f"{regularization_type.capitalize()} regression coefficients: {coef}\nIntercept: {intercept}",
            'model': self.regularization_model
            }
        else:
            # Add constant for intercept
            X_const = sm.add_constant(X)

            # Fit OLS model
            self.model = sm.OLS(y, X_const).fit()

            # Calculate VIF
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X_const.columns
            vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

            # Perform heteroskedasticity test
            _, p_value, _, _ = het_breuschpagan(self.model.resid, X_const)

            # Calculate Durbin-Watson statistic
            dw_statistic = durbin_watson(self.model.resid)

            return {
                'summary': self.model.summary(),
                'vif': vif_data,
                'heteroskedasticity_p_value': p_value,
                'durbin_watson': dw_statistic
            }

    def predict(self, X):
        '''Make predictions using the fitted model.'''
        X_processed, _ = self.preprocess_data(X)
        if self.regularization_model:
            return self.regularization_model.predict(X_processed)
        else:
            X_const = sm.add_constant(X_processed)
            return self.model.predict(X_const)

    def plot_diagnostics(self, X, y):
        '''Plot diagnostic plots for the OLS model.'''
        plt.figure(figsize=(15, 12))

        # Residuals vs Fitted
        plt.subplot(221)
        plt.scatter(self.model.fittedvalues, self.model.resid, alpha=0.5, s=10)
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted')
        plt.axhline(y=0, color='r', linestyle='--')
        sns.regplot(x=self.model.fittedvalues, y=self.model.resid, lowess=True, scatter_kws={'s': 10}, line_kws={'color': 'red', 'lw': 1})

        # Q-Q plot
        plt.subplot(222)
        sm.graphics.qqplot(self.model.resid, line='45', fit=True, ax=plt.gca(), markersize=5)
        plt.title('Q-Q Plot')

        # Scale-Location
        plt.subplot(223)
        plt.scatter(self.model.fittedvalues, np.sqrt(np.abs(self.model.resid)), alpha=0.5, s=10)
        plt.xlabel('Fitted values')
        plt.ylabel('√|Residuals|')
        plt.title('Scale-Location')
        sns.regplot(x=self.model.fittedvalues, y=np.sqrt(np.abs(self.model.resid)), lowess=True, scatter_kws={'s': 10}, line_kws={'color': 'red', 'lw': 1})

        # Residuals vs Leverage
        plt.subplot(224)
        ax = plt.gca()
        sm.graphics.influence_plot(self.model, criterion="cooks", ax=ax, marker='o', markersize=2, alpha=0.75)
        plt.title('Residuals vs Leverage')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def save_model(self, filename='ols_model.pkl'):
        #Save the trained model.
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)


def main():
    # Load data
    train_data = pd.read_csv('C:/Users/dita_/Desktop/AIBAS/teamwork/AI-CPS-1/data/cleaned/training_data.csv')
    test_data = pd.read_csv('C:/Users/dita_/Desktop/AIBAS/teamwork/AI-CPS-1/data/cleaned/test_data.csv')

    # Initialize model
    ols_model = TrainDelayOLS()
    X_train, features = ols_model.preprocess_data(train_data)
    y_train = train_data['arrival_delay_m']

    # Fit OLS model and get diagnostics
    ols_diagnostics = ols_model.fit(X_train, y_train, use_regularization=False)

    print("OLS Model Summary:")
    print(ols_diagnostics['summary'])
    print("\nVariance Inflation Factors:")
    print(ols_diagnostics['vif'])
    print(f"\nHeteroskedasticity test p-value: {ols_diagnostics['heteroskedasticity_p_value']}")
    print(f"Durbin-Watson statistic: {ols_diagnostics['durbin_watson']}")

    # Plot OLS diagnostics
    ols_model.plot_diagnostics(X_train, y_train)

    # Fit Ridge regression model
    ridge_diagnostics = ols_model.fit(X_train, y_train, use_regularization=True, regularization_type='ridge')

    print("\nRidge Regression Results:")
    print(ridge_diagnostics['summary'])

    # Make predictions and calculate metrics for both models
    X_test, _ = ols_model.preprocess_data(test_data)
    y_test = test_data['arrival_delay_m']

    ols_predictions = ols_model.predict(test_data)
    ridge_predictions = ridge_diagnostics['model'].predict(X_test)

    for model_name, predictions in [("OLS", ols_predictions), ("Ridge", ridge_predictions)]:
        mse = np.mean((y_test - predictions) ** 2)
        r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        print(f"\n{model_name} Test MSE: {mse}")
        print(f"{model_name} Test R-squared: {r2}")

    # Save the OLS model
    ols_model.save_model()

if __name__ == '__main__':
    main()


'''
reminder results:

OLS Test MSE: 1.0403326032959934
OLS Test R-squared: 0.05369457075611883

improved from the last commit'''