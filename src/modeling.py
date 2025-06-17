import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def train_evaluate_regression_model(model, X_train, y_train, X_test, y_test, model_name):
    """Trains and evaluates a regression model."""
    print(f"\n--- Training and Evaluating {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  RMSE: {rmse:.4f}")
    print(f"  R-squared: {r2:.4f}")
    return model, {'rmse': rmse, 'r2': r2}

def train_evaluate_classification_model(model, X_train, y_train, X_test, y_test, model_name):
    """Trains and evaluates a classification model."""
    print(f"\n--- Training and Evaluating {model_name} (Classification) ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability for positive class

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    return model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}