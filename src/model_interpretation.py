import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Needed for displaying feature names

def plot_shap_summary(model, X_transformed, feature_names, plot_type='bar', max_display=10):
    """
    Generates and plots SHAP summary for model interpretability.
    Args:
        model: Trained model (e.g., XGBoost, RandomForest).
        X_transformed (pd.DataFrame or np.array): Preprocessed feature data.
        feature_names (list): List of feature names corresponding to X_transformed columns.
        plot_type (str): 'bar' for mean absolute SHAP value, 'dot' for summary plot.
        max_display (int): Number of top features to display.
    """
    print("\n--- Generating SHAP Summary Plot ---")
    explainer = shap.Explainer(model, X_transformed) # Or shap.TreeExplainer for tree models
    shap_values = explainer(X_transformed)

    # Set feature names if X_transformed is a numpy array
    if isinstance(X_transformed, np.ndarray) and feature_names:
        shap_values.feature_names = feature_names

    plt.figure(figsize=(10, 6))
    if plot_type == 'bar':
        shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False, max_display=max_display)
    elif plot_type == 'dot':
        shap.summary_plot(shap_values, X_transformed, show=False, max_display=max_display)
    plt.title(f"SHAP Feature Importance ({plot_type.capitalize()} Plot)")
    plt.tight_layout()
    plt.show()

    # You can also return raw SHAP values for further analysis
    return explainer, shap_values

# Example function to get feature importance from tree-based models (simpler)
def get_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importances = pd.Series(importances, index=feature_names)
        return feature_importances.nlargest(top_n)
    return None