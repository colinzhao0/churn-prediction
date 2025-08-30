import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_models(results):
    best_model = None
    best_score = 0
    
    for name, result in results.items():
        print(f"\n{name} Results:")
        print("=" * 50)
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"ROC-AUC: {result['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(result['classification_report'])
        print("\nConfusion Matrix:")
        print(result['confusion_matrix'])
        
        if result['roc_auc'] > best_score:
            best_score = result['roc_auc']
            best_model = result['model']
    
    return best_model

# Feature importance for tree-based models
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()