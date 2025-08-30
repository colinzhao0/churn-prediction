import pandas as pd
import numpy as np
from data.data_preprocessing import preprocess_data
from model_training import train_models, hyperparameter_tuning, save_model
from utils import evaluate_models, plot_feature_importance

def main():
    # Load or create data
    print("Loading data...")
    # Replace this with your actual data loading
    df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv') 
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(df)
    
    print("Training models...")
    results = train_models(X_train, X_test, y_train, y_test)
    
    print("Evaluating models...")
    best_model = evaluate_models(results)
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    tuned_model = hyperparameter_tuning(X_train, y_train)
    
    # Save the best model
    save_model(tuned_model, 'models/best_churn_model.pkl')
    save_model(scaler, 'models/scaler.pkl')
    
    # Plot feature importance
    feature_names = X_train.columns.tolist()
    plot_feature_importance(tuned_model, feature_names)
    
    print("Project completed successfully!")

if __name__ == "__main__":
    main()