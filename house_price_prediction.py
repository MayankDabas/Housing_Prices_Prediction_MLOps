"""
This script performs the following tasks:
1. Loads and preprocesses the California Housing dataset.
2. Supports hyperparameter tuning for a RandomForestRegressor using command-line arguments.
3. Evaluates the model's performance on a validation set.
4. Loads the best hyperparameters from a file and trains the final model.
5. Saves the final model to a file for future use.

The script allows for both hyperparameter tuning and final model training, making it 
flexible for different stages of model development.
"""
import argparse
import itertools
import joblib
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


housing_dataset = fetch_california_housing(as_frame=True)

features = housing_dataset.data
target_value = housing_dataset.target

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)


def parse_arguments():
    """
    Parse command-line arguments for Random Forest hyperparameters.
    """
    parser = argparse.ArgumentParser(description='Random Forest Hyperparameter Tuning')

    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the tree')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum number of samples required to split an internal node')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='Minimum number of samples required to be at a leaf node')

    args = parser.parse_args()
    return args

def load_best_hyperparameters(file_path='best_hyperparams.json'):
    """
    Load the best hyperparameters from the given JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Best hyperparameters file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        best_hyperparams = json.load(f)
    
    return best_hyperparams

def split_dataset(features, target_value, test_size, seed):
    """
    Split the dataset into train and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, target_value, test_size=test_size, random_state=seed
    )

    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, y_train, X_val, y_val, args):
    """
    Train and evaluate the RandomForest model with given hyperparameters.
    """
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42
    )

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)

    print(f"R² score on Validation Set: {r2}")
    
    hyperparams = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf
    }

    print(f"Hyperparameters: {json.dumps(hyperparams)}")

    with open('results.txt', 'w') as f:
        f.write(f'Hyperparameters: {json.dumps(hyperparams)}\n')
        f.write(f'R² score on Validation Set: {r2}\n')

def train_final_model(features, target_value, best_hyperparams):
    """
    Train the final RandomForest model using the best hyperparameters.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, target_value, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=best_hyperparams['n_estimators'],
        max_depth=best_hyperparams['max_depth'],
        min_samples_split=best_hyperparams['min_samples_split'],
        min_samples_leaf=best_hyperparams['min_samples_leaf'],
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"R² score on Test Set: {r2}")
    joblib.dump(model, 'final_model.pkl')
    print("Final model saved as 'final_model.pkl'")

    return model


X_train, X_test, y_train, y_test = split_dataset(features_scaled, target_value, 0.2, 42)
X_train, X_val, y_train, y_val = split_dataset(X_train, y_train, 0.125, 42)

if __name__ == "__main__":
    args = parse_arguments()

    # Uncomment the following line if you want to run the hyperparameter tuning
    # train_and_evaluate_model(X_train, y_train, X_val, y_val, args)

    try:
        best_hyperparams = load_best_hyperparameters('best_hyperparams.json')
    except FileNotFoundError as e:
        print(e)
        exit(1)
    
    train_final_model(features_scaled, target_value, best_hyperparams)
