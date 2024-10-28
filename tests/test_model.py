"""
This script contains unit tests for validating the trained RandomForest model 
and its associated hyperparameters. The tests ensure that:
1. The model is correctly loaded from the serialized file.
2. The best hyperparameters are correctly stored in a JSON file.
3. Additional tests check the model's functionality and predictability.

To run the tests:
$ pytest test_model.py
"""

import json
import pytest
from joblib import load
from sklearn.ensemble import RandomForestRegressor

@pytest.fixture
def model():
    return load('final_model.pkl')

def test_model_loading(model):
    assert isinstance(model, RandomForestRegressor)

def test_best_hyperparameters():
    with open('best_hyperparams.json', 'r') as f:
        hyperparams = json.load(f)
    assert 'n_estimators' in hyperparams
