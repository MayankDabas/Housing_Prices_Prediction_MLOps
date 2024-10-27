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
