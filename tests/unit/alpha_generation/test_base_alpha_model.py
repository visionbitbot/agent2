import pytest
import pandas as pd
from chimera.alpha_generation.base_alpha_model import BaseAlphaModel

# Create a concrete implementation for testing BaseAlphaModel initialization and get_info
class ConcreteTestModel(BaseAlphaModel):
    def preprocess_features(self, features_df: pd.DataFrame) -> any: return features_df
    def train(self, training_data: pd.DataFrame, labels: pd.Series): pass
    def predict_alpha(self, processed_features: any) -> tuple[pd.DataFrame, pd.DataFrame]:
        return pd.DataFrame(), pd.DataFrame()
    def load_model(self, file_path: str): pass
    def save_model(self, file_path: str): pass

def test_base_model_constructor_valid():
    model = ConcreteTestModel(model_name="TestModel", version="1.0", config={"param": 1})
    assert model.model_name == "TestModel"
    assert model.version == "1.0"
    assert model.config == {"param": 1}
    assert model.trained_model is None

def test_base_model_constructor_invalid_names():
    with pytest.raises(ValueError): ConcreteTestModel(model_name="", version="1.0")
    with pytest.raises(ValueError): ConcreteTestModel(model_name="Test", version="")

def test_base_model_get_info():
    model = ConcreteTestModel(model_name="InfoModel", version="1.1")
    info = model.get_info()
    assert info["model_name"] == "InfoModel"
    assert info["version"] == "1.1"
    assert info["is_trained"] is False
    model.trained_model = "dummy_state" # Simulate training
    assert model.get_info()["is_trained"] is True
