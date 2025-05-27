from abc import ABC, abstractmethod
import pandas as pd
import numpy as np # Ensure numpy is imported

class BaseAlphaModel(ABC):
    def __init__(self, model_name: str, version: str, config: dict | None = None):
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Model name must be a non-empty string.")
        if not isinstance(version, str) or not version:
            raise ValueError("Version must be a non-empty string.")
        
        self.model_name = model_name
        self.version = version
        self.config = config if config is not None else {}
        self.trained_model = None 

    @abstractmethod
    def preprocess_features(self, features_df: pd.DataFrame) -> any:
        pass

    @abstractmethod
    def train(self, training_data: pd.DataFrame, labels: pd.Series | pd.DataFrame): # Allow DataFrame for multi-output
        pass

    @abstractmethod
    def predict_alpha(self, processed_features: any) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def load_model(self, file_path: str):
        pass

    @abstractmethod
    def save_model(self, file_path: str):
        pass

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "config": self.config,
            "is_trained": self.trained_model is not None
        }
