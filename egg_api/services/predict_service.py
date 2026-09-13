"""
Prediction Service Module
Loads trained model.pkl and handles egg yolk fan score predictions.
"""

import os
import joblib
import numpy as np


class PredictService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(base_dir, 'model', 'model.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model = joblib.load(model_path)

    def predict(self, r: float, g: float, b: float, l: float, a: float, b_lab: float):
        """
        Predict Yolk Color Fan score based on color features [r, g, b, l, a, b_lab].
        
        :return: tuple (predicted_score: int, raw_score: float)
        """
        features = np.array([[r, g, b, l, a, b_lab]], dtype=np.float64)
        raw_score = float(self.model.predict(features)[0])
        
        # Clip predicted score to valid fan score range [1, 15] and round to integer
        clipped_score = max(1.0, min(15.0, raw_score))
        predicted_score = int(round(clipped_score))

        return predicted_score, round(raw_score, 2)


_predict_service_instance = None

def get_predict_service() -> PredictService:
    global _predict_service_instance
    if _predict_service_instance is None:
        _predict_service_instance = PredictService()
    return _predict_service_instance
