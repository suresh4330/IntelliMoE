"""
router/ml_classifier_router.py
------------------------------
Machine Learning Intent Classifier Router.
Uses a trained scikit-learn model and TF-IDF vectorizer to predict
the most relevant expert for a user query.
"""

import os
import logging
from typing import Dict, Tuple

import joblib

logger = logging.getLogger(__name__)


class MLClassifierRouter:
    """
    ML Classifier Router.
    
    Provides text classification inference using serialized scikit-learn models.
    Supports lazy loading and queries fallback checks.
    """

    def __init__(self) -> None:
        self._model = None
        self._vectorizer = None

    def _load_model(self) -> None:
        """
        Lazy-loads the classifier and vectorizer from disk only on the first call.
        """
        if self._model is not None and self._vectorizer is not None:
            return

        from config.settings import INTENT_MODEL_PATH, VECTORIZER_PATH

        if not os.path.exists(INTENT_MODEL_PATH):
            logger.error("Intent classifier model file not found at: %s", INTENT_MODEL_PATH)
            raise FileNotFoundError(
                f"Model file not found: {INTENT_MODEL_PATH}. Run scripts/train_classifier.py first."
            )

        if not os.path.exists(VECTORIZER_PATH):
            logger.error("Vectorizer file not found at: %s", VECTORIZER_PATH)
            raise FileNotFoundError(
                f"Vectorizer file not found: {VECTORIZER_PATH}. Run scripts/train_classifier.py first."
            )

        try:
            logger.info("Loading ML Intent Classifier model and vectorizer...")
            self._model = joblib.load(INTENT_MODEL_PATH)
            self._vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("ML Intent Classifier successfully loaded.")
        except Exception as e:
            logger.exception("Failed to load classifier model or vectorizer.")
            raise RuntimeError(f"Failed to load ML Router resources: {e}") from e

    def predict_expert(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict the target expert and confidence for the query.

        Parameters
        ----------
        query : str
            The input user query.

        Returns
        -------
        predicted_expert : str
            The name of the predicted expert.
        confidence : float
            The prediction confidence score (probability in range [0.0, 1.0]).
        probabilities : dict[str, float]
            Dictionary mapping all expert names to their respective probabilities.
        """
        if not query.strip():
            raise ValueError("Query string cannot be empty.")

        self._load_model()

        try:
            logger.debug("Predicting expert class for query: '%.60s'", query)
            # Transform text query into TF-IDF feature space
            features = self._vectorizer.transform([query])
            
            # Predict probabilities (requires probability=True on SVM)
            probs = self._model.predict_proba(features)[0]
            classes = self._model.classes_

            # Build probability dictionary
            prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}

            # Select the class with the highest probability
            best_idx = probs.argmax()
            predicted_expert = str(classes[best_idx])
            confidence = float(probs[best_idx])

            logger.info("ML Predictor: predicted '%s' with confidence %.2f%%", predicted_expert, confidence * 100)
            return predicted_expert, confidence, prob_dict

        except Exception as e:
            logger.exception("Error occurred during ML expert prediction.")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def get_prediction_details(self, query: str) -> Dict[str, float]:
        """
        Return the prediction probabilities for all experts for the given query.

        Parameters
        ----------
        query : str
            The input query.

        Returns
        -------
        dict[str, float]
            Map of expert class names to their prediction probabilities.
        """
        _, _, probabilities = self.predict_expert(query)
        return probabilities
