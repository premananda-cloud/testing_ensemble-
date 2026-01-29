"""
Services package for multi-model ensemble inference and evaluation

This package provides functionality for:
- Loading BERT, RoBERTa, and TF-IDF models
- Running ensemble predictions
- Evaluating model performance
"""

from .init import ModelLoader, initialize_models
from .inference import ModelEnsemble, Prediction, predict_single, predict_batch
from .test import (
    ModelEvaluator, 
    evaluate_models, 
    load_test_data,
    print_evaluation_summary
)

__version__ = "1.0.0"

__all__ = [
    # Model loading
    "ModelLoader",
    "initialize_models",
    
    # Inference
    "ModelEnsemble",
    "Prediction",
    "predict_single",
    "predict_batch",
    
    # Evaluation
    "ModelEvaluator",
    "evaluate_models",
    "load_test_data",
    "print_evaluation_summary",
]