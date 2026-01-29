"""
Inference module for ensemble predictions
"""
import torch
import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional
from dataclasses import dataclass
import logging
import os
import sys

# Add services directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from init import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Data class for prediction results"""
    text: str
    bert_pred: Optional[int] = None
    roberta_pred: Optional[int] = None
    tfidf_pred: Optional[int] = None
    bert_probs: Optional[List[float]] = None
    roberta_probs: Optional[List[float]] = None
    tfidf_probs: Optional[List[float]] = None
    ensemble_pred: Optional[int] = None
    ensemble_probs: Optional[List[float]] = None
    confidence: Optional[float] = None


class ModelEnsemble:
    """Ensemble model combining BERT, RoBERTa, and TF-IDF"""
    
    def __init__(self, models_base_path: str = "models", weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble with all models
        
        Args:
            models_base_path: Base path to models directory
            weights: Optional custom weights for ensemble (dict with 'bert', 'roberta', 'tfidf' keys)
        """
        self.loader = ModelLoader(models_base_path)
        self.models = self.loader.load_all_models()
        self.device = self.loader.device
        
        # Determine model types
        self.has_bert = self.models['bert'] is not None
        self.has_roberta = self.models['roberta'] is not None
        self.has_tfidf = self.models['tfidf'] is not None
        
        # Set ensemble weights
        if weights is None:
            self.weights = {
                'bert': 0.4 if self.has_bert else 0,
                'roberta': 0.4 if self.has_roberta else 0,
                'tfidf': 0.2 if self.has_tfidf else 0
            }
        else:
            self.weights = weights
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total
        else:
            raise ValueError("No valid models loaded or all weights are zero")
        
        logger.info(f"Initialized ensemble with weights: {self.weights}")
        logger.info(f"Models available: BERT={self.has_bert}, RoBERTa={self.has_roberta}, TF-IDF={self.has_tfidf}")
    
    def _preprocess_transformer(self, text: str, tokenizer, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Preprocess text for transformer models"""
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def _predict_bert(self, text: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Make prediction using BERT model"""
        if not self.has_bert:
            return None, None
        
        try:
            model_info = self.models['bert']
            inputs = self._preprocess_transformer(text, model_info['tokenizer'])
            
            with torch.no_grad():
                outputs = model_info['model'](**inputs)
                logits = outputs.logits
            
            # Get probabilities and prediction
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            
            return prediction.cpu().item(), probabilities.cpu().numpy()[0]
        
        except Exception as e:
            logger.error(f"Error in BERT prediction: {str(e)}")
            return None, None
    
    def _predict_roberta(self, text: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Make prediction using RoBERTa model"""
        if not self.has_roberta:
            return None, None
        
        try:
            model_info = self.models['roberta']
            inputs = self._preprocess_transformer(text, model_info['tokenizer'])
            
            with torch.no_grad():
                outputs = model_info['model'](**inputs)
                logits = outputs.logits
            
            # Get probabilities and prediction
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            
            return prediction.cpu().item(), probabilities.cpu().numpy()[0]
        
        except Exception as e:
            logger.error(f"Error in RoBERTa prediction: {str(e)}")
            return None, None
    
    def _predict_tfidf(self, text: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Make prediction using TF-IDF model"""
        if not self.has_tfidf:
            return None, None
        
        try:
            model = self.models['tfidf']['model']
            
            # Make prediction
            prediction = model.predict([text])[0]
            
            # Try to get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba([text])[0]
            else:
                # Create one-hot encoding if predict_proba not available
                # Assume binary classification (adjust if needed)
                num_classes = 2
                probabilities = np.zeros(num_classes)
                probabilities[prediction] = 1.0
            
            return int(prediction), probabilities
        
        except Exception as e:
            logger.error(f"Error in TF-IDF prediction: {str(e)}")
            return None, None
    
    def predict(self, text: str, return_individual: bool = True) -> Prediction:
        """
        Make ensemble prediction
        
        Args:
            text: Input text to classify
            return_individual: If True, return individual model predictions in Prediction object
            
        Returns:
            Prediction object containing all results
        """
        # Get predictions from all models
        bert_pred, bert_probs = self._predict_bert(text)
        roberta_pred, roberta_probs = self._predict_roberta(text)
        tfidf_pred, tfidf_probs = self._predict_tfidf(text)
        
        # Determine number of classes from available predictions
        num_classes = None
        for probs in [bert_probs, roberta_probs, tfidf_probs]:
            if probs is not None:
                num_classes = len(probs)
                break
        
        if num_classes is None:
            raise ValueError("Could not determine number of classes from any model")
        
        # Initialize ensemble probabilities
        ensemble_probs = np.zeros(num_classes)
        total_weight = 0.0
        
        # Combine probabilities using weighted average
        if bert_probs is not None and self.weights['bert'] > 0:
            ensemble_probs += self.weights['bert'] * bert_probs
            total_weight += self.weights['bert']
        
        if roberta_probs is not None and self.weights['roberta'] > 0:
            ensemble_probs += self.weights['roberta'] * roberta_probs
            total_weight += self.weights['roberta']
        
        if tfidf_probs is not None and self.weights['tfidf'] > 0:
            ensemble_probs += self.weights['tfidf'] * tfidf_probs
            total_weight += self.weights['tfidf']
        
        # Normalize if needed
        if total_weight > 0:
            ensemble_probs /= total_weight
        
        # Get final prediction and confidence
        ensemble_pred = int(np.argmax(ensemble_probs))
        confidence = float(np.max(ensemble_probs))
        
        # Create prediction object
        prediction = Prediction(
            text=text,
            bert_pred=bert_pred,
            roberta_pred=roberta_pred,
            tfidf_pred=tfidf_pred,
            bert_probs=bert_probs.tolist() if bert_probs is not None else None,
            roberta_probs=roberta_probs.tolist() if roberta_probs is not None else None,
            tfidf_probs=tfidf_probs.tolist() if tfidf_probs is not None else None,
            ensemble_pred=ensemble_pred,
            ensemble_probs=ensemble_probs.tolist(),
            confidence=confidence
        )
        
        return prediction
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, 
                     show_progress: bool = True) -> List[Prediction]:
        """
        Make predictions for a batch of texts
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once (not used in current implementation)
            show_progress: Whether to show progress
            
        Returns:
            List of Prediction objects
        """
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and (i % 10 == 0 or i == total - 1):
                logger.info(f"Processing {i+1}/{total} texts...")
            
            results.append(self.predict(text, return_individual=True))
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'device': str(self.device),
            'models_loaded': [],
            'ensemble_weights': self.weights,
            'model_details': {}
        }
        
        for name, model_info in self.models.items():
            if model_info is not None:
                info['models_loaded'].append(name)
                info['model_details'][name] = {
                    'type': model_info['type'],
                    'name': model_info['name']
                }
                
                if model_info['type'] == 'transformer':
                    info['model_details'][name]['num_params'] = model_info['num_params']
        
        return info
    
    def print_info(self):
        """Print model information"""
        info = self.get_model_info()
        
        print("\n" + "="*60)
        print("MODEL ENSEMBLE INFORMATION")
        print("="*60)
        print(f"Device: {info['device']}")
        print(f"Models Loaded: {', '.join(info['models_loaded'])}")
        print(f"\nEnsemble Weights:")
        for model, weight in info['ensemble_weights'].items():
            print(f"  {model:10} : {weight:.3f}")
        
        print(f"\nModel Details:")
        for model, details in info['model_details'].items():
            print(f"  {model}:")
            for key, value in details.items():
                print(f"    {key}: {value}")
        print("="*60 + "\n")


# Convenience functions
def predict_single(text: str, models_base_path: str = "models") -> Prediction:
    """
    Convenience function for single prediction
    
    Args:
        text: Input text
        models_base_path: Path to models directory
        
    Returns:
        Prediction object
    """
    ensemble = ModelEnsemble(models_base_path)
    return ensemble.predict(text, return_individual=True)


def predict_batch(texts: List[str], models_base_path: str = "models") -> List[Prediction]:
    """
    Convenience function for batch predictions
    
    Args:
        texts: List of input texts
        models_base_path: Path to models directory
        
    Returns:
        List of Prediction objects
    """
    ensemble = ModelEnsemble(models_base_path)
    return ensemble.predict_batch(texts)