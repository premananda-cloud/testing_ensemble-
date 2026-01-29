"""
Inference Service for BERT and RoBERTa Models
Loads trained models and performs predictions
"""

import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel
)
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path


class BertClassifier(nn.Module):
    """BERT-based classifier."""
    
    def __init__(self, bert_model, num_labels=2, dropout_rate=0.3):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RobertaClassifier(nn.Module):
    """RoBERTa-based classifier."""
    
    def __init__(self, roberta_model, num_labels=2, dropout_rate=0.3):
        super(RobertaClassifier, self).__init__()
        self.roberta = roberta_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(roberta_model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class ModelInference:
    """
    Inference service for trained BERT and RoBERTa models.
    """
    
    def __init__(
        self,
        bert_model_path: Optional[str] = None,
        roberta_model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_length: int = 512
    ):
        """
        Initialize inference service.
        
        Args:
            bert_model_path: Path to BERT model directory
            roberta_model_path: Path to RoBERTa model directory
            device: Device for inference ('cuda' or 'cpu')
            max_length: Maximum sequence length
        """
        self.device = device
        self.max_length = max_length
        
        # Initialize models
        self.bert_model = None
        self.bert_tokenizer = None
        self.roberta_model = None
        self.roberta_tokenizer = None
        
        # Load BERT if path provided
        if bert_model_path:
            self.load_bert_model(bert_model_path)
        
        # Load RoBERTa if path provided
        if roberta_model_path:
            self.load_roberta_model(roberta_model_path)
    
    def load_bert_model(self, model_path: str):
        """
        Load BERT model from checkpoint.
        
        Args:
            model_path: Path to BERT model directory
        """
        print(f"Loading BERT model from {model_path}...")
        
        try:
            # Load tokenizer
            tokenizer_path = Path(model_path).parent.parent / "bert-base-uncased"
            if tokenizer_path.exists():
                self.bert_tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
            else:
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Load BERT base model
            bert_base = BertModel.from_pretrained('bert-base-uncased')
            
            # Create classifier model
            self.bert_model = BertClassifier(bert_base, num_labels=2, dropout_rate=0.3)
            
            # Load weights from safetensors
            from safetensors.torch import load_file
            state_dict = load_file(f"{model_path}/model.safetensors")
            self.bert_model.load_state_dict(state_dict, strict=False)
            
            # Move to device and set to eval mode
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            print("✓ BERT model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading BERT model: {e}")
            self.bert_model = None
            self.bert_tokenizer = None
    
    def load_roberta_model(self, model_path: str):
        """
        Load RoBERTa model from checkpoint.
        
        Args:
            model_path: Path to RoBERTa model directory
        """
        print(f"Loading RoBERTa model from {model_path}...")
        
        try:
            # Load tokenizer from model directory
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained(model_path)
            
            # Load RoBERTa base model
            roberta_base = RobertaModel.from_pretrained(model_path)
            
            # Create classifier model
            self.roberta_model = RobertaClassifier(roberta_base, num_labels=2, dropout_rate=0.3)
            
            # Load weights from safetensors
            from safetensors.torch import load_file
            state_dict = load_file(f"{model_path}/model.safetensors")
            self.roberta_model.load_state_dict(state_dict, strict=False)
            
            # Move to device and set to eval mode
            self.roberta_model.to(self.device)
            self.roberta_model.eval()
            
            print("✓ RoBERTa model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading RoBERTa model: {e}")
            self.roberta_model = None
            self.roberta_tokenizer = None
    
    def predict_bert(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Predict using BERT model.
        
        Args:
            texts: List of text samples
        
        Returns:
            Dictionary with predictions, probabilities, and logits
        """
        if self.bert_model is None or self.bert_tokenizer is None:
            raise ValueError("BERT model not loaded")
        
        # Tokenize
        encodings = self.bert_tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.bert_model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }
    
    def predict_roberta(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Predict using RoBERTa model.
        
        Args:
            texts: List of text samples
        
        Returns:
            Dictionary with predictions, probabilities, and logits
        """
        if self.roberta_model is None or self.roberta_tokenizer is None:
            raise ValueError("RoBERTa model not loaded")
        
        # Tokenize
        encodings = self.roberta_tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.roberta_model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }
    
    def predict_ensemble(
        self,
        texts: List[str],
        method: str = 'average'
    ) -> Dict[str, np.ndarray]:
        """
        Predict using ensemble of BERT and RoBERTa.
        
        Args:
            texts: List of text samples
            method: Ensemble method ('average', 'voting', 'weighted')
        
        Returns:
            Dictionary with ensemble predictions and probabilities
        """
        results = {}
        
        # Get predictions from available models
        if self.bert_model is not None:
            bert_results = self.predict_bert(texts)
            results['bert'] = bert_results
        
        if self.roberta_model is not None:
            roberta_results = self.predict_roberta(texts)
            results['roberta'] = roberta_results
        
        if not results:
            raise ValueError("No models available for ensemble")
        
        # If only one model available, return its predictions
        if len(results) == 1:
            model_results = list(results.values())[0]
            return {
                'predictions': model_results['predictions'],
                'probabilities': model_results['probabilities'],
                'method': 'single_model',
                'individual_results': results
            }
        
        # Ensemble predictions
        bert_probs = results['bert']['probabilities']
        roberta_probs = results['roberta']['probabilities']
        
        if method == 'average':
            # Average probabilities
            ensemble_probs = (bert_probs + roberta_probs) / 2
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        elif method == 'voting':
            # Majority voting
            bert_preds = results['bert']['predictions']
            roberta_preds = results['roberta']['predictions']
            ensemble_preds = np.where(
                bert_preds == roberta_preds,
                bert_preds,
                bert_preds  # Tie-breaker: use BERT
            )
            ensemble_probs = (bert_probs + roberta_probs) / 2
        
        elif method == 'weighted':
            # Weighted average (BERT: 0.4, RoBERTa: 0.6)
            ensemble_probs = 0.4 * bert_probs + 0.6 * roberta_probs
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return {
            'predictions': ensemble_preds,
            'probabilities': ensemble_probs,
            'method': method,
            'individual_results': results
        }
    
    def predict(
        self,
        texts: List[str],
        model: str = 'ensemble',
        ensemble_method: str = 'average'
    ) -> Dict[str, np.ndarray]:
        """
        General prediction method.
        
        Args:
            texts: List of text samples
            model: Model to use ('bert', 'roberta', 'ensemble')
            ensemble_method: Ensemble method if model='ensemble'
        
        Returns:
            Prediction results
        """
        if model == 'bert':
            return self.predict_bert(texts)
        elif model == 'roberta':
            return self.predict_roberta(texts)
        elif model == 'ensemble':
            return self.predict_ensemble(texts, method=ensemble_method)
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        available = []
        if self.bert_model is not None:
            available.append('bert')
        if self.roberta_model is not None:
            available.append('roberta')
        if len(available) > 1:
            available.append('ensemble')
        return available


if __name__ == "__main__":
    # Test inference service
    print("Testing Inference Service...")
    
    # Initialize (paths would be provided in actual use)
    inference = ModelInference(
        bert_model_path='models/bert/bert/final_model',
        roberta_model_path='models/robert/final_model',
        device='cpu'
    )
    
    # Test texts
    test_texts = [
        "Scientists discover new breakthrough in cancer research.",
        "You won't believe this shocking miracle cure!"
    ]
    
    # Get available models
    print(f"\nAvailable models: {inference.get_available_models()}")
    
    # Test predictions
    if 'bert' in inference.get_available_models():
        print("\nBERT Predictions:")
        bert_results = inference.predict_bert(test_texts)
        print(f"Predictions: {bert_results['predictions']}")
        print(f"Probabilities: {bert_results['probabilities']}")
    
    if 'ensemble' in inference.get_available_models():
        print("\nEnsemble Predictions:")
        ensemble_results = inference.predict_ensemble(test_texts)
        print(f"Predictions: {ensemble_results['predictions']}")
        print(f"Probabilities: {ensemble_results['probabilities']}")