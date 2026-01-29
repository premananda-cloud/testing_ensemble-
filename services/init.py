"""
Model initialization and loading module for ensemble system
"""
import os
import torch
import joblib
import logging
from typing import Dict, Optional, Any
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and initialization of all models in the ensemble"""
    
    def __init__(self, models_base_path: str = "models"):
        """
        Initialize model loader
        
        Args:
            models_base_path: Base directory containing model folders
        """
        self.models_base_path = models_base_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Define model paths
        self.bert_path = os.path.join(models_base_path, "bert", "bert", "final_model")
        self.roberta_path = os.path.join(models_base_path, "robert", "final_model")
        self.tfidf_path = os.path.join(models_base_path, "tf_idf", "tfidf_model.joblib")
    
    def load_bert(self) -> Optional[Dict[str, Any]]:
        """
        Load BERT model and tokenizer
        
        Returns:
            Dictionary containing model, tokenizer, and metadata or None if loading fails
        """
        try:
            if not os.path.exists(self.bert_path):
                logger.warning(f"BERT model not found at {self.bert_path}")
                return None
            
            logger.info(f"Loading BERT model from {self.bert_path}")
            
            # Load tokenizer
            tokenizer = BertTokenizer.from_pretrained(self.bert_path)
            
            # Load model
            model = BertForSequenceClassification.from_pretrained(self.bert_path)
            model.to(self.device)
            model.eval()
            
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"BERT model loaded successfully with {num_params:,} parameters")
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'type': 'transformer',
                'name': 'bert',
                'num_params': num_params
            }
        
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            return None
    
    def load_roberta(self) -> Optional[Dict[str, Any]]:
        """
        Load RoBERTa model and tokenizer
        
        Returns:
            Dictionary containing model, tokenizer, and metadata or None if loading fails
        """
        try:
            if not os.path.exists(self.roberta_path):
                logger.warning(f"RoBERTa model not found at {self.roberta_path}")
                return None
            
            logger.info(f"Loading RoBERTa model from {self.roberta_path}")
            
            # Load tokenizer
            tokenizer = RobertaTokenizer.from_pretrained(self.roberta_path)
            
            # Load model
            model = RobertaForSequenceClassification.from_pretrained(self.roberta_path)
            model.to(self.device)
            model.eval()
            
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"RoBERTa model loaded successfully with {num_params:,} parameters")
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'type': 'transformer',
                'name': 'roberta',
                'num_params': num_params
            }
        
        except Exception as e:
            logger.error(f"Error loading RoBERTa model: {str(e)}")
            return None
    
    def load_tfidf(self) -> Optional[Dict[str, Any]]:
        """
        Load TF-IDF model
        
        Returns:
            Dictionary containing model and metadata or None if loading fails
        """
        try:
            if not os.path.exists(self.tfidf_path):
                logger.warning(f"TF-IDF model not found at {self.tfidf_path}")
                return None
            
            logger.info(f"Loading TF-IDF model from {self.tfidf_path}")
            
            # Load model using joblib
            model = joblib.load(self.tfidf_path)
            
            logger.info(f"TF-IDF model loaded successfully")
            
            return {
                'model': model,
                'type': 'tfidf',
                'name': 'tfidf'
            }
        
        except Exception as e:
            logger.error(f"Error loading TF-IDF model: {str(e)}")
            return None
    
    def load_all_models(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Load all available models
        
        Returns:
            Dictionary containing all loaded models
        """
        logger.info("Loading all models...")
        
        models = {
            'bert': self.load_bert(),
            'roberta': self.load_roberta(),
            'tfidf': self.load_tfidf()
        }
        
        # Count successfully loaded models
        loaded_count = sum(1 for model in models.values() if model is not None)
        logger.info(f"Successfully loaded {loaded_count}/3 models")
        
        if loaded_count == 0:
            raise RuntimeError("No models could be loaded. Please check model paths.")
        
        return models
    
    def get_model_summary(self) -> str:
        """
        Get a summary of available models
        
        Returns:
            String summary of model availability
        """
        summary = []
        summary.append("=" * 60)
        summary.append("MODEL AVAILABILITY SUMMARY")
        summary.append("=" * 60)
        
        models = {
            'BERT': self.bert_path,
            'RoBERTa': self.roberta_path,
            'TF-IDF': self.tfidf_path
        }
        
        for name, path in models.items():
            exists = os.path.exists(path)
            status = "✓ Available" if exists else "✗ Not Found"
            summary.append(f"{name:12} : {status:15} ({path})")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)


def initialize_models(models_base_path: str = "models") -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Convenience function to initialize all models
    
    Args:
        models_base_path: Base directory containing model folders
        
    Returns:
        Dictionary containing all loaded models
    """
    loader = ModelLoader(models_base_path)
    return loader.load_all_models()