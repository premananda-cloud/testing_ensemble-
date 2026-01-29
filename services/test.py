"""
Testing Script
Evaluates trained models on test dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import argparse
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from inference import ModelInference


class ModelTester:
    """
    Test trained models on evaluation dataset.
    """
    
    def __init__(self, inference_service: ModelInference):
        """
        Initialize tester.
        
        Args:
            inference_service: Initialized ModelInference instance
        """
        self.inference = inference_service
    
    def load_test_data(self, filepath: str) -> Tuple[List[str], List[int]]:
        """
        Load test data from TSV file.
        
        Args:
            filepath: Path to TSV file
        
        Returns:
            Tuple of (texts, labels)
        """
        print(f"Loading test data from {filepath}...")
        
        # Read TSV file
        df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'text'])
        
        # Clean data
        df = df.dropna()
        df = df[df['label'].isin([0, 1])]
        
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        
        print(f"✓ Loaded {len(texts)} samples")
        print(f"  Real news (0): {labels.count(0)}")
        print(f"  Fake news (1): {labels.count(1)}")
        
        return texts, labels
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for positive class)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add AUC-ROC if possible
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc_roc'] = 0.0
        
        return metrics
    
    def print_results(
        self,
        model_name: str,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """
        Print formatted results.
        
        Args:
            model_name: Name of the model
            metrics: Computed metrics
            y_true: True labels
            y_pred: Predicted labels
        """
        print(f"\n{'='*70}")
        print(f"{model_name.upper()} RESULTS")
        print(f"{'='*70}")
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f} ({metrics['auc_roc']*100:.2f}%)")
        print(f"{'='*70}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Real    Fake")
        print(f"Actual Real   {cm[0,0]:<6}  {cm[0,1]:<6}")
        print(f"       Fake   {cm[1,0]:<6}  {cm[1,1]:<6}")
        
        # Per-class accuracy
        print(f"\nPer-Class Accuracy:")
        print(f"Real news: {cm[0,0]/cm[0].sum():.4f} ({cm[0,0]}/{cm[0].sum()})")
        print(f"Fake news: {cm[1,1]/cm[1].sum():.4f} ({cm[1,1]}/{cm[1].sum()})")
        print(f"{'='*70}\n")
    
    def test_model(
        self,
        texts: List[str],
        labels: List[int],
        model: str = 'bert',
        ensemble_method: str = 'average',
        batch_size: int = 32
    ) -> Dict:
        """
        Test a specific model.
        
        Args:
            texts: List of text samples
            labels: List of true labels
            model: Model to test ('bert', 'roberta', 'ensemble')
            ensemble_method: Ensemble method if model='ensemble'
            batch_size: Batch size for inference
        
        Returns:
            Dictionary with results
        """
        print(f"\nTesting {model.upper()} model...")
        
        # Batch prediction
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Predict
            results = self.inference.predict(
                batch_texts,
                model=model,
                ensemble_method=ensemble_method
            )
            
            all_predictions.extend(results['predictions'])
            all_probabilities.extend(results['probabilities'][:, 1])  # Prob of class 1
        
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        y_true = np.array(labels)
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        
        # Print results
        model_display_name = f"{model} ({ensemble_method})" if model == 'ensemble' else model
        self.print_results(model_display_name, metrics, y_true, y_pred)
        
        return {
            'model': model,
            'ensemble_method': ensemble_method if model == 'ensemble' else None,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def test_all_models(
        self,
        texts: List[str],
        labels: List[int],
        ensemble_method: str = 'average',
        batch_size: int = 32
    ) -> Dict[str, Dict]:
        """
        Test all available models.
        
        Args:
            texts: List of text samples
            labels: List of true labels
            ensemble_method: Ensemble method
            batch_size: Batch size
        
        Returns:
            Dictionary mapping model names to results
        """
        results = {}
        available_models = self.inference.get_available_models()
        
        print(f"\n{'='*70}")
        print(f"TESTING ALL MODELS")
        print(f"Available models: {', '.join(available_models)}")
        print(f"{'='*70}")
        
        # Test individual models
        for model in ['bert', 'roberta']:
            if model in available_models:
                results[model] = self.test_model(
                    texts, labels, model=model, batch_size=batch_size
                )
        
        # Test ensemble if multiple models available
        if 'ensemble' in available_models:
            results[f'ensemble_{ensemble_method}'] = self.test_model(
                texts, labels, model='ensemble',
                ensemble_method=ensemble_method,
                batch_size=batch_size
            )
        
        return results
    
    def compare_models(self, results: Dict[str, Dict]):
        """
        Compare results from multiple models.
        
        Args:
            results: Dictionary of results from test_all_models
        """
        if len(results) < 2:
            return
        
        print(f"\n{'='*70}")
        print(f"MODEL COMPARISON")
        print(f"{'='*70}")
        
        # Create comparison table
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        
        # Header
        model_names = list(results.keys())
        header = f"{'Metric':<15}"
        for name in model_names:
            header += f"{name:<15}"
        print(header)
        print("-" * (15 + 15 * len(model_names)))
        
        # Rows
        for metric in metrics:
            row = f"{metric:<15}"
            for name in model_names:
                value = results[name]['metrics'][metric]
                row += f"{value:.4f}         "
            print(row)
        
        print(f"{'='*70}")
        
        # Find best model for each metric
        print(f"\nBest Models:")
        for metric in metrics:
            best_model = max(
                results.keys(),
                key=lambda m: results[m]['metrics'][metric]
            )
            best_score = results[best_model]['metrics'][metric]
            print(f"  {metric:<15}: {best_model} ({best_score:.4f})")
        
        print(f"{'='*70}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test trained models')
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/test_file.tsv',
        help='Path to test data (TSV format)'
    )
    parser.add_argument(
        '--bert-model',
        type=str,
        default='models/bert/bert/final_model',
        help='Path to BERT model'
    )
    parser.add_argument(
        '--roberta-model',
        type=str,
        default='models/robert/final_model',
        help='Path to RoBERTa model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['bert', 'roberta', 'ensemble', 'all'],
        help='Model to test'
    )
    parser.add_argument(
        '--ensemble-method',
        type=str,
        default='average',
        choices=['average', 'voting', 'weighted'],
        help='Ensemble method'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    
    return parser.parse_args()


def main():
    """Main testing function."""
    args = parse_args()
    
    print(f"\n{'='*70}")
    print("MODEL TESTING")
    print(f"{'='*70}\n")
    
    # Check paths
    bert_path = Path(args.bert_model)
    roberta_path = Path(args.roberta_model)
    data_path = Path(args.data)
    
    if not data_path.exists():
        print(f"Error: Test data not found at {args.data}")
        return
    
    # Initialize inference service
    print("Initializing inference service...")
    inference = ModelInference(
        bert_model_path=str(bert_path) if bert_path.exists() else None,
        roberta_model_path=str(roberta_path) if roberta_path.exists() else None,
        device=args.device
    )
    
    available = inference.get_available_models()
    if not available:
        print("Error: No models could be loaded")
        return
    
    print(f"✓ Available models: {', '.join(available)}\n")
    
    # Initialize tester
    tester = ModelTester(inference)
    
    # Load test data
    texts, labels = tester.load_test_data(args.data)
    
    # Test models
    if args.model == 'all':
        results = tester.test_all_models(
            texts, labels,
            ensemble_method=args.ensemble_method,
            batch_size=args.batch_size
        )
        tester.compare_models(results)
    else:
        result = tester.test_model(
            texts, labels,
            model=args.model,
            ensemble_method=args.ensemble_method,
            batch_size=args.batch_size
        )
    
    print("\n✓ Testing completed!\n")


if __name__ == "__main__":
    main()