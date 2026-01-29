"""
Test and evaluation module for model ensemble
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import sys

# Add services directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available. Plotting will be disabled.")

from inference import ModelEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for testing model performance"""
    
    def __init__(self, ensemble: ModelEnsemble):
        """
        Initialize evaluator
        
        Args:
            ensemble: ModelEnsemble instance to evaluate
        """
        self.ensemble = ensemble
    
    def load_test_data(self, data_path: str, text_col: str = 'text', 
                      label_col: str = 'label') -> Tuple[List[str], List[int]]:
        """
        Load test data from various formats
        
        Supported formats:
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        - JSON files (.json)
        - Text files (.txt) with format: label\ttext
        
        Args:
            data_path: Path to test data file
            text_col: Name of text column (for structured formats)
            label_col: Name of label column (for structured formats)
            
        Returns:
            Tuple of (texts, labels)
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data not found at {data_path}")
        
        logger.info(f"Loading test data from {data_path}")
        
        # Determine file type and load accordingly
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
            
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
            
        elif data_path.endswith('.txt'):
            # Assume format: label\ttext
            texts, labels = [], []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            labels.append(int(parts[0]))
                            texts.append(parts[1])
                        except ValueError:
                            logger.warning(f"Line {line_num}: Could not parse label as integer")
                    else:
                        logger.warning(f"Line {line_num}: Invalid format (expected: label\\ttext)")
            
            logger.info(f"Loaded {len(texts)} samples from text file")
            return texts, labels
            
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Extract texts and labels from DataFrame
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in data. Available columns: {df.columns.tolist()}")
        
        texts = df[text_col].astype(str).tolist()
        
        if label_col not in df.columns:
            logger.warning(f"Label column '{label_col}' not found. Using default labels (0)")
            labels = [0] * len(df)
        else:
            labels = df[label_col].astype(int).tolist()
        
        logger.info(f"Loaded {len(texts)} samples from {data_path}")
        return texts, labels
    
    def evaluate(self, texts: List[str], true_labels: List[int], 
                save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate model ensemble on test data
        
        Args:
            texts: List of input texts
            true_labels: List of true labels
            save_path: Optional directory path to save evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if len(texts) != len(true_labels):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of labels ({len(true_labels)})")
        
        logger.info(f"Evaluating on {len(texts)} samples...")
        
        # Get predictions
        predictions = self.ensemble.predict_batch(texts, show_progress=True)
        
        # Extract predictions and probabilities
        ensemble_preds = []
        bert_preds = []
        roberta_preds = []
        tfidf_preds = []
        confidences = []
        
        for pred in predictions:
            ensemble_preds.append(pred.ensemble_pred)
            confidences.append(pred.confidence)
            
            if pred.bert_pred is not None:
                bert_preds.append(pred.bert_pred)
            if pred.roberta_pred is not None:
                roberta_preds.append(pred.roberta_pred)
            if pred.tfidf_pred is not None:
                tfidf_preds.append(pred.tfidf_pred)
        
        # Calculate metrics for ensemble
        logger.info("Calculating metrics...")
        ensemble_metrics = self._calculate_metrics(true_labels, ensemble_preds, "ensemble")
        
        # Calculate metrics for individual models
        metrics = {'ensemble': ensemble_metrics}
        
        if bert_preds and len(bert_preds) == len(true_labels):
            metrics['bert'] = self._calculate_metrics(true_labels, bert_preds, "bert")
        
        if roberta_preds and len(roberta_preds) == len(true_labels):
            metrics['roberta'] = self._calculate_metrics(true_labels, roberta_preds, "roberta")
        
        if tfidf_preds and len(tfidf_preds) == len(true_labels):
            metrics['tfidf'] = self._calculate_metrics(true_labels, tfidf_preds, "tfidf")
        
        # Add confidence statistics
        metrics['confidence_stats'] = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences))
        }
        
        # Add sample information
        metrics['sample_info'] = {
            'total_samples': len(texts),
            'num_classes': len(set(true_labels)),
            'class_distribution': {int(k): int(v) for k, v in 
                                  pd.Series(true_labels).value_counts().to_dict().items()}
        }
        
        # Save results if path provided
        if save_path:
            self._save_results(metrics, texts, true_labels, predictions, save_path)
        
        return metrics
    
    def _calculate_metrics(self, true_labels: List[int], pred_labels: List[int], 
                          model_name: str) -> Dict[str, Any]:
        """
        Calculate evaluation metrics
        
        Args:
            true_labels: True labels
            pred_labels: Predicted labels
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(true_labels, pred_labels)),
            'precision': float(precision_score(true_labels, pred_labels, 
                                              average='weighted', zero_division=0)),
            'recall': float(recall_score(true_labels, pred_labels, 
                                        average='weighted', zero_division=0)),
            'f1_score': float(f1_score(true_labels, pred_labels, 
                                      average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist()
        }
        
        # Add classification report
        try:
            report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        except Exception as e:
            logger.warning(f"Could not generate classification report: {str(e)}")
        
        return metrics
    
    def _save_results(self, metrics: Dict[str, Any], texts: List[str], 
                     true_labels: List[int], predictions: List[Any], 
                     save_path: str):
        """
        Save evaluation results to files
        
        Args:
            metrics: Metrics dictionary
            texts: Input texts
            true_labels: True labels
            predictions: List of Prediction objects
            save_path: Directory to save results
        """
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving results to {save_path}")
        
        # Save metrics as JSON
        metrics_path = os.path.join(save_path, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Metrics saved to {metrics_path}")
        
        # Save detailed predictions as CSV
        pred_data = []
        for text, true_label, pred in zip(texts, true_labels, predictions):
            row = {
                'text': text,
                'true_label': true_label,
                'ensemble_pred': pred.ensemble_pred,
                'ensemble_confidence': pred.confidence,
                'correct': true_label == pred.ensemble_pred
            }
            
            if pred.bert_pred is not None:
                row['bert_pred'] = pred.bert_pred
            if pred.roberta_pred is not None:
                row['roberta_pred'] = pred.roberta_pred
            if pred.tfidf_pred is not None:
                row['tfidf_pred'] = pred.tfidf_pred
            
            pred_data.append(row)
        
        results_df = pd.DataFrame(pred_data)
        csv_path = os.path.join(save_path, 'detailed_predictions.csv')
        results_df.to_csv(csv_path, index=False)
        logger.info(f"✓ Detailed predictions saved to {csv_path}")
        
        # Create and save confusion matrix plot
        if PLOTTING_AVAILABLE:
            try:
                self._plot_confusion_matrix(metrics['ensemble']['confusion_matrix'], save_path)
            except Exception as e:
                logger.warning(f"Could not create confusion matrix plot: {str(e)}")
        
        # Save summary report
        self._save_summary_report(metrics, save_path)
    
    def _plot_confusion_matrix(self, cm: List[List[int]], save_path: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Ensemble Model - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        plot_path = os.path.join(save_path, 'confusion_matrix.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"✓ Confusion matrix plot saved to {plot_path}")
    
    def _save_summary_report(self, metrics: Dict[str, Any], save_path: str):
        """Save a human-readable summary report"""
        report_path = os.path.join(save_path, 'evaluation_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MODEL EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Sample information
            f.write("Dataset Information:\n")
            f.write("-"*70 + "\n")
            sample_info = metrics.get('sample_info', {})
            f.write(f"Total Samples: {sample_info.get('total_samples', 'N/A')}\n")
            f.write(f"Number of Classes: {sample_info.get('num_classes', 'N/A')}\n")
            
            class_dist = sample_info.get('class_distribution', {})
            if class_dist:
                f.write("\nClass Distribution:\n")
                for label, count in sorted(class_dist.items()):
                    percentage = (count / sample_info['total_samples'] * 100)
                    f.write(f"  Class {label}: {count} samples ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Model performance
            f.write("Model Performance:\n")
            f.write("-"*70 + "\n")
            
            for model_name in ['ensemble', 'bert', 'roberta', 'tfidf']:
                if model_name in metrics and model_name != 'confidence_stats':
                    model_metrics = metrics[model_name]
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write(f"  Accuracy:  {model_metrics['accuracy']:.4f}\n")
                    f.write(f"  Precision: {model_metrics['precision']:.4f}\n")
                    f.write(f"  Recall:    {model_metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score:  {model_metrics['f1_score']:.4f}\n")
            
            # Confidence statistics
            f.write("\n")
            f.write("Confidence Statistics:\n")
            f.write("-"*70 + "\n")
            conf_stats = metrics.get('confidence_stats', {})
            f.write(f"Mean:   {conf_stats.get('mean', 0):.4f}\n")
            f.write(f"Std:    {conf_stats.get('std', 0):.4f}\n")
            f.write(f"Min:    {conf_stats.get('min', 0):.4f}\n")
            f.write(f"Max:    {conf_stats.get('max', 0):.4f}\n")
            f.write(f"Median: {conf_stats.get('median', 0):.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        logger.info(f"✓ Summary report saved to {report_path}")


def evaluate_models(test_data_path: str, models_base_path: str = "models", 
                   text_col: str = 'text', label_col: str = 'label',
                   save_results: bool = True, output_dir: str = "results") -> Dict[str, Any]:
    """
    Convenience function for evaluating models
    
    Args:
        test_data_path: Path to test data file
        models_base_path: Path to models directory
        text_col: Name of text column
        label_col: Name of label column
        save_results: Whether to save results
        output_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load ensemble
    logger.info("Initializing model ensemble...")
    ensemble = ModelEnsemble(models_base_path)
    ensemble.print_info()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(ensemble)
    
    # Load test data
    texts, labels = evaluator.load_test_data(test_data_path, text_col, label_col)
    
    # Evaluate
    save_path = os.path.join(output_dir, 'evaluation') if save_results else None
    metrics = evaluator.evaluate(texts, labels, save_path)
    
    # Print summary to console
    print_evaluation_summary(metrics)
    
    return metrics


def print_evaluation_summary(metrics: Dict[str, Any]):
    """Print evaluation summary to console"""
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    # Print model performance
    for model_name in ['ensemble', 'bert', 'roberta', 'tfidf']:
        if model_name in metrics and isinstance(metrics[model_name], dict):
            model_metrics = metrics[model_name]
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:  {model_metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {model_metrics.get('precision', 0):.4f}")
            print(f"  Recall:    {model_metrics.get('recall', 0):.4f}")
            print(f"  F1-Score:  {model_metrics.get('f1_score', 0):.4f}")
    
    # Print confidence statistics
    if 'confidence_stats' in metrics:
        print(f"\nConfidence Statistics:")
        conf = metrics['confidence_stats']
        print(f"  Mean:   {conf.get('mean', 0):.4f}")
        print(f"  Std:    {conf.get('std', 0):.4f}")
        print(f"  Min:    {conf.get('min', 0):.4f}")
        print(f"  Max:    {conf.get('max', 0):.4f}")
        print(f"  Median: {conf.get('median', 0):.4f}")
    
    print("="*70 + "\n")


def load_test_data(data_path: str, text_col: str = 'text', 
                  label_col: str = 'label') -> Tuple[List[str], List[int]]:
    """
    Convenience function for loading test data
    
    Args:
        data_path: Path to test data file
        text_col: Name of text column
        label_col: Name of label column
        
    Returns:
        Tuple of (texts, labels)
    """
    # Create a temporary ensemble just for loading data
    ensemble = ModelEnsemble()
    evaluator = ModelEvaluator(ensemble)
    return evaluator.load_test_data(data_path, text_col, label_col)