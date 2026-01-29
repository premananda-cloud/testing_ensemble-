"""
Main entry point for the model ensemble system
"""
import os
import sys
import argparse
import logging
from typing import List, Optional

# Add services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

from services.inference import ModelEnsemble, predict_single, predict_batch
from services.test import evaluate_models
from services.init import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_inference(texts: List[str], models_path: str = "models"):
    """
    Run inference on a list of texts
    
    Args:
        texts: List of input texts
        models_path: Path to models directory
    """
    logger.info("="*70)
    logger.info("RUNNING INFERENCE")
    logger.info("="*70)
    
    # Initialize ensemble
    ensemble = ModelEnsemble(models_path)
    ensemble.print_info()
    
    # Run predictions
    logger.info(f"\nProcessing {len(texts)} text(s)...")
    predictions = ensemble.predict_batch(texts)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    for i, pred in enumerate(predictions, 1):
        print(f"\nText {i}: {pred.text[:100]}{'...' if len(pred.text) > 100 else ''}")
        print(f"  Ensemble Prediction: {pred.ensemble_pred} (confidence: {pred.confidence:.4f})")
        
        if pred.bert_pred is not None:
            print(f"  BERT Prediction:     {pred.bert_pred}")
        if pred.roberta_pred is not None:
            print(f"  RoBERTa Prediction:  {pred.roberta_pred}")
        if pred.tfidf_pred is not None:
            print(f"  TF-IDF Prediction:   {pred.tfidf_pred}")
    
    print("="*70 + "\n")
    
    return predictions


def run_evaluation(test_data_path: str, models_path: str = "models", 
                  text_col: str = "text", label_col: str = "label",
                  save_results: bool = True):
    """
    Run model evaluation
    
    Args:
        test_data_path: Path to test data file
        models_path: Path to models directory
        text_col: Name of text column
        label_col: Name of label column
        save_results: Whether to save results
    """
    logger.info("="*70)
    logger.info("RUNNING EVALUATION")
    logger.info("="*70)
    
    # Run evaluation
    metrics = evaluate_models(
        test_data_path=test_data_path,
        models_base_path=models_path,
        text_col=text_col,
        label_col=label_col,
        save_results=save_results
    )
    
    return metrics


def check_models(models_path: str = "models"):
    """
    Check which models are available
    
    Args:
        models_path: Path to models directory
    """
    logger.info("="*70)
    logger.info("CHECKING MODEL AVAILABILITY")
    logger.info("="*70)
    
    loader = ModelLoader(models_path)
    print(loader.get_model_summary())


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Multi-Model Ensemble System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check available models
  python main.py --check-models
  
  # Run inference on sample texts
  python main.py --mode inference --text "This is a great product!" "I hate this service."
  
  # Run evaluation on test data
  python main.py --mode evaluate --test-data data/test.csv
  
  # Run evaluation with custom column names
  python main.py --mode evaluate --test-data data/test.csv --text-col review --label-col sentiment
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['inference', 'evaluate'],
        help='Mode: inference or evaluate'
    )
    
    parser.add_argument(
        '--models-path',
        type=str,
        default='models',
        help='Path to models directory (default: models)'
    )
    
    parser.add_argument(
        '--text',
        nargs='+',
        type=str,
        help='Text(s) to classify (for inference mode)'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data file (for evaluate mode)'
    )
    
    parser.add_argument(
        '--text-col',
        type=str,
        default='text',
        help='Name of text column in test data (default: text)'
    )
    
    parser.add_argument(
        '--label-col',
        type=str,
        default='label',
        help='Name of label column in test data (default: label)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save evaluation results'
    )
    
    parser.add_argument(
        '--check-models',
        action='store_true',
        help='Check which models are available'
    )
    
    args = parser.parse_args()
    
    try:
        # Check models if requested
        if args.check_models:
            check_models(args.models_path)
            return
        
        # Validate mode-specific arguments
        if args.mode == 'inference':
            if not args.text:
                parser.error("--text is required for inference mode")
            run_inference(args.text, args.models_path)
            
        elif args.mode == 'evaluate':
            if not args.test_data:
                parser.error("--test-data is required for evaluate mode")
            run_evaluation(
                test_data_path=args.test_data,
                models_path=args.models_path,
                text_col=args.text_col,
                label_col=args.label_col,
                save_results=not args.no_save
            )
        
        else:
            # If no mode specified, show help
            parser.print_help()
            print("\n" + "="*70)
            print("Quick Start Examples:")
            print("="*70)
            print("\n1. Check available models:")
            print("   python main.py --check-models")
            print("\n2. Run inference:")
            print('   python main.py --mode inference --text "Sample text to classify"')
            print("\n3. Evaluate on test data:")
            print("   python main.py --mode evaluate --test-data data/test.csv")
            print("\n" + "="*70 + "\n")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()