"""
Main Script - Fake News Detector
Provides an interactive interface for fake news detection using BERT and RoBERTa models
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from services import ModelInference, ModelTester
from services.test import ModelTester
from typing import List, Dict
import numpy as np


class FakeNewsDetector:
    """
    Main interface for fake news detection.
    """
    
    def __init__(
        self,
        bert_model_path: str = 'models/bert/bert/final_model',
        roberta_model_path: str = 'models/robert/final_model',
        device: str = 'cuda'
    ):
        """
        Initialize detector.
        
        Args:
            bert_model_path: Path to BERT model
            roberta_model_path: Path to RoBERTa model
            device: Device for inference
        """
        print(f"\n{'='*70}")
        print("FAKE NEWS DETECTOR")
        print(f"{'='*70}\n")
        
        # Initialize inference service
        print("Loading models...")
        self.inference = ModelInference(
            bert_model_path=bert_model_path if Path(bert_model_path).exists() else None,
            roberta_model_path=roberta_model_path if Path(roberta_model_path).exists() else None,
            device=device
        )
        
        available = self.inference.get_available_models()
        if not available:
            raise ValueError("No models could be loaded")
        
        print(f"\n✓ Loaded models: {', '.join(available)}")
        print(f"{'='*70}\n")
    
    def interpret_prediction(
        self,
        text: str,
        prediction: int,
        probabilities: np.ndarray,
        model_name: str = 'Ensemble'
    ) -> Dict:
        """
        Interpret and format prediction results.
        
        Args:
            text: Input text
            prediction: Predicted class (0=Real, 1=Fake)
            probabilities: Class probabilities
            model_name: Name of the model used
        
        Returns:
            Formatted interpretation
        """
        label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        confidence = probabilities[prediction] * 100
        
        # Determine risk level
        if confidence >= 90:
            risk = "VERY HIGH" if prediction == 1 else "VERY LOW"
        elif confidence >= 75:
            risk = "HIGH" if prediction == 1 else "LOW"
        elif confidence >= 60:
            risk = "MODERATE"
        else:
            risk = "UNCERTAIN"
        
        # Create interpretation
        interpretation = {
            'text': text,
            'prediction': label,
            'confidence': confidence,
            'risk_level': risk,
            'model': model_name,
            'probabilities': {
                'real': probabilities[0] * 100,
                'fake': probabilities[1] * 100
            }
        }
        
        return interpretation
    
    def display_result(self, interpretation: Dict, detailed: bool = True):
        """
        Display prediction result in a formatted way.
        
        Args:
            interpretation: Interpretation dictionary
            detailed: Whether to show detailed information
        """
        print(f"\n{'='*70}")
        print("DETECTION RESULT")
        print(f"{'='*70}")
        
        # Main prediction
        pred_symbol = "🚫" if "FAKE" in interpretation['prediction'] else "✓"
        print(f"\n{pred_symbol} PREDICTION: {interpretation['prediction']}")
        print(f"📊 Confidence: {interpretation['confidence']:.2f}%")
        print(f"⚠️  Risk Level: {interpretation['risk_level']}")
        print(f"🤖 Model: {interpretation['model']}")
        
        if detailed:
            print(f"\n{'-'*70}")
            print("CLASS PROBABILITIES:")
            print(f"  Real News: {interpretation['probabilities']['real']:.2f}%")
            print(f"  Fake News: {interpretation['probabilities']['fake']:.2f}%")
            
            print(f"\n{'-'*70}")
            print("TEXT PREVIEW:")
            text = interpretation['text']
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"  {preview}")
        
        # Recommendation
        print(f"\n{'-'*70}")
        print("RECOMMENDATION:")
        if "FAKE" in interpretation['prediction']:
            if interpretation['confidence'] >= 75:
                print("  ⛔ This content is likely FAKE NEWS.")
                print("  ⚠️  Do NOT share or trust this information.")
                print("  ℹ️  Verify with trusted sources before believing.")
            else:
                print("  ⚠️  This content may be unreliable.")
                print("  ℹ️  Cross-check with multiple trusted sources.")
        else:
            if interpretation['confidence'] >= 75:
                print("  ✓ This content appears to be REAL NEWS.")
                print("  ℹ️  However, always verify important information.")
            else:
                print("  ℹ️  This content seems credible but verify if important.")
        
        print(f"{'='*70}\n")
    
    def analyze_text(
        self,
        text: str,
        model: str = 'ensemble',
        ensemble_method: str = 'average',
        detailed: bool = True
    ) -> Dict:
        """
        Analyze a single text.
        
        Args:
            text: Text to analyze
            model: Model to use ('bert', 'roberta', 'ensemble')
            ensemble_method: Ensemble method if using ensemble
            detailed: Whether to display detailed results
        
        Returns:
            Interpretation dictionary
        """
        # Predict
        results = self.inference.predict([text], model=model, ensemble_method=ensemble_method)
        
        # Interpret
        model_display = f"{model.upper()} ({ensemble_method})" if model == 'ensemble' else model.upper()
        interpretation = self.interpret_prediction(
            text=text,
            prediction=results['predictions'][0],
            probabilities=results['probabilities'][0],
            model_name=model_display
        )
        
        # Display
        if detailed:
            self.display_result(interpretation, detailed=True)
        
        return interpretation
    
    def analyze_batch(
        self,
        texts: List[str],
        model: str = 'ensemble',
        ensemble_method: str = 'average'
    ) -> List[Dict]:
        """
        Analyze multiple texts.
        
        Args:
            texts: List of texts to analyze
            model: Model to use
            ensemble_method: Ensemble method
        
        Returns:
            List of interpretations
        """
        print(f"\nAnalyzing {len(texts)} texts with {model.upper()}...")
        
        # Predict
        results = self.inference.predict(texts, model=model, ensemble_method=ensemble_method)
        
        # Interpret all
        interpretations = []
        for i in range(len(texts)):
            model_display = f"{model.upper()} ({ensemble_method})" if model == 'ensemble' else model.upper()
            interpretation = self.interpret_prediction(
                text=texts[i],
                prediction=results['predictions'][i],
                probabilities=results['probabilities'][i],
                model_name=model_display
            )
            interpretations.append(interpretation)
        
        # Summary
        fake_count = sum(1 for i in interpretations if "FAKE" in i['prediction'])
        real_count = len(interpretations) - fake_count
        
        print(f"\n{'='*70}")
        print("BATCH ANALYSIS SUMMARY")
        print(f"{'='*70}")
        print(f"Total texts: {len(texts)}")
        print(f"Real news: {real_count} ({real_count/len(texts)*100:.1f}%)")
        print(f"Fake news: {fake_count} ({fake_count/len(texts)*100:.1f}%)")
        print(f"{'='*70}\n")
        
        return interpretations
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("Starting interactive mode...")
        print("Type your text and press Enter (or 'quit' to exit)\n")
        
        while True:
            try:
                text = input("Enter text to analyze: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("\nExiting interactive mode. Goodbye!")
                    break
                
                if not text:
                    continue
                
                # Analyze
                self.analyze_text(text, model='ensemble', ensemble_method='average')
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def run_evaluation(self, test_data_path: str = 'data/test_file.tsv'):
        """
        Run full evaluation on test dataset.
        
        Args:
            test_data_path: Path to test data
        """
        tester = ModelTester(self.inference)
        
        # Load test data
        texts, labels = tester.load_test_data(test_data_path)
        
        # Test all models
        results = tester.test_all_models(texts, labels, ensemble_method='average')
        tester.compare_models(results)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fake News Detector - BERT & RoBERTa Ensemble'
    )
    
    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        default='interactive',
        choices=['interactive', 'single', 'batch', 'evaluate'],
        help='Operation mode'
    )
    
    # Input
    parser.add_argument(
        '--text',
        type=str,
        help='Single text to analyze (for single mode)'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='File with texts to analyze (for batch mode)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default='data/test_file.tsv',
        help='Test data path (for evaluate mode)'
    )
    
    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='ensemble',
        choices=['bert', 'roberta', 'ensemble'],
        help='Model to use'
    )
    parser.add_argument(
        '--ensemble-method',
        type=str,
        default='average',
        choices=['average', 'voting', 'weighted'],
        help='Ensemble method'
    )
    
    # Paths
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
    
    # System
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Simple output (less detailed)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Initialize detector
    try:
        detector = FakeNewsDetector(
            bert_model_path=args.bert_model,
            roberta_model_path=args.roberta_model,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Run based on mode
    if args.mode == 'interactive':
        detector.interactive_mode()
    
    elif args.mode == 'single':
        if not args.text:
            print("Error: --text required for single mode")
            return
        detector.analyze_text(
            args.text,
            model=args.model,
            ensemble_method=args.ensemble_method,
            detailed=not args.simple
        )
    
    elif args.mode == 'batch':
        if not args.file:
            print("Error: --file required for batch mode")
            return
        
        # Read texts from file
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = detector.analyze_batch(
            texts,
            model=args.model,
            ensemble_method=args.ensemble_method
        )
        
        # Display individual results
        for i, result in enumerate(results, 1):
            print(f"\nText {i}:")
            detector.display_result(result, detailed=not args.simple)
    
    elif args.mode == 'evaluate':
        detector.run_evaluation(args.test_data)


if __name__ == "__main__":
    main()