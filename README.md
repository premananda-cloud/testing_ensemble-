# Multi-Model Ensemble Evaluation System

A comprehensive evaluation framework for assessing ensemble performance of BERT, RoBERTa, and TF-IDF models trained using the Self-Paced Supervised Teacher (SPST) methodology.

## Overview

This repository provides tools to evaluate models trained via the [SPST-based BERT training pipeline](https://github.com/premananda-cloud/Bert_training_via_SPST), which implements the methodology from [BERT Fake News Detection using SPST](https://github.com/premananda-cloud/Beert_fake_news_SPST).

The evaluation system combines multiple trained models into an ensemble and provides comprehensive performance metrics, enabling rigorous assessment of model effectiveness for text classification tasks.

## Features

- **Multi-Model Ensemble Evaluation**: Assess BERT, RoBERTa, and TF-IDF models individually and as an ensemble
- **Comprehensive Metrics**: Calculate accuracy, precision, recall, F1-score, and confusion matrices
- **Flexible Data Support**: Compatible with CSV, Excel, JSON, and text file formats
- **Detailed Reporting**: Generate performance reports, visualizations, and per-sample analysis
- **Configurable Weights**: Adjust ensemble contribution weights for optimal performance

## Related Repositories

- **Training Pipeline**: [Bert_training_via_SPST](https://github.com/premananda-cloud/Bert_training_via_SPST) - Train models using SPST methodology
- **Research Paper**: [Beert_fake_news_SPST](https://github.com/premananda-cloud/Beert_fake_news_SPST) - Original SPST research implementation

## Project Structure

```
.
├── main.py                    # Command-line evaluation interface
├── examples.py                # Usage examples and demonstrations
├── requirements.txt           # Python dependencies
├── services/
│   ├── __init__.py           # Package initialization
│   ├── init.py               # Model loading utilities
│   ├── inference.py          # Ensemble prediction engine
│   └── test.py               # Evaluation and metrics calculation
├── models/
│   ├── bert/                 # BERT model files (from SPST training)
│   ├── robert/               # RoBERTa model files
│   └── tf_idf/               # TF-IDF model files
└── results/                  # Evaluation outputs (generated)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/premananda-cloud/testing_ensemble-.git
cd testing_ensemble-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- torch
- transformers
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- joblib

3. Place your trained models in the `models/` directory following the expected structure:
```
models/
├── bert/bert/final_model/
├── robert/final_model/
└── tf_idf/tfidf_model.joblib
```

## Usage

### Quick Start

1. **Verify Model Availability**:
```bash
python main.py --check-models
```

2. **Evaluate on Test Dataset**:
```bash
python main.py --mode evaluate --test-data data/test.csv
```

3. **Evaluate with Custom Column Names**:
```bash
python main.py --mode evaluate --test-data data/test.csv --text-col review --label-col sentiment
```

### Python API

#### Comprehensive Model Evaluation

```python
from services.test import evaluate_models

# Evaluate all models on test data
metrics = evaluate_models(
    test_data_path="data/test.csv",
    models_base_path="models",
    text_col="text",
    label_col="label",
    save_results=True
)

# Access ensemble performance
print(f"Ensemble Accuracy: {metrics['ensemble']['accuracy']:.4f}")
print(f"Ensemble F1-Score: {metrics['ensemble']['f1_score']:.4f}")

# Compare individual model performance
print(f"BERT F1-Score: {metrics['bert']['f1_score']:.4f}")
print(f"RoBERTa F1-Score: {metrics['roberta']['f1_score']:.4f}")
print(f"TF-IDF F1-Score: {metrics['tfidf']['f1_score']:.4f}")
```

#### Custom Evaluation Pipeline

```python
from services.inference import ModelEnsemble
from services.test import ModelEvaluator

# Initialize ensemble with custom weights
ensemble = ModelEnsemble(
    models_base_path="models",
    weights={'bert': 0.5, 'roberta': 0.3, 'tfidf': 0.2}
)

# Create evaluator
evaluator = ModelEvaluator(ensemble)

# Load test data
texts, labels = evaluator.load_test_data("data/test.csv")

# Run evaluation
metrics = evaluator.evaluate(
    texts=texts,
    true_labels=labels,
    save_path="results/custom_evaluation"
)
```

#### Single Text Classification (Optional)

```python
from services.inference import ModelEnsemble

ensemble = ModelEnsemble()
prediction = ensemble.predict("Sample text for classification")

print(f"Prediction: {prediction.ensemble_pred}")
print(f"Confidence: {prediction.confidence:.4f}")
```

## Test Data Format

### CSV/Excel Format

Your test data should contain text and label columns:

```csv
text,label
"This is a sample text for evaluation",1
"Another sample for testing the model",0
```

### Text File Format

Tab-separated format:
```
0	This is a negative sample
1	This is a positive sample
```

### JSON Format

```json
[
  {"text": "Sample text one", "label": 1},
  {"text": "Sample text two", "label": 0}
]
```

## Evaluation Outputs

When `save_results=True`, the system generates:

```
results/evaluation/
├── metrics.json              # Detailed performance metrics
├── detailed_predictions.csv  # Per-sample predictions and correctness
├── confusion_matrix.png      # Visual confusion matrix
└── evaluation_summary.txt    # Human-readable performance report
```

### Metrics Calculated

For each model (BERT, RoBERTa, TF-IDF) and the ensemble:

- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction accuracy (weighted)
- **Recall**: True positive detection rate (weighted)
- **F1-Score**: Harmonic mean of precision and recall (weighted)
- **Confusion Matrix**: Detailed classification breakdown
- **Classification Report**: Per-class performance metrics

Additionally for ensemble:
- **Confidence Statistics**: Mean, std, min, max, median confidence scores

## Ensemble Configuration

### Default Weights

```python
weights = {
    'bert': 0.4,      # 40% contribution
    'roberta': 0.4,   # 40% contribution
    'tfidf': 0.2      # 20% contribution
}
```

### Custom Weights

Adjust weights based on individual model performance:

```python
ensemble = ModelEnsemble(
    models_base_path="models",
    weights={
        'bert': 0.5,      # Increase BERT weight
        'roberta': 0.35,
        'tfidf': 0.15
    }
)
```

## Examples

See `examples.py` for comprehensive usage demonstrations:

```bash
# Run evaluation example
python examples.py 3

# Run custom evaluation with error analysis
python examples.py 8

# Create sample test data
python examples.py 9
```

## Performance Analysis

The evaluation system provides insights into:

1. **Overall Model Performance**: Compare accuracy, F1-score across models
2. **Ensemble Effectiveness**: Assess if ensemble outperforms individual models
3. **Confidence Analysis**: Identify low-confidence predictions requiring review
4. **Error Analysis**: Examine misclassified samples for model improvement
5. **Class-wise Performance**: Understand per-class strengths and weaknesses

## Advanced Usage

### Batch Evaluation Across Multiple Datasets

```python
from services.test import evaluate_models

datasets = ['test_set_1.csv', 'test_set_2.csv', 'test_set_3.csv']

for dataset in datasets:
    print(f"\nEvaluating on {dataset}...")
    metrics = evaluate_models(
        test_data_path=dataset,
        save_results=True,
        output_dir=f"results/{dataset.split('.')[0]}"
    )
    print(f"Accuracy: {metrics['ensemble']['accuracy']:.4f}")
```

### Error Analysis

```python
from services.inference import ModelEnsemble
from services.test import ModelEvaluator

ensemble = ModelEnsemble()
evaluator = ModelEvaluator(ensemble)

texts, labels = evaluator.load_test_data("test.csv")
predictions = ensemble.predict_batch(texts)

# Identify misclassifications
errors = [
    (text, true, pred.ensemble_pred, pred.confidence)
    for text, true, pred in zip(texts, labels, predictions)
    if true != pred.ensemble_pred
]

print(f"Error Rate: {len(errors)/len(texts)*100:.2f}%")
for text, true, pred, conf in errors[:5]:
    print(f"\nText: {text[:100]}...")
    print(f"True: {true}, Predicted: {pred}, Confidence: {conf:.4f}")
```

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster evaluation)
- 8GB+ RAM (16GB+ recommended for large models)

## Troubleshooting

### Models Not Found

```bash
# Verify model paths
python main.py --check-models

# Check directory structure
ls -la models/bert/bert/final_model/
ls -la models/robert/final_model/
ls -la models/tf_idf/
```

### Out of Memory

The system automatically falls back to CPU if GPU memory is insufficient. For large datasets:
- Process in smaller batches
- Evaluate one model at a time
- Use a machine with more RAM

### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/testing_ensemble-

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## Citation

If you use this evaluation framework in your research, please cite the related work:

```bibtex
@article{spst_fake_news,
  title={BERT-based Fake News Detection using Self-Paced Supervised Teacher},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={[Year]},
  url={https://github.com/premananda-cloud/Beert_fake_news_SPST}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add evaluation feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project follows the same license as the [SPST training pipeline](https://github.com/premananda-cloud/Bert_training_via_SPST).

## Acknowledgments

- Based on the Self-Paced Supervised Teacher (SPST) methodology
- Models trained using [Bert_training_via_SPST](https://github.com/premananda-cloud/Bert_training_via_SPST)
- Research foundation: [Beert_fake_news_SPST](https://github.com/premananda-cloud/Beert_fake_news_SPST)

## Support

For issues or questions:
1. Check the [Quickstart Guide](Quickstart.md)
2. Review [examples.py](examples.py) for usage patterns
3. Open an issue on GitHub with:
   - Error message/logs
   - Steps to reproduce
   - System information (OS, Python version, GPU/CPU)

## Roadmap

- [ ] Add support for additional metrics (ROC-AUC, PR curves)
- [ ] Implement cross-validation evaluation
- [ ] Add model comparison visualizations
- [ ] Support for multi-class classification (>2 classes)
- [ ] Integration with MLflow for experiment tracking
- [ ] Automated hyperparameter tuning for ensemble weights

---

**Maintained by**: [Your Name/Organization]  
**Last Updated**: January 2026  
**Status**: Active Development
