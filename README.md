# Fake News Detector - BERT & RoBERTa Inference System

A production-ready inference system for fake news detection using trained BERT and RoBERTa models with ensemble capabilities.

---

## 📋 Features

✅ **Multi-Model Support**
- BERT model inference
- RoBERTa model inference
- Ensemble predictions (average, voting, weighted)

✅ **Flexible Usage**
- Interactive mode for real-time analysis
- Single text analysis
- Batch processing
- Full evaluation on test datasets

✅ **Comprehensive Results**
- Prediction with confidence scores
- Risk level assessment
- Detailed probability breakdown
- Formatted recommendations

✅ **Easy Testing**
- Automated testing on TSV datasets
- Model comparison
- Performance metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Confusion matrices

---

## 🗂️ Project Structure

```
.
├── core/                          # Core utilities (future use)
├── data/
│   └── test_file.tsv             # Test dataset (TSV format)
├── models/
│   ├── bert/
│   │   └── bert/
│   │       └── final_model/      # BERT model checkpoint
│   │           └── model.safetensors
│   └── robert/
│       └── final_model/          # RoBERTa model checkpoint
│           ├── config.json
│           ├── model.safetensors
│           └── ... (tokenizer files)
├── services/
│   ├── __init__.py
│   ├── inference.py              # Model inference service
│   └── test.py                   # Testing & evaluation
├── training/                      # Training code (if needed)
├── main.py                        # Main entry point ⭐
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Model Files

Make sure your trained models are in place:
- BERT: `models/bert/bert/final_model/model.safetensors`
- RoBERTa: `models/robert/final_model/model.safetensors`

### 3. Run Interactive Mode (Default)

```bash
python main.py
```

Or explicitly:

```bash
python main.py --mode interactive
```

---

## 💻 Usage Examples

### 🎮 Interactive Mode

Start an interactive session:

```bash
python main.py --mode interactive
```

Then enter texts to analyze:

```
Enter text to analyze: Scientists discover breakthrough in renewable energy
[Results displayed...]

Enter text to analyze: You won't believe this shocking miracle cure!
[Results displayed...]
```

### 📝 Single Text Analysis

Analyze a single text:

```bash
python main.py --mode single --text "Breaking: New research shows promising results"
```

With specific model:

```bash
# Use BERT only
python main.py --mode single --text "Your text here" --model bert

# Use RoBERTa only
python main.py --mode single --text "Your text here" --model roberta

# Use ensemble (default)
python main.py --mode single --text "Your text here" --model ensemble
```

Choose ensemble method:

```bash
# Average probabilities (default)
python main.py --mode single --text "Text" --ensemble-method average

# Majority voting
python main.py --mode single --text "Text" --ensemble-method voting

# Weighted average (BERT: 0.4, RoBERTa: 0.6)
python main.py --mode single --text "Text" --ensemble-method weighted
```

### 📦 Batch Processing

Create a text file (one text per line):

```bash
# Create input file
cat > texts.txt << EOF
Scientists announce breakthrough in cancer research.
You won't believe this shocking discovery!
Government announces new policy on education.
EOF

# Process batch
python main.py --mode batch --file texts.txt
```

### 📊 Full Evaluation

Run complete evaluation on test dataset:

```bash
python main.py --mode evaluate --test-data data/test_file.tsv
```

This will:
- Test all available models (BERT, RoBERTa, Ensemble)
- Calculate comprehensive metrics
- Display confusion matrices
- Compare model performance

---

## 🧪 Testing with test.py

For detailed testing and evaluation:

```bash
# Test all models
python services/test.py --data data/test_file.tsv --model all

# Test specific model
python services/test.py --data data/test_file.tsv --model bert
python services/test.py --data data/test_file.tsv --model roberta
python services/test.py --data data/test_file.tsv --model ensemble

# Custom ensemble method
python services/test.py --data data/test_file.tsv --model ensemble --ensemble-method voting

# Use CPU
python services/test.py --data data/test_file.tsv --device cpu
```

---

## 📄 Data Format

### Test Data (TSV Format)

Your `data/test_file.tsv` should be tab-separated:

```
0	This is a real news article text...
1	This is a fake news article text...
```

**Format:**
- Column 1: Label (0 = Real, 1 = Fake)
- Column 2: Text content
- Separator: TAB character

---

## 🎯 Output Format

### Single Text Analysis

```
======================================================================
DETECTION RESULT
======================================================================

🚫 PREDICTION: FAKE NEWS
📊 Confidence: 89.23%
⚠️  Risk Level: HIGH
🤖 Model: ENSEMBLE (average)

----------------------------------------------------------------------
CLASS PROBABILITIES:
  Real News: 10.77%
  Fake News: 89.23%

----------------------------------------------------------------------
TEXT PREVIEW:
  You won't believe this shocking miracle cure that doctors don't...

----------------------------------------------------------------------
RECOMMENDATION:
  ⛔ This content is likely FAKE NEWS.
  ⚠️  Do NOT share or trust this information.
  ℹ️  Verify with trusted sources before believing.
======================================================================
```

### Evaluation Results

```
======================================================================
BERT RESULTS
======================================================================
Accuracy:  0.7508 (75.08%)
Precision: 0.8649 (86.49%)
Recall:    0.7053 (70.53%)
F1-Score:  0.7770 (77.70%)
AUC-ROC:   0.8317 (83.17%)
======================================================================

Confusion Matrix:
                Predicted
              Real    Fake
Actual Real   45      5
       Fake   8       42

Per-Class Accuracy:
Real news: 0.9000 (45/50)
Fake news: 0.8400 (42/50)
======================================================================
```

---

## 🔧 Configuration Options

### Command Line Arguments

#### Main Script (main.py)

**Mode Options:**
- `--mode interactive`: Interactive text input (default)
- `--mode single`: Analyze single text
- `--mode batch`: Process multiple texts from file
- `--mode evaluate`: Full evaluation on test data

**Input Options:**
- `--text "TEXT"`: Text to analyze (single mode)
- `--file PATH`: File with texts (batch mode)
- `--test-data PATH`: Test dataset path (evaluate mode)

**Model Options:**
- `--model bert`: Use BERT only
- `--model roberta`: Use RoBERTa only
- `--model ensemble`: Use ensemble (default)
- `--ensemble-method average|voting|weighted`: Ensemble strategy

**System Options:**
- `--device cuda|cpu`: Computation device
- `--simple`: Simplified output
- `--bert-model PATH`: Custom BERT model path
- `--roberta-model PATH`: Custom RoBERTa model path

#### Test Script (services/test.py)

- `--data PATH`: Test data path
- `--model bert|roberta|ensemble|all`: Model to test
- `--ensemble-method average|voting|weighted`: Ensemble method
- `--batch-size N`: Batch size for inference
- `--device cuda|cpu`: Device

---

## 📊 Ensemble Methods

### 1. Average (Default)

Averages the probability outputs from both models:

```python
ensemble_prob = (bert_prob + roberta_prob) / 2
```

**Best for:** Balanced predictions

### 2. Voting

Uses majority voting from model predictions:

```python
if bert_pred == roberta_pred:
    ensemble_pred = bert_pred
else:
    ensemble_pred = bert_pred  # Tie-breaker
```

**Best for:** High confidence when models agree

### 3. Weighted

Weighted average with custom weights (BERT: 0.4, RoBERTa: 0.6):

```python
ensemble_prob = 0.4 * bert_prob + 0.6 * roberta_prob
```

**Best for:** Leveraging stronger model (RoBERTa typically better)

---

## 🎨 Risk Levels

The system automatically assigns risk levels based on confidence:

| Confidence | Fake Prediction | Real Prediction |
|------------|----------------|-----------------|
| ≥ 90% | VERY HIGH | VERY LOW |
| ≥ 75% | HIGH | LOW |
| ≥ 60% | MODERATE | MODERATE |
| < 60% | UNCERTAIN | UNCERTAIN |

---

## 🐛 Troubleshooting

### Model Loading Errors

**Issue:** "Error loading BERT/RoBERTa model"

**Solutions:**
1. Check model file paths
2. Ensure `model.safetensors` exists
3. Verify model directory structure
4. Try reinstalling safetensors: `pip install -U safetensors`

### CUDA Errors

**Issue:** CUDA out of memory or not available

**Solutions:**
```bash
# Use CPU instead
python main.py --device cpu

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Data Loading Errors

**Issue:** Error reading test data

**Solutions:**
1. Verify file exists
2. Check TSV format (tab-separated)
3. Ensure labels are 0 or 1
4. Check file encoding (should be UTF-8)

---

## 📈 Performance Tips

1. **Use GPU for faster inference**
   ```bash
   python main.py --device cuda
   ```

2. **Batch processing for multiple texts**
   - Much faster than individual predictions
   - Use `--mode batch` instead of multiple single calls

3. **Choose appropriate ensemble method**
   - `average`: Best general performance
   - `voting`: Best when high confidence needed
   - `weighted`: Best if one model is stronger

4. **Adjust batch size for memory**
   ```bash
   python services/test.py --batch-size 16  # Default: 32
   ```

---

## 🔍 Example Workflows

### Workflow 1: Quick Text Check

```bash
# Check if news is fake
python main.py --mode single \
  --text "Breaking: Miracle cure discovered!" \
  --simple
```

### Workflow 2: Evaluate All Models

```bash
# Compare BERT, RoBERTa, and Ensemble
python services/test.py \
  --data data/test_file.tsv \
  --model all \
  --ensemble-method average
```

### Workflow 3: Process News Articles

```bash
# Create file with articles
echo "Article 1 text..." > articles.txt
echo "Article 2 text..." >> articles.txt

# Analyze all
python main.py --mode batch --file articles.txt --model ensemble
```

### Workflow 4: Interactive Analysis

```bash
# Start interactive session
python main.py

# Type or paste texts
# Get instant feedback
# Type 'quit' to exit
```

---

## 📚 API Usage (Python)

### Basic Usage

```python
from services.inference import ModelInference

# Initialize
inference = ModelInference(
    bert_model_path='models/bert/bert/final_model',
    roberta_model_path='models/robert/final_model',
    device='cuda'
)

# Single prediction
texts = ["Your news article here"]
results = inference.predict(texts, model='ensemble')

print(f"Prediction: {results['predictions'][0]}")  # 0 or 1
print(f"Probability: {results['probabilities'][0]}")  # [p_real, p_fake]
```

### Advanced Usage

```python
from main import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector(device='cuda')

# Analyze text
interpretation = detector.analyze_text(
    text="Scientists discover breakthrough...",
    model='ensemble',
    ensemble_method='average'
)

# Access results
print(interpretation['prediction'])     # "REAL NEWS" or "FAKE NEWS"
print(interpretation['confidence'])     # Confidence percentage
print(interpretation['risk_level'])     # Risk assessment
```

---

## 🙏 Credits

This inference system uses:
- **BERT**: Bidirectional Encoder Representations from Transformers (Google)
- **RoBERTa**: Robustly Optimized BERT Approach (Facebook AI)
- **Transformers**: Hugging Face library
- **PyTorch**: Deep learning framework

---

## 📝 Notes

- Models must be trained before inference
- Default paths assume standard model structure
- GPU recommended for faster inference
- Ensemble typically provides best results

---

## 🆘 Support

For issues or questions:
1. Check troubleshooting section
2. Verify model files and paths
3. Test with sample data first
4. Check logs for detailed errors

---

**Ready to detect fake news! 🎉**