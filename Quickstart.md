# Quick Start Guide - Fake News Detector

## 🚀 Get Started in 3 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Model Files

Check that your models are in place:

```bash
# BERT model
ls models/bert/bert/final_model/model.safetensors

# RoBERTa model  
ls models/robert/final_model/model.safetensors
```

### Step 3: Run!

**Interactive Mode (Easiest):**

```bash
python main.py
```

Then type or paste text and press Enter!

---

## 📋 Common Commands

### Analyze Single Text

```bash
python main.py --mode single --text "Scientists discover breakthrough in AI"
```

### Test Your Models

```bash
python services/test.py --data data/test_file.tsv --model all
```

### Process Multiple Texts

```bash
# Create a file with texts (one per line)
cat > my_texts.txt << EOF
Scientists announce breakthrough in renewable energy.
You won't believe this shocking miracle!
Government proposes new healthcare policy.
EOF

# Analyze all at once
python main.py --mode batch --file my_texts.txt
```

---

## 🎯 Model Options

| Command | Description |
|---------|-------------|
| `--model bert` | Use BERT only |
| `--model roberta` | Use RoBERTa only |
| `--model ensemble` | Use both (default, best) |

**Ensemble Methods:**
- `--ensemble-method average` - Average probabilities (default)
- `--ensemble-method voting` - Majority vote
- `--ensemble-method weighted` - Weighted (BERT: 40%, RoBERTa: 60%)

---

## 💡 Examples

### Example 1: Check News Article

```bash
python main.py --mode single \
  --text "Breaking: World leaders meet to discuss climate change solutions" \
  --model ensemble
```

**Output:**
```
✓ PREDICTION: REAL NEWS
📊 Confidence: 87.45%
⚠️  Risk Level: LOW
```

### Example 2: Compare All Models

```bash
python services/test.py \
  --data data/test_file.tsv \
  --model all
```

**Shows:**
- BERT performance
- RoBERTa performance  
- Ensemble performance
- Model comparison table

### Example 3: Interactive Session

```bash
python main.py
```

```
Enter text to analyze: Scientists discover new planet
✓ PREDICTION: REAL NEWS
Confidence: 82.3%

Enter text to analyze: One weird trick to lose weight!
🚫 PREDICTION: FAKE NEWS
Confidence: 91.7%

Enter text to analyze: quit
Exiting...
```

---

## 🔍 Understanding Results

### Prediction

- **REAL NEWS** ✓ - Content appears legitimate
- **FAKE NEWS** 🚫 - Content likely unreliable

### Confidence

- **90-100%** - Very confident
- **75-89%** - High confidence
- **60-74%** - Moderate confidence  
- **<60%** - Uncertain

### Risk Level

- **VERY HIGH/HIGH** - Strong fake news indicators
- **MODERATE** - Mixed signals, verify carefully
- **LOW/VERY LOW** - Likely credible

---

## ⚡ Performance Tips

1. **Use GPU for speed:**
   ```bash
   python main.py --device cuda
   ```

2. **Batch process multiple texts** instead of one-by-one

3. **Use ensemble** for best accuracy

4. **Use CPU if GPU unavailable:**
   ```bash
   python main.py --device cpu
   ```

---

## 🐛 Quick Troubleshooting

**Problem:** Models won't load
```bash
# Check paths
ls -la models/bert/bert/final_model/
ls -la models/robert/final_model/
```

**Problem:** CUDA error
```bash
# Use CPU instead
python main.py --device cpu
```

**Problem:** Data file error  
```bash
# Check format (tab-separated)
head data/test_file.tsv
```

---

## 📚 More Help

- Full documentation: `README.md`
- Test specific models: `python services/test.py --help`
- Main options: `python main.py --help`

---

## 🎉 You're Ready!

Start with interactive mode:
```bash
python main.py
```

Or analyze a quick text:
```bash
python main.py --mode single --text "Your news text here"
```

Happy fake news detecting! 🔍
