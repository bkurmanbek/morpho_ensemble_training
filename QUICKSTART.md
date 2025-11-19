# Quick Start Guide

Get started with the Kazakh Morphology Ensemble in 5 minutes.

## Prerequisites

```bash
# Hardware
- GPU with 24GB+ RAM (RTX 4090, A100, etc.)
- Or use Google Colab/AWS/Azure for training

# Software
- Python 3.9+
- CUDA 11.8+
```

## Installation (2 minutes)

```bash
# Clone repository
git clone <repo_url>
cd kazakh_morphology_ensemble

# Install dependencies
pip install -r requirements.txt

# Download your data files
# - all_structured_kazakh_data.json (training data)
# - all_kazakh_grammar_data.json (grammar definitions)
```

## Option A: Use Pre-trained Models (Recommended)

If you have pre-trained models, skip to inference:

```python
from ensemble_model import create_ensemble

# Create ensemble with your models
ensemble = create_ensemble(
    grammar_data_path="all_kazakh_grammar_data.json",
    pattern_db_path="pattern_database.json"
)

# Load models (takes 2-3 minutes)
ensemble.load_models()

# Predict!
result = ensemble.predict(word="кітап", pos_tag="Зат есім")
print(result.to_dict())
```

## Option B: Train Your Own Models

### Step 1: Quick Training (Light Version)

For testing, use smaller models:

```bash
# Build pattern database (5 minutes)
python train_sozjasam.py \
    --mode build_db \
    --data_path all_structured_kazakh_data.json \
    --pattern_db_path pattern_database.json

# Train light structure model (2-3 hours on RTX 4090)
python train_structure.py \
    --data_path all_structured_kazakh_data.json \
    --grammar_path all_kazakh_grammar_data.json \
    --output_dir ./models/structure_model \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --num_epochs 2 \
    --batch_size 4 \
    --use_qlora

# Skip lexical model for now (use structure model's limited meanings)

# Train sozjasam model (1-2 hours)
python train_sozjasam.py \
    --mode train \
    --data_path all_structured_kazakh_data.json \
    --pattern_db_path pattern_database.json \
    --output_dir ./models/sozjasam_model \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --num_epochs 3
```

### Step 2: Test Your Models

```python
from ensemble_model import MorphologyEnsemble

# Create ensemble with your trained models
ensemble = MorphologyEnsemble(
    grammar_data=grammar_data,
    structure_model_path="./models/structure_model",
    lexical_model_path="./models/structure_model",  # Reuse structure model
    sozjasam_model_path="./models/sozjasam_model",
    pattern_db_path="pattern_database.json"
)

ensemble.load_models()

# Test
result = ensemble.predict("кітап", "Зат есім")
print(f"Confidence: {result.confidence:.2f}")
print(result.to_dict())
```

### Step 3: Full Training (Production)

When ready for production, train all models:

```bash
# Run full training pipeline (17-22 hours total)
bash train_all.sh
```

## Common Workflows

### Workflow 1: Single Word Analysis

```python
from ensemble_model import create_ensemble

ensemble = create_ensemble("all_kazakh_grammar_data.json", "pattern_database.json")
ensemble.load_models()

# Analyze word
result = ensemble.predict("кітап", "Зат есім")

# Check quality
print(f"Confidence: {result.confidence:.2f}")
if result.confidence < 0.7:
    print("Low confidence - review manually")
```

### Workflow 2: Batch Processing

```python
# Process 100 words
words = [("кітап", "Зат есім"), ("жылдам", "Сын есім"), ...]  # 100 items

results = ensemble.predict_batch(words, batch_size=16)

# Save results
import json
with open("results.json", "w", encoding="utf-8") as f:
    output = [r.to_dict() for r in results]
    json.dump(output, f, ensure_ascii=False, indent=2)
```

### Workflow 3: API Server

```bash
# Start API server
python deployment.py --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"word": "кітап", "pos_tag": "Зат есім"}'
```

### Workflow 4: Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --test_data test_data.json \
    --grammar_data all_kazakh_grammar_data.json \
    --pattern_db pattern_database.json
```

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Use smaller batch size
```python
results = ensemble.predict_batch(words, batch_size=4)  # Instead of 16
```

**Solution 2**: Use 8-bit quantization
```python
# In model loading, add:
load_in_8bit=True
```

**Solution 3**: Use CPU (slow but works)
```python
# Set device_map
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu"
)
```

### Issue: Slow Inference

**Solution 1**: Batch process
```python
# Process 100 words at once instead of one by one
results = ensemble.predict_batch(words, batch_size=16)
```

**Solution 2**: Disable validation
```python
# Skip validation for speed (not recommended for production)
result = ensemble.predict(word, pos_tag, use_validation=False)
```

### Issue: Low Accuracy

**Solution 1**: Check confidence scores
```python
if result.confidence < 0.7:
    # Retrain or use GPT-4o-mini as fallback
    result = gpt4o_fallback(word, pos_tag)
```

**Solution 2**: Retrain with more data
```bash
# Add more training examples and retrain
python train_structure.py ...
```

### Issue: JSON Errors

**Solution**: Always use validation
```python
# Validation fixes most JSON errors
result = ensemble.predict(word, pos_tag, use_validation=True)

# Check if valid
is_valid, errors = ensemble.validator.validate(result.to_dict(), pos_tag)
```

## Performance Tips

1. **Batch Processing**: Always use batch mode for multiple words
   ```python
   # Good
   results = ensemble.predict_batch(words, batch_size=16)
   
   # Bad
   results = [ensemble.predict(w, p) for w, p in words]
   ```

2. **GPU Utilization**: Use appropriate batch size
   - 24GB GPU: batch_size=8-12
   - 40GB GPU: batch_size=16-24
   - 80GB GPU: batch_size=32-48

3. **Caching**: Cache results for repeated queries
   ```python
   cache = {}
   
   def predict_cached(word, pos_tag):
       key = f"{word}_{pos_tag}"
       if key not in cache:
           cache[key] = ensemble.predict(word, pos_tag)
       return cache[key]
   ```

4. **Confidence Thresholds**: Use different thresholds per POS
   ```python
   thresholds = {
       "Етістік": 0.6,  # Lower threshold for harder POS
       "Шылау": 0.8,    # Higher for easier POS
   }
   
   threshold = thresholds.get(pos_tag, 0.7)
   if result.confidence < threshold:
       # Handle low confidence
       pass
   ```

## Next Steps

1. **Read the full documentation**: See [README.md](README.md)
2. **Check performance analysis**: See [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)
3. **Run examples**: `python examples.py 1`
4. **Deploy to production**: See [deployment.py](deployment.py)

## Support

- **Issues**: Open an issue on GitHub
- **Questions**: Check documentation or ask in discussions
- **Custom training**: Contact for consulting

## Quick Reference

```python
# Import
from ensemble_model import create_ensemble

# Setup
ensemble = create_ensemble("grammar.json", "patterns.json")
ensemble.load_models()

# Single prediction
result = ensemble.predict("кітап", "Зат есім")

# Batch prediction
results = ensemble.predict_batch([(w, p), ...])

# API server
python deployment.py --port 8000

# Training
bash train_all.sh

# Evaluation
python evaluate.py --test_data test.json
```

That's it! You're ready to use the Kazakh Morphology Ensemble.