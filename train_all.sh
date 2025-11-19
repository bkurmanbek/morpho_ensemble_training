#!/bin/bash
# Complete Training and Deployment Pipeline
# ==========================================

set -e  # Exit on error

echo "================================================================================"
echo "Kazakh Morphology Ensemble - Complete Training Pipeline"
echo "================================================================================"

# Configuration
DATA_PATH="all_structured_kazakh_data.json"
GRAMMAR_PATH="all_kazakh_grammar_data.json"
PATTERN_DB_PATH="pattern_database.json"
MODEL_DIR="./models"

# Check if data files exist
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$GRAMMAR_PATH" ]; then
    echo "Error: Grammar data not found: $GRAMMAR_PATH"
    exit 1
fi

# Create model directory
mkdir -p $MODEL_DIR

echo ""
echo "Step 1: Build Pattern Database"
echo "--------------------------------------------------------------------------------"
python train_sozjasam.py \
    --mode build_db \
    --data_path $DATA_PATH \
    --pattern_db_path $PATTERN_DB_PATH

echo ""
echo "Step 2: Train Structure Model (Qwen 2.5 7B)"
echo "--------------------------------------------------------------------------------"
echo "This will take approximately 6-8 hours on A100 40GB..."
python train_structure.py \
    --data_path $DATA_PATH \
    --grammar_path $GRAMMAR_PATH \
    --output_dir $MODEL_DIR/structure_model \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --use_qlora

echo ""
echo "Step 3: Train Lexical Model (Qwen 2.5 14B)"
echo "--------------------------------------------------------------------------------"
echo "This will take approximately 8-10 hours on A100 40GB..."
python train_lexical.py \
    --data_path $DATA_PATH \
    --output_dir $MODEL_DIR/lexical_model \
    --model_name Qwen/Qwen2.5-14B-Instruct \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --use_8bit

echo ""
echo "Step 4: Train Sozjasam Model (Phi-3 Mini)"
echo "--------------------------------------------------------------------------------"
echo "This will take approximately 3-4 hours on RTX 4090..."
python train_sozjasam.py \
    --mode train \
    --data_path $DATA_PATH \
    --pattern_db_path $PATTERN_DB_PATH \
    --output_dir $MODEL_DIR/sozjasam_model \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --num_epochs 5 \
    --batch_size 4 \
    --learning_rate 2e-4

echo ""
echo "================================================================================"
echo "Training Complete!"
echo "================================================================================"
echo ""
echo "Models saved to:"
echo "  - Structure Model: $MODEL_DIR/structure_model"
echo "  - Lexical Model: $MODEL_DIR/lexical_model"
echo "  - Sozjasam Model: $MODEL_DIR/sozjasam_model"
echo "  - Pattern Database: $PATTERN_DB_PATH"
echo ""
echo "Next steps:"
echo "  1. Evaluate the ensemble:"
echo "     python evaluate.py --test_data test_data.json \\"
echo "                        --grammar_data $GRAMMAR_PATH \\"
echo "                        --pattern_db $PATTERN_DB_PATH"
echo ""
echo "  2. Run examples:"
echo "     python examples.py 1  # Single prediction"
echo "     python examples.py 2  # Batch prediction"
echo ""
echo "  3. Deploy to production (see deployment.py)"
echo ""