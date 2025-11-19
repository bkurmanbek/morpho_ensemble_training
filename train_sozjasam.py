"""
Training Script for Sozjasam Model + Pattern Database Builder
==============================================================

1. Builds a pattern database from training data for RAG
2. Fine-tunes Phi-3 Mini for word formation pattern prediction
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import logging
import argparse
from typing import List, Dict
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternDatabase:
    """Build and query pattern database for RAG"""
    
    def __init__(self):
        self.patterns = defaultdict(list)
    
    def build_from_data(self, data: List[Dict]) -> None:
        """Build pattern database from training data"""
        
        for item in data:
            pos_tag = item.get('POS tag')
            word = item.get('word')
            pattern = item.get('sozjasam', {}).get('тәсілін, құрамын шартты қысқартумен беру', 'NaN')
            
            if pattern and pattern != 'NaN':
                entry = {
                    'word': word,
                    'POS tag': pos_tag,
                    'pattern': pattern,
                    'morphology': item.get('morphology', {}),
                    'lemma': item.get('lemma', word)
                }
                
                self.patterns[pos_tag].append(entry)
        
        logger.info(f"Built pattern database with {sum(len(v) for v in self.patterns.values())} entries")
        for pos, entries in self.patterns.items():
            logger.info(f"  {pos}: {len(entries)} patterns")
    
    def save(self, path: str) -> None:
        """Save pattern database to JSON"""
        
        # Flatten the database
        all_patterns = []
        for pos_tag, entries in self.patterns.items():
            all_patterns.extend(entries)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(all_patterns, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved pattern database to {path}")
    
    def load(self, path: str) -> None:
        """Load pattern database from JSON"""
        
        with open(path, 'r', encoding='utf-8') as f:
            all_patterns = json.load(f)
        
        self.patterns = defaultdict(list)
        for entry in all_patterns:
            pos_tag = entry.get('POS tag')
            self.patterns[pos_tag].append(entry)
        
        logger.info(f"Loaded pattern database with {len(all_patterns)} entries")
    
    def query(self, pos_tag: str, morphology: Dict = None, k: int = 5) -> List[Dict]:
        """Query pattern database for similar examples"""

        candidates = self.patterns.get(pos_tag, [])

        if not candidates:
            return []

        # If morphology provided, filter by similar morphology
        if morphology:
            morph_type = self._get_morphology_type(morphology)

            # For large candidate pools, sample first then filter
            if len(candidates) > 1000:
                sample_pool = random.sample(candidates, min(1000, len(candidates)))
                filtered = [
                    c for c in sample_pool
                    if self._get_morphology_type(c.get('morphology', {})) == morph_type
                ]
            else:
                filtered = [
                    c for c in candidates
                    if self._get_morphology_type(c.get('morphology', {})) == morph_type
                ]

            if filtered:
                candidates = filtered

        # Return random k samples if pool is large, else top k
        if len(candidates) > k:
            return random.sample(candidates, k)
        return candidates[:k]
    
    def _get_morphology_type(self, morphology: Dict) -> str:
        """Determine morphology type"""
        
        if morphology.get('дара, негізгі') != 'NaN':
            return 'дара_негізгі'
        elif morphology.get('дара, туынды') != 'NaN':
            return 'дара_туынды'
        elif morphology.get('күрделі, біріккен, Бірік.') != 'NaN':
            return 'күрделі_біріккен'
        elif morphology.get('күрделі, қосарланған, Қос.') != 'NaN':
            return 'күрделі_қосарланған'
        else:
            return 'unknown'


class SozjasamDatasetBuilder:
    """Build training dataset for sozjasam model"""
    
    def __init__(self, data_path: str, pattern_db: PatternDatabase):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.pattern_db = pattern_db
    
    def build_dataset(self, sample_size: int = None) -> Dataset:
        """Build HuggingFace dataset with RAG examples"""

        from tqdm import tqdm

        # Optionally sample data for faster testing
        data_to_process = self.data
        if sample_size and sample_size < len(self.data):
            logger.info(f"Sampling {sample_size} items from {len(self.data)} for testing")
            data_to_process = random.sample(self.data, sample_size)

        examples = []
        for item in tqdm(data_to_process, desc="Building dataset"):
            # Only include items with valid patterns
            pattern = item.get('sozjasam', {}).get('тәсілін, құрамын шартты қысқартумен беру', 'NaN')
            if pattern and pattern != 'NaN':
                example = self._create_sozjasam_example(item)
                examples.append(example)

        logger.info(f"Created {len(examples)} sozjasam training examples")
        return Dataset.from_list(examples)
    
    def _create_sozjasam_example(self, item: dict) -> dict:
        """Create a single training example with retrieval context"""
        
        pos_tag = item['POS tag']
        word = item['word']
        pattern = item['sozjasam']['тәсілін, құрамын шартты қысқартумен беру']
        morphology = item.get('morphology', {})
        
        # Retrieve similar patterns (excluding the current word)
        similar_patterns = self.pattern_db.query(pos_tag, morphology, k=3)
        similar_patterns = [p for p in similar_patterns if p['word'] != word][:3]
        
        # Build examples string
        examples_str = ""
        if similar_patterns:
            examples_str = "\n\nҰҚСАС МЫСАЛДАР:"
            for i, ex in enumerate(similar_patterns, 1):
                examples_str += f"\n{i}. {ex['word']}: {ex['pattern']}"
        
        # Morphology info
        morph_info = ""
        if morphology:
            morph_type = self.pattern_db._get_morphology_type(morphology)
            morph_info = f"\nМорфология түрі: {morph_type}"
        
        system_prompt = f"""Сіз қазақ тілінің сөзжасамы бойынша сарапшысыз.

МІНДЕТ: Берілген сөздің СӨЗЖАСАМ ТӘСІЛІН қысқартумен беріңіз.

ҚЫСҚАРТУ ФОРМАТТАРЫ:
- Дара, негізгі: зт/Ø, сн/Ø
- Дара, туынды: зт/-шы, ет/-ла
- Күрделі, біріккен: зт+зт, сн+зт
- Күрделі, қосарланған: зт-зт

НҰСҚАУЛАР:
1. ТЕК қысқартылған форматты жазыңыз
2. Ұқсас мысалдарды пайдаланыңыз
3. Басқа түсініктеме жоқ"""

        user_prompt = f"""СӨЗ: {word}
СӨЗ ТАБЫ: {pos_tag}{morph_info}{examples_str}

Қысқарту:"""

        return {
            'system': system_prompt,
            'user': user_prompt,
            'assistant': pattern,
            'pos_tag': pos_tag,
            'word': word
        }


def build_pattern_database(data_path: str, output_path: str):
    """Build and save pattern database"""
    
    logger.info("Building pattern database...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    db = PatternDatabase()
    db.build_from_data(data)
    db.save(output_path)
    
    return db


def train_sozjasam_model(
    data_path: str,
    pattern_db_path: str,
    output_dir: str,
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    sample_size: int = None
):
    """Train the sozjasam model"""

    # Load pattern database
    logger.info(f"Loading pattern database from {pattern_db_path}")
    pattern_db = PatternDatabase()
    pattern_db.load(pattern_db_path)

    # Build dataset
    logger.info("Building dataset...")
    builder = SozjasamDatasetBuilder(data_path, pattern_db)
    dataset = builder.build_dataset(sample_size=sample_size)
    
    # Split train/val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Format dataset
    def format_fn(example):
        messages = [
            {"role": "system", "content": example['system']},
            {"role": "user", "content": example['user']},
            {"role": "assistant", "content": example['assistant']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    
    train_dataset = train_dataset.map(format_fn)
    val_dataset = val_dataset.map(format_fn)
    
    # Detect device
    if torch.cuda.is_available():
        device_type = "cuda"
        logger.info("Using CUDA")
    elif torch.backends.mps.is_available():
        device_type = "mps"
        logger.info("Using Apple MPS")
    else:
        device_type = "cpu"
        logger.info("Using CPU")

    # Load model
    logger.info(f"Loading model: {model_name}")

    if device_type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # For CPU/MPS, load without device_map to avoid offloading issues
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable training mode and ensure gradients are enabled
    model.train()
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=(device_type == "cuda"),
        bf16=False,
        gradient_checkpointing=False,
        report_to="none",
        use_mps_device=(device_type == "mps")
    )
    
    # Tokenize
    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['build_db', 'train'], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--pattern_db_path", required=True)
    parser.add_argument("--output_dir", help="Output directory for model (train mode only)")
    parser.add_argument("--model_name", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for testing (use smaller value for quick tests)")

    args = parser.parse_args()
    
    if args.mode == 'build_db':
        build_pattern_database(args.data_path, args.pattern_db_path)
    else:
        if not args.output_dir:
            raise ValueError("--output_dir required for train mode")
        
        train_sozjasam_model(
            data_path=args.data_path,
            pattern_db_path=args.pattern_db_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            sample_size=args.sample_size
        )