"""
Training Script for Lexical Model
==================================

Fine-tunes Qwen 2.5 14B for word meaning generation.
Focuses specifically on lexics.мағынасы field.
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LexicalDatasetBuilder:
    """Build training dataset for lexical model"""
    
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def build_dataset(self) -> Dataset:
        """Build HuggingFace dataset"""
        
        examples = []
        for item in self.data:
            # Only include items with valid meanings
            meaning = item.get('lexics', {}).get('мағынасы', 'NaN')
            if meaning and meaning != 'NaN' and len(meaning) > 10:
                example = self._create_lexical_example(item)
                examples.append(example)
        
        logger.info(f"Created {len(examples)} lexical training examples")
        return Dataset.from_list(examples)
    
    def _create_lexical_example(self, item: dict) -> dict:
        """Create a single training example for lexical meaning"""
        
        pos_tag = item['POS tag']
        word = item['word']
        meaning = item['lexics']['мағынасы']
        
        # Add morphological context if available
        morph_info = ""
        if 'morphology' in item:
            column = item['morphology'].get('column', 'NaN')
            if column != 'NaN':
                morph_info = f"\nМорфологиялық форма: {column}"
        
        # Add semantic context
        sem_info = ""
        if 'semantics' in item:
            active_sem = [k for k, v in item['semantics'].items() if v != 'NaN']
            if active_sem:
                sem_info = f"\nСемантикалық категория: {', '.join(active_sem[:2])}"
        
        system_prompt = f"""Сіз қазақ тілінің лексикологиясы бойынша сарапшысыз.

МІНДЕТ: Берілген сөздің ТОЛЫҚ ЛЕКСИКАЛЫҚ МАҒЫНАСЫН беріңіз.

НҰСҚАУЛАР:
1. Сөздің негізгі мағынасын толық және нақты түсіндіріңіз
2. Сөз табын атап өтіңіз
3. Қазақ тілінде жазыңыз
4. Мысалдар қосуға болады
5. ТЕК мағынаны жазыңыз

ФОРМАТ: сөз -сөз_табы. мағынасы."""

        user_prompt = f"""СӨЗ: {word}
СӨЗ ТАБЫ: {pos_tag}{morph_info}{sem_info}

Мағынасы:"""

        return {
            'system': system_prompt,
            'user': user_prompt,
            'assistant': meaning,
            'pos_tag': pos_tag,
            'word': word
        }


def train_lexical_model(
    data_path: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    use_8bit: bool = True
):
    """Train the lexical model"""
    
    logger.info("Building dataset...")
    builder = LexicalDatasetBuilder(data_path)
    dataset = builder.build_dataset()
    
    # Split train/val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Format dataset
    def format_fn(example):
        messages = [
            {"role": "system", "content": example['system']},
            {"role": "user", "content": example['user']},
            {"role": "assistant", "content": example['assistant']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}
    
    train_dataset = train_dataset.map(format_fn)
    val_dataset = val_dataset.map(format_fn)
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    
    if use_8bit:
        # 8-bit quantization for 14B model
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,  # Reduced for memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Compensate with more accumulation
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,  # Use fp16 instead of bf16 for better compatibility
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if use_8bit else "adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,  # Reduce memory usage
        max_grad_norm=0.3
    )
    
    # Tokenize
    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # Reduced for memory
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
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_8bit", action="store_true")
    
    args = parser.parse_args()
    
    train_lexical_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_8bit=args.use_8bit
    )