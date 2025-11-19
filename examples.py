"""
Example Usage of Kazakh Morphology Ensemble
============================================

This script demonstrates various ways to use the ensemble system.
"""

import json
from ensemble_model import create_ensemble, MorphologyOutput
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_single_prediction():
    """Example 1: Single word prediction"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Word Prediction")
    print("="*80)
    
    # Create ensemble
    ensemble = create_ensemble(
        grammar_data_path="all_kazakh_grammar_data.json",
        pattern_db_path="pattern_database.json"
    )
    
    # Load models (this takes a few minutes)
    print("Loading models...")
    ensemble.load_models()
    
    # Predict
    print("\nPredicting morphology for 'кітап' (Зат есім)...")
    result = ensemble.predict(word="кітап", pos_tag="Зат есім")
    
    # Print results
    print(f"\nWord: {result.word}")
    print(f"POS Tag: {result.pos_tag}")
    print(f"Lemma: {result.lemma}")
    print(f"Confidence: {result.confidence:.2f}")
    print("\nFull Output:")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def example_2_batch_prediction():
    """Example 2: Batch prediction"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Prediction")
    print("="*80)
    
    # Create ensemble
    ensemble = create_ensemble(
        grammar_data_path="all_kazakh_grammar_data.json",
        pattern_db_path="pattern_database.json"
    )
    ensemble.load_models()
    
    # Words to analyze
    words = [
        ("кітап", "Зат есім"),
        ("жылдам", "Сын есім"),
        ("оқу", "Етістік"),
        ("мектеп", "Зат есім"),
        ("әдемі", "Сын есім")
    ]
    
    print(f"\nAnalyzing {len(words)} words...")
    results = ensemble.predict_batch(words, batch_size=8)
    
    # Print summary
    print("\nResults Summary:")
    print(f"{'Word':<15} {'POS Tag':<15} {'Lemma':<15} {'Confidence':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result.word:<15} {result.pos_tag:<15} {result.lemma:<15} {result.confidence:<12.2f}")


def example_3_with_validation():
    """Example 3: Using validation"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: With Validation")
    print("="*80)
    
    ensemble = create_ensemble(
        grammar_data_path="all_kazakh_grammar_data.json",
        pattern_db_path="pattern_database.json"
    )
    ensemble.load_models()
    
    word = "кітап"
    pos_tag = "Зат есім"
    
    # Predict with validation
    print(f"\nPredicting with validation: {word} ({pos_tag})")
    result = ensemble.predict(word, pos_tag, use_validation=True)
    
    # Check validation
    is_valid, errors = ensemble.validator.validate(result.to_dict(), pos_tag)
    
    print(f"Valid: {is_valid}")
    if not is_valid:
        print(f"Validation Errors: {errors}")
    else:
        print("No validation errors!")
    
    print(f"Confidence: {result.confidence:.2f}")


def example_4_compare_with_baseline():
    """Example 4: Compare with baseline results"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Compare with Baseline")
    print("="*80)
    
    # Simulate baseline (GPT-4o-mini) result
    baseline_result = {
        "POS tag": "Зат есім",
        "word": "кітап",
        "lemma": "кітап",
        "morphology": {
            "column": "Лемма",
            "дара, негізгі": "негізгі",
            "дара, туынды": "NaN",
            "күрделі, біріккен, Бірік.": "NaN",
            "күрделі, қосарланған, Қос.": "NaN",
            "күрделі, қысқарған, Қыс.": "NaN",
            "күрделі, тіркескен, Тірк.": "NaN"
        },
        "semantics": {
            "жалпы": "жалпы",
            "жалқы": "NaN",
            "адамзат, Адз.": "NaN",
            "ғаламзат, Ғалз.": "Ғалз",
            "деректі, Дер.": "Дер",
            "дерексіз, Дерз.": "NaN"
        },
        "lexics": {
            "мағынасы": "кітап -зт. Басылып шыққан еңбек."
        },
        "sozjasam": {
            "тәсілін, құрамын шартты қысқартумен беру": "зт/Ø"
        }
    }
    
    # Get ensemble prediction
    ensemble = create_ensemble(
        grammar_data_path="all_kazakh_grammar_data.json",
        pattern_db_path="pattern_database.json"
    )
    ensemble.load_models()
    
    ensemble_result = ensemble.predict("кітап", "Зат есім")
    
    # Compare
    print("\nBaseline (GPT-4o-mini):")
    print(json.dumps(baseline_result, ensure_ascii=False, indent=2))
    
    print("\nEnsemble:")
    print(json.dumps(ensemble_result.to_dict(), ensure_ascii=False, indent=2))
    
    # Field-by-field comparison
    def flatten(d, prefix=""):
        flat = {}
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten(v, new_key))
            else:
                flat[new_key] = v
        return flat
    
    baseline_flat = flatten(baseline_result)
    ensemble_flat = flatten(ensemble_result.to_dict())
    
    print("\nField-by-Field Comparison:")
    print(f"{'Field':<50} {'Match':<10}")
    print("-" * 60)
    
    matches = 0
    total = 0
    
    for key in sorted(baseline_flat.keys()):
        baseline_val = baseline_flat[key]
        ensemble_val = ensemble_flat.get(key, "MISSING")
        match = "✓" if baseline_val == ensemble_val else "✗"
        
        if match == "✓":
            matches += 1
        total += 1
        
        print(f"{key:<50} {match:<10}")
    
    print(f"\nField Accuracy: {matches}/{total} = {matches/total:.2%}")


def example_5_process_dataset():
    """Example 5: Process entire dataset"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Process Dataset")
    print("="*80)
    
    # Load dataset
    with open("test_data.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Dataset size: {len(test_data)}")
    
    # Create ensemble
    ensemble = create_ensemble(
        grammar_data_path="all_kazakh_grammar_data.json",
        pattern_db_path="pattern_database.json"
    )
    ensemble.load_models()
    
    # Process in batches
    batch_size = 16
    all_results = []
    
    print(f"\nProcessing in batches of {batch_size}...")
    
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        words = [(item['word'], item['POS tag']) for item in batch]
        
        results = ensemble.predict_batch(words, batch_size=8)
        all_results.extend(results)
        
        print(f"Processed {min(i+batch_size, len(test_data))}/{len(test_data)}")
    
    # Save results
    output_data = [result.to_dict() for result in all_results]
    
    with open("ensemble_predictions.json", 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved predictions to ensemble_predictions.json")
    
    # Calculate statistics
    avg_confidence = sum(r.confidence for r in all_results) / len(all_results)
    
    print(f"\nStatistics:")
    print(f"  Total predictions: {len(all_results)}")
    print(f"  Average confidence: {avg_confidence:.2f}")
    
    # Confidence distribution
    high_conf = sum(1 for r in all_results if r.confidence >= 0.8)
    medium_conf = sum(1 for r in all_results if 0.5 <= r.confidence < 0.8)
    low_conf = sum(1 for r in all_results if r.confidence < 0.5)
    
    print(f"  High confidence (≥0.8): {high_conf} ({high_conf/len(all_results):.1%})")
    print(f"  Medium confidence (0.5-0.8): {medium_conf} ({medium_conf/len(all_results):.1%})")
    print(f"  Low confidence (<0.5): {low_conf} ({low_conf/len(all_results):.1%})")


def example_6_individual_models():
    """Example 6: Using individual models"""
    
    print("\n" + "="*80)
    print("EXAMPLE 6: Using Individual Models")
    print("="*80)
    
    from ensemble_model import StructureModel, LexicalModel, SozjasamModel
    
    word = "кітап"
    pos_tag = "Зат есім"
    
    # Use only structure model
    print("\n1. Structure Model Only:")
    structure_model = StructureModel("./models/structure_model")
    structure_model.load_model()
    structure_output = structure_model.predict(word, pos_tag)
    print(json.dumps(structure_output, ensure_ascii=False, indent=2))
    
    # Use only lexical model
    print("\n2. Lexical Model Only:")
    lexical_model = LexicalModel("./models/lexical_model")
    lexical_model.load_model()
    lexical_output = lexical_model.predict(word, pos_tag, context=structure_output)
    print(json.dumps(lexical_output, ensure_ascii=False, indent=2))
    
    # Use only sozjasam model
    print("\n3. Sozjasam Model Only:")
    sozjasam_model = SozjasamModel("./models/sozjasam_model", "pattern_database.json")
    sozjasam_model.load_model()
    sozjasam_output = sozjasam_model.predict(word, pos_tag, context=structure_output)
    print(json.dumps(sozjasam_output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Single prediction', example_1_single_prediction),
        '2': ('Batch prediction', example_2_batch_prediction),
        '3': ('With validation', example_3_with_validation),
        '4': ('Compare with baseline', example_4_compare_with_baseline),
        '5': ('Process dataset', example_5_process_dataset),
        '6': ('Individual models', example_6_individual_models)
    }
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"\nRunning: {name}")
            func()
        else:
            print(f"Invalid example number: {example_num}")
            print("Available examples:")
            for num, (name, _) in examples.items():
                print(f"  {num}. {name}")
    else:
        print("Usage: python examples.py <example_number>")
        print("\nAvailable examples:")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")