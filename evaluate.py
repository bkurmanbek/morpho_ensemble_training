"""
Evaluation Script for Morphology Ensemble
==========================================

Evaluates the ensemble system and compares with baseline GPT-4o-mini.
"""

import json
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from ensemble_model import create_ensemble, MorphologyOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MorphologyEvaluator:
    """Evaluates morphology predictions"""
    
    def __init__(self):
        self.results = defaultdict(dict)
    
    def evaluate(self, 
                 predictions: List[MorphologyOutput],
                 ground_truth: List[Dict]) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Returns:
            Dictionary with evaluation metrics by POS tag
        """
        
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truth)} ground truth")
        
        # Group by POS tag
        by_pos = defaultdict(lambda: {'preds': [], 'truth': []})
        
        for pred, truth in zip(predictions, ground_truth):
            pos_tag = truth['POS tag']
            by_pos[pos_tag]['preds'].append(pred.to_dict())
            by_pos[pos_tag]['truth'].append(truth)
        
        # Evaluate each POS tag
        results = {}
        for pos_tag, data in by_pos.items():
            logger.info(f"Evaluating {pos_tag}...")
            results[pos_tag] = self._evaluate_pos_tag(
                data['preds'], 
                data['truth'],
                pos_tag
            )
        
        # Overall results
        results['overall'] = self._aggregate_results(results)
        
        return results
    
    def _evaluate_pos_tag(self, 
                          predictions: List[Dict],
                          ground_truth: List[Dict],
                          pos_tag: str) -> Dict:
        """Evaluate predictions for a single POS tag"""
        
        n = len(predictions)
        exact_match = 0
        field_hits = defaultdict(int)
        field_total = defaultdict(int)
        
        for pred, truth in zip(predictions, ground_truth):
            # Exact match
            if self._dicts_equal(pred, truth):
                exact_match += 1
            
            # Field-level accuracy
            pred_flat = self._flatten(pred)
            truth_flat = self._flatten(truth)
            
            all_keys = set(pred_flat) | set(truth_flat)
            for key in all_keys:
                field_total[key] += 1
                if pred_flat.get(key) == truth_flat.get(key):
                    field_hits[key] += 1
        
        # Calculate accuracies
        field_acc = {k: field_hits[k] / field_total[k] for k in field_hits}
        
        # Overall field accuracy
        overall_field_acc = sum(field_acc.values()) / len(field_acc) if field_acc else 0.0
        
        # Accuracy excluding lexics and sozjasam
        filtered_acc = {
            k: v for k, v in field_acc.items()
            if not (k.startswith('lexics.') or k.startswith('sozjasam.'))
        }
        filtered_field_acc = sum(filtered_acc.values()) / len(filtered_acc) if filtered_acc else 0.0
        
        return {
            'count': n,
            'exact_match': exact_match / n,
            'field_accuracy': overall_field_acc,
            'field_accuracy_no_lexics_sozjasam': filtered_field_acc,
            'field_accuracies': field_acc
        }
    
    def _aggregate_results(self, results: Dict) -> Dict:
        """Aggregate results across all POS tags"""
        
        total_count = 0
        total_exact = 0
        all_field_accs = []
        all_filtered_accs = []
        
        for pos_tag, metrics in results.items():
            if pos_tag == 'overall':
                continue
            
            count = metrics['count']
            total_count += count
            total_exact += metrics['exact_match'] * count
            all_field_accs.append(metrics['field_accuracy'])
            all_filtered_accs.append(metrics['field_accuracy_no_lexics_sozjasam'])
        
        return {
            'count': total_count,
            'exact_match': total_exact / total_count if total_count > 0 else 0.0,
            'field_accuracy': np.mean(all_field_accs) if all_field_accs else 0.0,
            'field_accuracy_no_lexics_sozjasam': np.mean(all_filtered_accs) if all_filtered_accs else 0.0
        }
    
    def _dicts_equal(self, d1: Dict, d2: Dict) -> bool:
        """Check if two dictionaries are equal"""
        return self._flatten(d1) == self._flatten(d2)
    
    def _flatten(self, d: Dict, prefix: str = "") -> Dict:
        """Flatten nested dictionary"""
        flat = {}
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten(v, new_key))
            else:
                flat[new_key] = v
        return flat
    
    def print_results(self, results: Dict):
        """Print evaluation results"""
        
        print("\n" + "="*80)
        print("ENSEMBLE EVALUATION RESULTS")
        print("="*80)
        
        # Overall results
        overall = results.get('overall', {})
        print(f"\nOVERALL RESULTS ({overall.get('count', 0)} examples):")
        print(f"  Exact Match Accuracy: {overall.get('exact_match', 0):.2%}")
        print(f"  Field-Level Accuracy: {overall.get('field_accuracy', 0):.2%}")
        print(f"  Field-Level Accuracy (excl. lexics, sozjasam): {overall.get('field_accuracy_no_lexics_sozjasam', 0):.2%}")
        
        # Per-POS results
        print("\n" + "-"*80)
        print("RESULTS BY POS TAG:")
        print("-"*80)
        
        for pos_tag in sorted(results.keys()):
            if pos_tag == 'overall':
                continue
            
            metrics = results[pos_tag]
            count = metrics['count']
            
            print(f"\n{pos_tag} ({count} examples):")
            print(f"  Exact Match: {metrics['exact_match']:.2%}")
            print(f"  Field-Level: {metrics['field_accuracy']:.2%}")
            print(f"  Field-Level (excl.): {metrics['field_accuracy_no_lexics_sozjasam']:.2%}")
            
            # Show worst performing fields
            field_accs = metrics['field_accuracies']
            worst_fields = sorted(field_accs.items(), key=lambda x: x[1])[:5]
            
            print(f"  Worst Fields:")
            for field, acc in worst_fields:
                marker = " (excluded)" if (field.startswith('lexics.') or field.startswith('sozjasam.')) else ""
                print(f"    • {field}: {acc:.2%}{marker}")
    
    def compare_with_baseline(self, 
                              ensemble_results: Dict,
                              baseline_results: Dict):
        """Compare ensemble results with baseline"""
        
        print("\n" + "="*80)
        print("COMPARISON: ENSEMBLE vs BASELINE (GPT-4o-mini)")
        print("="*80)
        
        # Overall comparison
        ens_overall = ensemble_results['overall']
        base_overall = baseline_results['overall']
        
        print(f"\nOVERALL:")
        print(f"  Metric                    Baseline    Ensemble    Improvement")
        print(f"  {'─'*70}")
        
        metrics = [
            ('Exact Match', 'exact_match'),
            ('Field-Level', 'field_accuracy'),
            ('Field-Level (excl.)', 'field_accuracy_no_lexics_sozjasam')
        ]
        
        for name, key in metrics:
            base_val = base_overall.get(key, 0)
            ens_val = ens_overall.get(key, 0)
            improvement = ens_val - base_val
            
            print(f"  {name:25} {base_val:>7.2%}    {ens_val:>7.2%}    {improvement:>+7.2%}")
        
        # Per-POS comparison
        print(f"\n{'─'*80}")
        print("PER-POS TAG EXACT MATCH COMPARISON:")
        print(f"{'─'*80}")
        print(f"  {'POS Tag':<20} {'Baseline':>10} {'Ensemble':>10} {'Improvement':>12}")
        print(f"  {'─'*70}")
        
        for pos_tag in sorted(ensemble_results.keys()):
            if pos_tag == 'overall':
                continue
            
            base_em = baseline_results.get(pos_tag, {}).get('exact_match', 0)
            ens_em = ensemble_results[pos_tag]['exact_match']
            improvement = ens_em - base_em
            
            marker = "✓" if improvement > 0 else "✗" if improvement < 0 else "="
            print(f"  {pos_tag:<20} {base_em:>9.2%} {ens_em:>9.2%} {improvement:>+10.2%} {marker}")


def load_baseline_results(results_path: str) -> Dict:
    """Load baseline GPT-4o-mini results"""
    
    # This would load your GPT-4o-mini evaluation results
    # For now, we'll use the numbers from your document
    
    baseline = {
        'Одағай': {
            'count': 49,
            'exact_match': 0.0,
            'field_accuracy': 0.9435,
            'field_accuracy_no_lexics_sozjasam': 0.9524
        },
        'Есімдік': {
            'count': 124,
            'exact_match': 0.0565,
            'field_accuracy': 0.8418,
            'field_accuracy_no_lexics_sozjasam': 0.9052
        },
        'Еліктеуіш': {
            'count': 124,
            'exact_match': 0.0,
            'field_accuracy': 0.8454,
            'field_accuracy_no_lexics_sozjasam': 0.8746
        },
        'Етістік': {
            'count': 130,
            'exact_match': 0.0,
            'field_accuracy': 0.6846,
            'field_accuracy_no_lexics_sozjasam': 0.7505
        },
        'Сан есім': {
            'count': 124,
            'exact_match': 0.0,
            'field_accuracy': 0.8730,
            'field_accuracy_no_lexics_sozjasam': 0.9066
        },
        'Сын есім': {
            'count': 124,
            'exact_match': 0.0,
            'field_accuracy': 0.8836,
            'field_accuracy_no_lexics_sozjasam': 0.9172
        },
        'Зат есім': {
            'count': 126,
            'exact_match': 0.0,
            'field_accuracy': 0.8343,
            'field_accuracy_no_lexics_sozjasam': 0.8552
        },
        'Үстеу': {
            'count': 124,
            'exact_match': 0.0806,
            'field_accuracy': 0.8765,
            'field_accuracy_no_lexics_sozjasam': 0.9231
        },
        'Шылау': {
            'count': 79,
            'exact_match': 0.4177,
            'field_accuracy': 0.9024,
            'field_accuracy_no_lexics_sozjasam': 0.9293
        }
    }
    
    # Calculate overall
    total_count = sum(v['count'] for v in baseline.values())
    total_exact = sum(v['exact_match'] * v['count'] for v in baseline.values())
    avg_field = np.mean([v['field_accuracy'] for v in baseline.values()])
    avg_filtered = np.mean([v['field_accuracy_no_lexics_sozjasam'] for v in baseline.values()])
    
    baseline['overall'] = {
        'count': total_count,
        'exact_match': total_exact / total_count,
        'field_accuracy': avg_field,
        'field_accuracy_no_lexics_sozjasam': avg_filtered
    }
    
    return baseline


def evaluate_ensemble(
    ensemble_model_paths: Dict[str, str],
    test_data_path: str,
    grammar_data_path: str,
    pattern_db_path: str,
    baseline_results_path: str = None
):
    """Main evaluation function"""
    
    logger.info("Loading test data...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"Test set size: {len(test_data)}")
    
    # Create ensemble
    logger.info("Creating ensemble...")
    ensemble = create_ensemble(grammar_data_path, pattern_db_path)
    
    # Load models
    logger.info("Loading ensemble models...")
    ensemble.load_models()
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = []
    
    for i, item in enumerate(test_data):
        if i % 10 == 0:
            logger.info(f"Processing {i}/{len(test_data)}...")
        
        word = item['word']
        pos_tag = item['POS tag']
        
        pred = ensemble.predict(word, pos_tag)
        predictions.append(pred)
    
    # Evaluate
    logger.info("Evaluating predictions...")
    evaluator = MorphologyEvaluator()
    results = evaluator.evaluate(predictions, test_data)
    
    # Print results
    evaluator.print_results(results)
    
    # Compare with baseline
    if baseline_results_path:
        logger.info("Loading baseline results...")
        baseline_results = load_baseline_results(baseline_results_path)
        evaluator.compare_with_baseline(results, baseline_results)
    
    # Save results
    output_path = "ensemble_evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        # Convert results to serializable format
        serializable_results = {}
        for pos_tag, metrics in results.items():
            serializable_results[pos_tag] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in metrics.items()
                if k != 'field_accuracies'  # Skip detailed field accuracies
            }
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True, help="Path to test data JSON")
    parser.add_argument("--grammar_data", required=True, help="Path to grammar data JSON")
    parser.add_argument("--pattern_db", required=True, help="Path to pattern database JSON")
    parser.add_argument("--baseline_results", help="Path to baseline results (optional)")
    parser.add_argument("--structure_model", default="./models/structure_model")
    parser.add_argument("--lexical_model", default="./models/lexical_model")
    parser.add_argument("--sozjasam_model", default="./models/sozjasam_model")
    
    args = parser.parse_args()
    
    model_paths = {
        'structure': args.structure_model,
        'lexical': args.lexical_model,
        'sozjasam': args.sozjasam_model
    }
    
    evaluate_ensemble(
        ensemble_model_paths=model_paths,
        test_data_path=args.test_data,
        grammar_data_path=args.grammar_data,
        pattern_db_path=args.pattern_db,
        baseline_results_path=args.baseline_results
    )