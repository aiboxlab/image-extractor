import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import click
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Tuple
import math

def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load CSV file with comma as decimal separator."""
    df = pd.read_csv(csv_path, decimal=',')
    return df

def get_metrics_columns() -> List[str]:
    """
    Returns the list of metric columns to evaluate.
    From cncadc to synstrutt and from verb_agreement_score to wrdverb.
    """
    # First group: cncadc to synstrutt
    first_group = [
        'adapted_dalechall', 'additive_neg_ratio', 'additive_pos_ratio', 'adjective_ratio', 'adjectives_max', 'adjectives_min', 'adjectives_std', 'adverbs', 'adverbs_diversity_ratio', 'adverbs_max', 'adverbs_min', 'adverbs_std', 'all_conn_ratio', 'and_ratio', 'brunet_indice', 'cause_neg_ratio', 'cause_pos_ratio', 'conjugation_first_ratio', 'conjugation_irregular_ratio', 'conjugation_score', 'conjugation_second_ratio', 'conjugation_third_ratio', 'content_words', 'cross_entropy', 'first_person_pronouns', 'flesch_indice', 'function_words', 'guiraud_index', 'gunning_fox_indice', 'hapax_legomena', 'honore_statistics', 'if_ratio', 'indefinite_pronoun_ratio', 'indicative_condition_ratio', 'indicative_future_ratio', 'infinitive_verbs', 'inflected_verbs', 'lexical_density', 'lexical_diversity', 'lexical_diversity_mtld', 'log_neg_ratio', 'log_pos_ratio', 'logic_operators_ratio', 'lsa_adj_mean', 'lsa_adj_std', 'lsa_all_mean', 'lsa_all_std', 'lsa_givenness_mean', 'lsa_givenness_std', 'lsa_paragraph_mean', 'lsa_paragraph_std', 'lsa_span_mean', 'lsa_span_std', 'negation_ratio', 'nominal_agreement_score', 'nominal_regency_score', 'non_inflected_verbs', 'nouns_max', 'nouns_min', 'nouns_ratio', 'nouns_std', 'oblique_pronouns_ratio', 'or_ratio', 'personal_pronouns', 'pos_dissimilarity', 'prepositions_per_clause', 'prepositions_per_sentence', 'pronoun_ratio', 'pronouns_max', 'pronouns_min', 'pronouns_std', 'punctuation_ratio', 'ratio_adj', 'ratio_adp', 'ratio_adv', 'ratio_aux', 'ratio_det', 'ratio_function_to_content_words', 'ratio_noun', 'ratio_propn', 'ratio_verb', 'readibility_indice', 'relative_pronouns_ratio', 'second_person_pronouns', 'sent_all_conn', 'sent_conn_adicao', 'sent_conn_alternancia', 'sent_conn_certeza', 'sent_conn_chamar_atencao', 'sent_conn_comparacao', 'sent_conn_comprovacao', 'sent_conn_concessao', 'sent_conn_conclusao', 'sent_conn_condicao', 'sent_conn_conformidade', 'sent_conn_consequencia', 'sent_conn_correcao', 'sent_conn_deducao', 'sent_conn_duvida', 'sent_conn_esclarecimento', 'sent_conn_exclusao', 'sent_conn_exemplificacao', 'sent_conn_explicacao', 'sent_conn_finalidade', 'sent_conn_inclusao', 'sent_conn_justificativa', 'sent_conn_marcacao', 'sent_conn_mediacao', 'sent_conn_minimidade', 'sent_conn_opiniao', 'sent_conn_oposicao', 'sent_conn_prioridade', 'sent_conn_proporcao', 'sent_conn_reafirmacao', 'sent_conn_restricao', 'sent_conn_resumo', 'sent_conn_tempo', 'sentence_length_max', 'sentence_length_min', 'sentence_length_std', 'sentences_per_paragraph', 'stopwords_ratio', 'syllables_per_content_word', 'third_person_pronouns', 'token_var_idx', 'total_paragraphs', 'total_sentences', 'total_stopwords', 'total_words', 'verb_agreement_score', 'verb_regency_score', 'verbs_max', 'verbs_min', 'verbs_ratio', 'verbs_std', 'words_per_sentence', 'yule_k'
    ]

    return first_group

def match_datasets(predictions: pd.DataFrame, ground_truth: pd.DataFrame, metrics_cols: List[str]) -> pd.DataFrame:
    """
    Match predictions with ground truth based on row index.
    Assumes both dataframes are aligned by row.
    """
    matched_data = []
    
    # Use the ID column as identifier
    id_col = 'Id (Local do arquivo da imagem)'
    
    total = min(len(predictions), len(ground_truth))
    
    for idx in range(total):
        pred_row = predictions.iloc[idx]
        gt_row = ground_truth.iloc[idx]
        
        # Create matched record
        record = {
            'id': idx,
            'filename': pred_row[id_col] if id_col in predictions.columns else f'row_{idx}'
        }
        
        # Add prediction and ground truth for each metric
        for metric in metrics_cols:
            if metric in predictions.columns and metric in ground_truth.columns:
                record[f'{metric}_pred'] = pred_row[metric]
                record[f'{metric}_gt'] = gt_row[metric]
            else:
                # Handle missing columns
                if metric not in predictions.columns:
                    print(f"Warning: Column '{metric}' not found in predictions")
                if metric not in ground_truth.columns:
                    print(f"Warning: Column '{metric}' not found in ground truth")
        
        matched_data.append(record)
    
    return pd.DataFrame(matched_data)

def root_mean_squared_error(y_true, y_pred):
    """Calculate RMSE."""
    # Filter out NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return math.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))

def calculate_metrics_for_column(y_true, y_pred, metric_name: str) -> Dict[str, float]:
    """Calculate MAE, RMSE, and exact match percentage for a single metric."""
    # Filter out NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    if mask.sum() == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'exact_match_pct': 0.0,
            'n_valid': 0
        }
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
    rmse = float(root_mean_squared_error(y_true_clean, y_pred_clean))
    mae_normalized = float(mae / y_true_clean.max() if y_true_clean.max() != 0 else 0)
    rmse_normalized = float(rmse / y_true_clean.max() if y_true_clean.max() != 0 else 0)
    mae_std = float(np.std(y_true_clean - y_pred_clean))
    rmse_std = float(np.std(y_true_clean - y_pred_clean))
    
    # Exact match - consider floating point tolerance
    exact_matches = np.sum(np.abs(y_true_clean - y_pred_clean) < 1e-9)
    exact_match_pct = float(exact_matches / len(y_true_clean) * 100)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mae_normalized': mae_normalized,
        'rmse_normalized': rmse_normalized,
        'mae_std': mae_std,
        'rmse_std': rmse_std,
        'exact_match_pct': exact_match_pct,
        'n_valid': int(mask.sum())
    }

def calculate_all_metrics(matched_df: pd.DataFrame, metrics_cols: List[str]) -> Dict[str, Any]:
    """Calculate evaluation metrics for all feature columns."""
    results = {}
    
    # Summary dictionaries
    mae_dict = {}
    rmse_dict = {}
    mae_normalized_dict = {}
    rmse_normalized_dict = {}
    mae_std_dict = {}
    rmse_std_dict = {}
    exact_match_dict = {}
    
    for metric in metrics_cols:
        pred_col = f'{metric}_pred'
        gt_col = f'{metric}_gt'
        
        if pred_col not in matched_df.columns or gt_col not in matched_df.columns:
            print(f"Skipping metric '{metric}' - columns not found")
            continue
        
        y_pred = matched_df[pred_col].values.astype(float)
        y_true = matched_df[gt_col].values.astype(float)
        
        metric_results = calculate_metrics_for_column(y_true, y_pred, metric)
        
        results[metric] = metric_results
        mae_dict[metric] = metric_results['mae']
        rmse_dict[metric] = metric_results['rmse']
        mae_normalized_dict[metric] = metric_results['mae_normalized']
        rmse_normalized_dict[metric] = metric_results['rmse_normalized']
        mae_std_dict[metric] = metric_results['mae_std']
        rmse_std_dict[metric] = metric_results['rmse_std']
        exact_match_dict[metric] = metric_results['exact_match_pct']
    
    # Calculate overall statistics
    valid_maes = [v for v in mae_dict.values() if not np.isnan(v)]
    valid_rmses = [v for v in rmse_dict.values() if not np.isnan(v)]
    valid_exact = [v for v in exact_match_dict.values() if not np.isnan(v)]
    
    summary = {
        'per_metric': results,
        'mae': mae_dict,
        'rmse': rmse_dict,
        'mae_normalized': mae_normalized_dict,
        'rmse_normalized': rmse_normalized_dict,
        'mae_std': mae_std_dict,
        'rmse_std': rmse_std_dict,
        'exact_match_percentage': exact_match_dict,
        'overall': {
            'mean_mae': float(np.mean(valid_maes)) if valid_maes else np.nan,
            'mean_rmse': float(np.mean(valid_rmses)) if valid_rmses else np.nan,
            'mean_exact_match_pct': float(np.mean(valid_exact)) if valid_exact else np.nan,
            'median_mae': float(np.median(valid_maes)) if valid_maes else np.nan,
            'median_rmse': float(np.median(valid_rmses)) if valid_rmses else np.nan,
            'median_exact_match_pct': float(np.median(valid_exact)) if valid_exact else np.nan,
        }
    }
    
    return summary

def generate_error_ranking(metrics: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Generate ranking of metrics by RMSE (worst to best)."""
    rmse_values = metrics['rmse']
    ranking = [(metric, rmse) for metric, rmse in rmse_values.items() if not np.isnan(rmse)]
    ranking.sort(key=lambda x: x[1], reverse=True)  # Sort by RMSE descending (worst first)
    return ranking

def save_results(matched_df: pd.DataFrame, metrics: Dict[str, Any], output_dir: str):
    """Save evaluation results to CSV and JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save matched data
    csv_path = output_path / 'features_evaluation_results.csv'
    matched_df.to_csv(csv_path, index=False)
    
    # Generate error ranking
    error_ranking = generate_error_ranking(metrics)
    
    # Prepare results for JSON
    results_dict = {
        'error_ranking': [
            {'metric': metric, 'rmse': float(rmse)} 
            for metric, rmse in error_ranking
        ],
        'metrics': metrics
    }
    
    json_path = output_path / 'features_evaluation_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    
    return csv_path, json_path

def print_summary(metrics: Dict[str, Any], num_samples: int):
    """Print evaluation summary to console."""
    print("\n" + "="*120)
    print("=== Features Evaluation Summary ===")
    print("="*120)
    print(f"Number of samples evaluated: {num_samples}")
    
    print(f"\n{'Metric':<30} {'MAE':>10} {'RMSE':>10} {'MAE Norm':>10} {'RMSE Norm':>11} {'MAE Std':>10} {'RMSE Std':>11} {'Exact %':>10}")
    print("-" * 120)
    
    # Sort by RMSE for display
    mae_dict = metrics['mae']
    rmse_dict = metrics['rmse']
    mae_normalized_dict = metrics['mae_normalized']
    rmse_normalized_dict = metrics['rmse_normalized']
    mae_std_dict = metrics['mae_std']
    rmse_std_dict = metrics['rmse_std']
    exact_dict = metrics['exact_match_percentage']
    
    # Get all metrics sorted by RMSE (descending)
    metrics_sorted = sorted(
        [(m, mae_dict[m], rmse_dict[m], mae_normalized_dict[m], rmse_normalized_dict[m], 
          mae_std_dict[m], rmse_std_dict[m], exact_dict[m]) 
         for m in rmse_dict.keys() if not np.isnan(rmse_dict[m])],
        key=lambda x: x[2],  # Sort by RMSE (index 2)
        reverse=True
    )
    
    for metric, mae, rmse, mae_norm, rmse_norm, mae_std, rmse_std, exact in metrics_sorted[:20]:  # Show top 20 worst
        print(f"{metric:<30} {mae:>10.6f} {rmse:>10.6f} {mae_norm:>10.6f} {rmse_norm:>11.6f} {mae_std:>10.6f} {rmse_std:>11.6f} {exact:>9.2f}%")
    
    if len(metrics_sorted) > 20:
        print(f"\n... and {len(metrics_sorted) - 20} more metrics ...")
    
    print("\n" + "="*120)
    print("=== Overall Statistics ===")
    print("="*120)
    overall = metrics['overall']
    print(f"Mean MAE:              {overall['mean_mae']:.6f}")
    print(f"Mean RMSE:             {overall['mean_rmse']:.6f}")
    print(f"Mean Exact Match:      {overall['mean_exact_match_pct']:.2f}%")
    print(f"Median MAE:            {overall['median_mae']:.6f}")
    print(f"Median RMSE:           {overall['median_rmse']:.6f}")
    print(f"Median Exact Match:    {overall['median_exact_match_pct']:.2f}%")
    print("="*120)
    
    # Show best performing metrics
    print("\n=== Top 10 Best Performing Metrics (by RMSE) ===")
    best_metrics = sorted(
        [(m, rmse_dict[m], mae_dict[m], rmse_normalized_dict[m], exact_dict[m]) 
         for m in rmse_dict.keys() if not np.isnan(rmse_dict[m])],
        key=lambda x: x[1]
    )[:10]
    
    for i, (metric, rmse, mae, rmse_norm, exact) in enumerate(best_metrics, 1):
        print(f"{i:2d}. {metric:<35} RMSE: {rmse:>10.6f}  MAE: {mae:>10.6f}  RMSE_Norm: {rmse_norm:>10.6f}  Exact: {exact:>6.2f}%")
    
    # Show worst performing metrics
    print("\n=== Top 10 Worst Performing Metrics (by RMSE) ===")
    worst_metrics = sorted(
        [(m, rmse_dict[m], mae_dict[m], rmse_normalized_dict[m], exact_dict[m]) 
         for m in rmse_dict.keys() if not np.isnan(rmse_dict[m])],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    for i, (metric, rmse, mae, rmse_norm, exact) in enumerate(worst_metrics, 1):
        print(f"{i:2d}. {metric:<35} RMSE: {rmse:>10.6f}  MAE: {mae:>10.6f}  RMSE_Norm: {rmse_norm:>10.6f}  Exact: {exact:>6.2f}%")
    
    print("="*120 + "\n")

@click.command()
@click.option('--predictions-csv', required=True, help='Path to the predictions CSV file (features.csv)')
@click.option('--ground-truth-csv', required=True, help='Path to the ground truth CSV file (features-gt.csv)')
@click.option('--output-dir', default='./evaluation_results', help='Directory to save evaluation results')
def evaluate(predictions_csv: str, ground_truth_csv: str, output_dir: str):
    """
    Evaluate agreement between prediction metrics and ground truth metrics.
    
    Calculates MAE, RMSE, and exact match percentage for all linguistic features
    from cncadc to synstrutt and from verb_agreement_score to wrdverb.
    """
    print("Loading datasets...")
    predictions = load_csv_data(predictions_csv)
    ground_truth = load_csv_data(ground_truth_csv)
    
    print(f"Predictions loaded: {len(predictions)} rows")
    print(f"Ground truth loaded: {len(ground_truth)} rows")
    
    # Get metrics columns
    metrics_cols = get_metrics_columns()
    print(f"\nEvaluating {len(metrics_cols)} metrics...")
    
    # Match datasets
    matched_df = match_datasets(predictions, ground_truth, metrics_cols)
    print(f"Successfully matched {len(matched_df)} samples")
    
    if len(matched_df) == 0:
        print("No matched data found. Check your datasets.")
        return
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics = calculate_all_metrics(matched_df, metrics_cols)
    
    # Save results
    csv_path, json_path = save_results(matched_df, metrics, output_dir)
    
    # Print summary
    print_summary(metrics, len(matched_df))
    
    print(f"\n✓ Detailed results saved to: {csv_path}")
    print(f"✓ Metrics saved to: {json_path}")

if __name__ == '__main__':
    evaluate()

