import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import click
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Tuple
import math

def load_original_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def load_prediction_files(predictions_dir: str, model_filter: str = None) -> List[Dict[str, Any]]:
    predictions_path = Path(predictions_dir)
    if model_filter:
        prediction_files = list(predictions_path.glob(f"{model_filter}*.json"))
    else:
        prediction_files = list(predictions_path.glob("*.json"))

    def extract_file_id(file_path: Path) -> int:
        try:
            return int(file_path.stem.split('_')[-1])
        except (ValueError, IndexError):
            return float('inf')

    prediction_files.sort(key=extract_file_id)

    predictions = []
    for file in prediction_files:
        with open(file, 'r', encoding='utf-8') as f:
            try:
                prediction = json.load(f)
                prediction['filename'] = file.name
                predictions.append(prediction)
            except json.JSONDecodeError:
                print(f"Error loading JSON from {file}")
    return predictions

def match_predictions_with_originals(predictions: List[Dict[str, Any]], originals: pd.DataFrame) -> List[Dict[str, Any]]:
    matched_data = []
    total = min(len(predictions), len(originals))

    # print("Ordem dos arquivos de predição:")
    # for idx, pred in enumerate(predictions[:5]):
    #     print(f"  {idx}: {pred.get('filename', 'sem nome')}")
    
    for idx in range(total):
        pred = predictions[idx]
        original = originals.iloc[idx]
        matched = {
            'id': idx,  
            'filename': pred.get('filename', ''),
            'c1_model': pred['c1'],
            'c2_model': pred['c2'],
            'c3_model': pred['c3'],
            'c4_model': pred['c4'],
            'c5_model': pred['c5'],
            'score_model': pred['total_score'],
            'c1_target': original['avg_c1'],
            'c2_target': original['avg_c2'],
            'c3_target': original['avg_c3'],
            'c4_target': original['avg_c4'],
            'c5_target': original['avg_c5'],
            'score_target': original['avg_total'],
            'elapsed': pred.get('elapsed', 0)
        }
        matched_data.append(matched)
    
    return matched_data    

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def calculate_metrics(matched_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    c1_pred = np.array([d['c1_model'] for d in matched_data])
    c2_pred = np.array([d['c2_model'] for d in matched_data])
    c3_pred = np.array([d['c3_model'] for d in matched_data])
    c4_pred = np.array([d['c4_model'] for d in matched_data])
    c5_pred = np.array([d['c5_model'] for d in matched_data])
    score_pred = np.array([d['score_model'] for d in matched_data])

    c1_true = np.array([d['c1_target'] for d in matched_data])
    c2_true = np.array([d['c2_target'] for d in matched_data])
    c3_true = np.array([d['c3_target'] for d in matched_data])
    c4_true = np.array([d['c4_target'] for d in matched_data])
    c5_true = np.array([d['c5_target'] for d in matched_data])
    score_true = np.array([d['score_target'] for d in matched_data])

    mse_c1 = float(mean_squared_error(c1_true, c1_pred))
    mse_c2 = float(mean_squared_error(c2_true, c2_pred))
    mse_c3 = float(mean_squared_error(c3_true, c3_pred))
    mse_c4 = float(mean_squared_error(c4_true, c4_pred))
    mse_c5 = float(mean_squared_error(c5_true, c5_pred))
    mse_score = float(mean_squared_error(score_true, score_pred))

    rmse_c1 = float(root_mean_squared_error(c1_true, c1_pred))
    rmse_c2 = float(root_mean_squared_error(c2_true, c2_pred))
    rmse_c3 = float(root_mean_squared_error(c3_true, c3_pred))
    rmse_c4 = float(root_mean_squared_error(c4_true, c4_pred))
    rmse_c5 = float(root_mean_squared_error(c5_true, c5_pred))
    rmse_score = float(root_mean_squared_error(score_true, score_pred))

    mae_c1 = float(mean_absolute_error(c1_true, c1_pred))
    mae_c2 = float(mean_absolute_error(c2_true, c2_pred))
    mae_c3 = float(mean_absolute_error(c3_true, c3_pred))
    mae_c4 = float(mean_absolute_error(c4_true, c4_pred))
    mae_c5 = float(mean_absolute_error(c5_true, c5_pred))
    mae_score = float(mean_absolute_error(score_true, score_pred))

    nmse_c1 = float(mse_c1 / (200**2))
    nmse_c2 = float(mse_c2 / (200**2))
    nmse_c3 = float(mse_c3 / (200**2))
    nmse_c4 = float(mse_c4 / (200**2))
    nmse_c5 = float(mse_c5 / (200**2))
    nmse_score = float(mse_score / (1000**2))

    nrmse_c1 = float(rmse_c1 / 200)
    nrmse_c2 = float(rmse_c2 / 200)
    nrmse_c3 = float(rmse_c3 / 200)
    nrmse_c4 = float(rmse_c4 / 200)
    nrmse_c5 = float(rmse_c5 / 200)
    nrmse_score = float(rmse_score / 1000)

    nmae_c1 = float(mae_c1 / 200)
    nmae_c2 = float(mae_c2 / 200)
    nmae_c3 = float(mae_c3 / 200)
    nmae_c4 = float(mae_c4 / 200)
    nmae_c5 = float(mae_c5 / 200)
    nmae_score = float(mae_score / 1000)

    res_error_c1 = (c1_true - c1_pred).tolist() 
    res_error_c2 = (c2_true - c2_pred).tolist()
    res_error_c3 = (c3_true - c3_pred).tolist()
    res_error_c4 = (c4_true - c4_pred).tolist()
    res_error_c5 = (c5_true - c5_pred).tolist()
    res_error_score = (score_true - score_pred).tolist()

    exact_c1 = float(sum(1 for t, p in zip(c1_true, c1_pred) if t == p) / len(c1_true) * 100)
    exact_c2 = float(sum(1 for t, p in zip(c2_true, c2_pred) if t == p) / len(c2_true) * 100)
    exact_c3 = float(sum(1 for t, p in zip(c3_true, c3_pred) if t == p) / len(c3_true) * 100)
    exact_c4 = float(sum(1 for t, p in zip(c4_true, c4_pred) if t == p) / len(c4_true) * 100)
    exact_c5 = float(sum(1 for t, p in zip(c5_true, c5_pred) if t == p) / len(c5_true) * 100)
    exact_score = float(sum(1 for t, p in zip(score_true, score_pred) if t == p) / len(score_true) * 100)

    metrics = {
        'mse': {
            'c1': mse_c1,
            'c2': mse_c2,
            'c3': mse_c3,
            'c4': mse_c4,
            'c5': mse_c5,
            'total_score': mse_score
        },
        'rmse': {
            'c1': rmse_c1,
            'c2': rmse_c2,
            'c3': rmse_c3,
            'c4': rmse_c4,
            'c5': rmse_c5,
            'total_score': rmse_score
        },
        'normalized_mse': {
            'c1': nmse_c1,
            'c2': nmse_c2,
            'c3': nmse_c3,
            'c4': nmse_c4,
            'c5': nmse_c5,
            'total_score': nmse_score
        },
        'normalized_rmse': {
            'c1': nrmse_c1,
            'c2': nrmse_c2,
            'c3': nrmse_c3,
            'c4': nrmse_c4,
            'c5': nrmse_c5,
            'total_score': nrmse_score
        },
        'mae': {
            'c1': mae_c1,
            'c2': mae_c2,
            'c3': mae_c3,
            'c4': mae_c4,
            'c5': mae_c5,
            'total_score': mae_score
        },
        'normalized_mae': {
            'c1': nmae_c1,
            'c2': nmae_c2,
            'c3': nmae_c3,
            'c4': nmae_c4,
            'c5': nmae_c5,
            'total_score': nmae_score
        },
        'residual_error': {
            "c1": res_error_c1,
            "c2": res_error_c2,
            "c3": res_error_c3,
            "c4": res_error_c4,
            "c5": res_error_c5,
            "score": res_error_score
        },
        'exact_match_percentage': {
            'c1': exact_c1,
            'c2': exact_c2,
            'c3': exact_c3,
            'c4': exact_c4,
            'c5': exact_c5,
            'total_score': exact_score
        }
    }
    return metrics

def generate_error_ranking(metrics: Dict[str, Any]) -> List[Tuple[str, float]]:
    nrmse_values = metrics['normalized_rmse']
    competencies = ['c1', 'c2', 'c3', 'c4', 'c5']
    error_ranking = [(comp, nrmse_values[comp]) for comp in competencies]
    error_ranking.sort(key=lambda x: x[1])
    return error_ranking

def save_results(matched_data: List[Dict[str, Any]], metrics: Dict[str, Any], output_dir: str, model_name: str = None):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    prefix = f"{model_name}_" if model_name else ""
    df = pd.DataFrame(matched_data)
    csv_path = output_path / f'{prefix}essay_evaluation_results.csv'
    df.to_csv(csv_path, index=False)
    error_ranking = generate_error_ranking(metrics)
    ranking_dict = {
        'error_ranking': [{'competency': comp, 'rmse_normalized': error} for comp, error in error_ranking],
        'metrics': metrics
    }
    json_path = output_path / f'{prefix}evaluation_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ranking_dict, f, indent=2)
    return csv_path, json_path

def print_summary(metrics: Dict[str, Any], error_ranking: List[Tuple[str, float]], num_samples: int):
    print("\n=== Essay Evaluation Summary ===")
    print(f"Number of samples evaluated: {num_samples}")
    print("\nMSE (Não normalizado, lower is better):")
    for comp, value in metrics['mse'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nRMSE (Não normalizado, lower is better):")
    for comp, value in metrics['rmse'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nNormalized MSE (0-1 scale, lower is better):")
    for comp, value in metrics['normalized_mse'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nNormalized RMSE (0-1 scale, lower is better):")
    for comp, value in metrics['normalized_rmse'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nMAE (lower is better, competencies in range 0-200 and total_score 0-1000):")
    for comp, value in metrics['mae'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nNormalized MAE (0-1 scale, lower is better):")
    for comp, value in metrics['normalized_mae'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nResidual Errors (lower is better):")
    for comp, errors in metrics['residual_error'].items():
        print(f"  Mean: {np.mean(errors):.4f}, Std: {np.std(errors):.4f}, Max: {np.max(errors):.4f}, Min: {np.min(errors):.4f}")
    print("\nExact Match Percentage:")
    for comp, value in metrics['exact_match_percentage'].items():
        print(f"  {comp}: {value:.2f}%")
    print("\nCompetencies Ranked by Performance (from best to worst):")
    for i, (comp, error) in enumerate(error_ranking, 1):
        comp_name = {
            'c1': 'Competência 1 (formal language)',
            'c2': 'Competência 2 (understanding topic)',
            'c3': 'Competência 3 (argument organization)',
            'c4': 'Competência 4 (linguistic mechanisms)',
            'c5': 'Competência 5 (solution proposal)'
        }.get(comp, comp)
        print(f"  {i}. {comp_name}: {error:.4f} NRMSE")

@click.command()
@click.option('--csv', required=True, help='Path to the original CSV file with actual scores')
@click.option('--predictions-dir', required=True, help='Directory containing prediction JSON files')
@click.option('--output-dir', default='./evaluation_results', help='Directory to save evaluation results')
@click.option('--model', default=None, help='Optional filter for model names (e.g., "anthropic")')
def evaluate(csv: str, predictions_dir: str, output_dir: str, model: str):
    original_data = load_original_data(csv)
    predictions = load_prediction_files(predictions_dir, model)
    print(f"Loaded {len(predictions)} prediction files")
    matched_data = match_predictions_with_originals(predictions, original_data)
    print(f"Successfully matched {len(matched_data)} predictions with original data")
    if not matched_data:
        print("No matched data found. Check your IDs or file structure.")
        return
    metrics = calculate_metrics(matched_data)
    error_ranking = generate_error_ranking(metrics)
    csv_path, json_path = save_results(matched_data, metrics, output_dir, model)
    print_summary(metrics, error_ranking, len(matched_data))
    print(f"\nDetailed results saved to {csv_path}")
    print(f"Metrics saved to {json_path}")

if __name__ == '__main__':
    evaluate()
