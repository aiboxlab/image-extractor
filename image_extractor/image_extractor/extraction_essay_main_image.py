from pathlib import Path
from enum import StrEnum
import click
import time
import json
import os
import re
import gc
import sys
from typing import Set, Tuple

from service.essay_evaluation_from_image import AnthropicEssayEvaluatorFromImage, HuggingFaceEssayEvaluatorFromImage, MistralEssayEvaluatorFromImage, OllamaEssayEvaluatorFromImage, OpenAiEssayEvaluatorFromImage, VertexAiEssayEvaluatorFromImage


class Model(StrEnum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"  
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    LLAMA = "llama"
    HUGGINGFACE = "huggingface"

@click.group()
def cli():
    pass

def load_themes_from_file(themes_file: str = "temas_red.txt") -> dict:
    """Load themes from temas_red.txt file"""
    themes_path = Path(__file__).parent / themes_file
    themes = {}
    
    if not themes_path.exists():
        click.echo(f"Warning: Themes file {themes_path} not found. Using default theme.")
        return {}
    
    with open(themes_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                # Parse format: "lote X = theme"
                match = re.match(r'lote\s+(\d+)\s*=\s*(.+)', line)
                if match:
                    lote_num = match.group(1)
                    theme = match.group(2).strip()
                    themes[lote_num] = theme
    
    return themes

def get_theme_for_lote(lote_dir_name: str, themes: dict) -> str:
    """Get theme for lote directory based on its name"""
    # If themes is empty, return default
    if not themes:
        return "Tema de redação ENEM"
    
    # Extract lote number from directory name (format: "Lote=X")
    match = re.match(r'Lote=(\d+)', lote_dir_name)
    if match:
        lote_num = match.group(1)
        if lote_num in themes:
            return themes[lote_num]
    
    return "Tema de redação ENEM"

def scan_existing_results(output_dir: Path) -> Set[Tuple[str, str]]:
    """
    Escaneia o diretório de saída para identificar quais redações já foram processadas.
    Retorna um set de tuplas (lote, imagem) já processadas.
    """
    existing_results = set()
    
    if not output_dir.exists():
        return existing_results
    
    # Padrão para extrair lote e imagem dos nomes dos arquivos JSON
    # Exemplo: vertexai_2.0-flash_essay_image_614_Lote=28_39.json
    pattern = r'.*_essay_image_\d+_(Lote=\d+)_(.+)\.json$'
    
    for json_file in output_dir.glob("*.json"):
        match = re.match(pattern, json_file.name)
        if match:
            lote = match.group(1)  # "Lote=28"
            image_name = match.group(2)  # "39"
            existing_results.add((lote, image_name))
    
    return existing_results

def should_process_image(lote_name: str, image_stem: str, existing_results: Set[Tuple[str, str]]) -> bool:
    """
    Verifica se uma imagem deve ser processada baseado nos resultados existentes.
    """
    return (lote_name, image_stem) not in existing_results

def clear_cache_and_memory():
    """
    Limpa cache e memória para evitar problemas de rate limiting e resource exhausted.
    """
    # Força garbage collection
    gc.collect()
    
    # Limpa cache de importações se possível
    try:
        # Limpa cache do sistema
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
    except:
        pass
    
    # Pequena pausa para dar tempo ao sistema
    time.sleep(0.5)

def create_evaluator(model: str):
    """
    Cria uma nova instância do evaluator baseado no modelo.
    """
    evaluator = None
    model_type = ""
    
    if model == Model.OPENAI.value:
        evaluator = OpenAiEssayEvaluatorFromImage()
        model_type = os.getenv("OPENAI_MODEL", "gpt-4o").replace("gpt-", "")
    elif model == Model.VERTEXAI.value:
        evaluator = VertexAiEssayEvaluatorFromImage()
        model_type = os.getenv("GEMINI_MODEL", "gemini-flash").replace("gemini-", "")
    elif model == Model.ANTHROPIC.value:
        evaluator = AnthropicEssayEvaluatorFromImage()
        model_type = os.getenv("ANTHROPIC_MODEL", "claude-sonnet").replace("claude-", "")
    elif model == Model.MISTRAL.value:
        evaluator = MistralEssayEvaluatorFromImage()
        model_type = os.getenv("MISTRAL_MODEL", "mistral-large").replace("mistral-", "")
    elif model == Model.LLAMA.value:
        evaluator = OllamaEssayEvaluatorFromImage()
        model_type = os.getenv("OLLAMA_MODEL", "llama3").replace("llama-", "")
    elif model == Model.HUGGINGFACE.value:
        evaluator = HuggingFaceEssayEvaluatorFromImage()
        model_type = os.getenv("HUGGINGFACE_REPO_ID", "huggingface").replace("huggingface-", "")
        model_type = model_type.replace("/", "_")
    else:
        raise ValueError(f"Unsupported model type: {model}")
    
    return evaluator, model_type

@cli.command()
@click.option(
    "--dataset_path", 
    default="dataset-test",
    help="Path to the dataset directory containing essay images"
)
@click.option(
    "--model",
    default=Model.ANTHROPIC.value,
    prompt="The model you want to use",
    type=click.Choice([m.value for m in Model]),
    help="The model type to be used",
)
@click.option(
    "--output_dir", default="./essay_results_from_images", help="Directory to save evaluation results"
)
@click.option(
    "--start_index", default=0, help="Index to start processing from"
)
@click.option(
    "--end_index", default=None, type=int, help="Index to end processing at"
)
@click.option(
    "--themes_file", default="temas_red.txt", help="Path to themes file"
)
@click.option(
    "--delay_between_requests", default=2.0, type=float, help="Delay in seconds between API requests to avoid rate limiting"
)
@click.option(
    "--clear_cache", is_flag=True, default=True, help="Clear cache and memory between requests"
)
@click.option(
    "--reinit_evaluator_every", default=10, type=int, help="Reinitialize evaluator every N requests to prevent memory buildup"
)
def evaluate_essays_from_images(dataset_path: str, model: str, output_dir: str, start_index: int, end_index: int, themes_file: str, delay_between_requests: float, clear_cache: bool, reinit_evaluator_every: int):
    """Evaluate essays from images in dataset-test using themes from temas_red.txt"""
    start = time.time()
    
    # Resolve dataset path
    if not os.path.isabs(dataset_path):
        dataset_path = Path(__file__).parent.parent / dataset_path
    else:
        dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory {dataset_path} does not exist.")
    
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    
    # Load themes from file
    themes = load_themes_from_file(themes_file)
    click.echo(f"Loaded {len(themes)} themes from {themes_file}")
    
    # Scan existing results to avoid reprocessing
    existing_results = scan_existing_results(output_dir_path)
    click.echo(f"Found {len(existing_results)} already processed essays in {output_dir}")
    
    # Initialize evaluator based on model
    evaluator, model_type = create_evaluator(model)
    click.echo(f"Initialized {model} evaluator with model type: {model_type}")
    
    # Configurações de rate limiting
    click.echo(f"Rate limiting settings:")
    click.echo(f"  • Delay between requests: {delay_between_requests}s")
    click.echo(f"  • Clear cache: {'Yes' if clear_cache else 'No'}")
    click.echo(f"  • Reinitialize evaluator every: {reinit_evaluator_every} requests")

    # Get all lote directories
    lote_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("Lote=")]
    lote_dirs.sort(key=lambda x: int(re.findall(r'\d+', x.name)[0]) if re.findall(r'\d+', x.name) else 0)
    
    click.echo(f"Found {len(lote_dirs)} lote directories to process")
    
    # Process each lote directory
    essay_counter = 0
    processed_count = 0
    skipped_count = 0
    
    for lote_dir in lote_dirs:
        # Get theme for this lote
        theme = get_theme_for_lote(lote_dir.name, themes)
        click.echo(f"\n=== Processing {lote_dir.name} with theme: {theme} ===")
        
        # Get imgs subdirectory
        imgs_dir = lote_dir / "imgs"
        if not imgs_dir.exists():
            click.echo(f"No imgs directory found in {lote_dir.name}. Skipping.")
            continue
        
        # Get all image files from imgs directory
        image_files = list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.jpeg")) + list(imgs_dir.glob("*.png"))
        image_files.sort(key=lambda x: int(re.findall(r'\d+', x.stem)[0]) if re.findall(r'\d+', x.stem) else 0)
        
        if not image_files:
            click.echo(f"No image files found in {imgs_dir}. Skipping.")
            continue
        
        click.echo(f"Found {len(image_files)} images in {lote_dir.name}")
        
        # Process each image in the lote
        for image_file in image_files:
            # Check if this image was already processed (based on lote and image name)
            if not should_process_image(lote_dir.name, image_file.stem, existing_results):
                click.echo(f"Essay ({lote_dir.name}/{image_file.name}) already processed. Skipping.")
                skipped_count += 1
                essay_counter += 1
                continue
            
            # Apply start/end index filtering
            if essay_counter < start_index:
                essay_counter += 1
                continue
            if end_index is not None and essay_counter >= end_index:
                break
            
            output_file = output_dir_path / f"{model}_{model_type}_essay_image_{essay_counter}_{lote_dir.name}_{image_file.stem}.json"
            
            click.echo(f"Evaluating essay {essay_counter} ({lote_dir.name}/{image_file.name})")

            # Reinicializa evaluator periodicamente para evitar buildup de memória
            if reinit_evaluator_every > 0 and processed_count > 0 and processed_count % reinit_evaluator_every == 0:
                click.echo(f"Reinitializing evaluator after {processed_count} processed essays...")
                del evaluator
                clear_cache_and_memory()
                evaluator, _ = create_evaluator(model)
                click.echo("Evaluator reinitialized successfully")

            # Limpa cache e memória antes de cada requisição se habilitado
            if clear_cache:
                clear_cache_and_memory()

            start_essay = time.time()
            try:
                result = evaluator.evaluate_essay_from_image(
                    image_path=image_file,
                    prompt_text=theme,
                    essay_id=essay_counter
                )
                elapsed_essay = time.time() - start_essay
                
                # Add metadata to result
                result["elapsed"] = elapsed_essay
                result["lote"] = lote_dir.name
                result["image_file"] = image_file.name
                result["theme"] = theme
                result["model"] = model
                result["model_type"] = model_type
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                    
                click.echo(f"Evaluation saved to {output_file}")
                click.echo(f"Scores: C1={result['c1']}, C2={result['c2']}, C3={result['c3']}, C4={result['c4']}, C5={result['c5']}, Total={result['total_score']}")
                processed_count += 1
                
                # Delay entre requisições para evitar rate limiting
                if delay_between_requests > 0:
                    click.echo(f"Waiting {delay_between_requests}s before next request...")
                    time.sleep(delay_between_requests)
                
            except Exception as e:
                error_msg = str(e)
                click.echo(f"Error evaluating essay {essay_counter} ({lote_dir.name}/{image_file.name}): {error_msg}")
                
                # Tratamento especial para erro 429 (Rate Limit)
                if "429" in error_msg or "Resource exhausted" in error_msg or "rate limit" in error_msg.lower():
                    click.echo("⚠️  Rate limit detected! Increasing delay and clearing cache...")
                    clear_cache_and_memory()
                    
                    # Reinicializa o evaluator em caso de rate limit
                    del evaluator
                    time.sleep(5)  # Pausa maior para rate limit
                    evaluator, _ = create_evaluator(model)
                    click.echo("Evaluator reinitialized after rate limit error")
                    
                    # Aumenta o delay temporariamente
                    extended_delay = max(delay_between_requests * 2, 5.0)
                    click.echo(f"Using extended delay of {extended_delay}s for next request")
                    time.sleep(extended_delay)
                
                # Save error result
                error_result = {
                    "error": error_msg,
                    "lote": lote_dir.name,
                    "image_file": image_file.name,
                    "theme": theme,
                    "model": model,
                    "model_type": model_type,
                    "id": essay_counter,
                    "elapsed": 0,
                    "c1": 0, "c2": 0, "c3": 0, "c4": 0, "c5": 0, "total_score": 0
                }
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False)
                
                # Delay mesmo em caso de erro para evitar spam de requisições
                if delay_between_requests > 0 and "429" not in error_msg:
                    click.echo(f"Waiting {delay_between_requests}s after error before next request...")
                    time.sleep(delay_between_requests)
            
            essay_counter += 1
            
            # Break if we've reached the end index
            if end_index is not None and essay_counter >= end_index:
                break
        
        # Break if we've reached the end index
        if end_index is not None and essay_counter >= end_index:
            break
    
    end = time.time()
    click.echo(f"\n=== PROCESSING SUMMARY ===")
    click.echo(f"Total essays found: {essay_counter}")
    click.echo(f"Already processed (skipped): {skipped_count}")
    click.echo(f"Newly processed: {processed_count}")
    click.echo(f"Total elapsed time: {end - start:.2f} seconds")

if __name__ == "__main__":
    cli()

# Exemplos de uso:
# 
# Processamento normal com configurações otimizadas para evitar rate limiting:
# python extraction_essay_main_image.py evaluate-essays-from-images --model vertexai --delay_between_requests 3.0
#
# Processamento mais agressivo (mais rápido, mas pode dar rate limit):
# python extraction_essay_main_image.py evaluate-essays-from-images --model vertexai --delay_between_requests 1.0
#
# Processamento conservador (mais lento, mas mais seguro):
# python extraction_essay_main_image.py evaluate-essays-from-images --model vertexai --delay_between_requests 5.0 --reinit_evaluator_every 5
