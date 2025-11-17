from pathlib import Path
from enum import StrEnum
import click
import time
import json
import os
import re

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
def evaluate_essays_from_images(dataset_path: str, model: str, output_dir: str, start_index: int, end_index: int, themes_file: str):
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
    
    # Initialize evaluator based on model
    evaluator = None
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

    # Get all lote directories
    lote_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("Lote=")]
    lote_dirs.sort(key=lambda x: int(re.findall(r'\d+', x.name)[0]) if re.findall(r'\d+', x.name) else 0)
    
    click.echo(f"Found {len(lote_dirs)} lote directories to process")
    
    # Process each lote directory
    essay_counter = 0
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
            # Apply start/end index filtering
            if essay_counter < start_index:
                essay_counter += 1
                continue
            if end_index is not None and essay_counter >= end_index:
                break
            
            output_file = output_dir_path / f"{model}_{model_type}_essay_image_{essay_counter}_{lote_dir.name}_{image_file.stem}.json"
            
            if output_file.exists():
                click.echo(f"Essay {essay_counter} ({lote_dir.name}/{image_file.name}) already evaluated. Skipping.")
                essay_counter += 1
                continue
            
            click.echo(f"Evaluating essay {essay_counter} ({lote_dir.name}/{image_file.name})")

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
                
            except Exception as e:
                click.echo(f"Error evaluating essay {essay_counter} ({lote_dir.name}/{image_file.name}): {e}")
                
                # Save error result
                error_result = {
                    "error": str(e),
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
            
            essay_counter += 1
            
            # Break if we've reached the end index
            if end_index is not None and essay_counter >= end_index:
                break
        
        # Break if we've reached the end index
        if end_index is not None and essay_counter >= end_index:
            break
    
    end = time.time()
    click.echo(f"Total elapsed time: {end - start:.2f} seconds")

if __name__ == "__main__":
    cli()
