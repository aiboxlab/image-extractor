from typing import List, Dict, Any
import time
import base64
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from config import cfg
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace
from model.essay_evaluate import EssayEvaluation
from PIL import Image
from io import BytesIO

PROMPT_INSTRUCTION_DIRECT_ESSAY = """
Avalie a redação diretamente da imagem, de acordo com os cinco critérios de avaliação do ENEM, atribuindo uma nota entre 0 e 200 para cada competência, totalizando um máximo de 1000 pontos. A pontuação deve ser dada em intervalos de 40 pontos, e a distribuição deve se aproximar das correções oficiais de redações semelhantes. 

Considere as seguintes competências:
Competência 1: Demonstrar domínio da modalidade escrita formal da língua portuguesa.
Competência 2: Compreender a proposta de redação e aplicar conceitos das várias áreas de conhecimento para desenvolver o tema, dentro dos limites estruturais do texto dissertativo-argumentativo em prosa.
Competência 3: Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista.
Competência 4: Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação.
Competência 5: Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos.

O formato da resposta deve ser o seguinte:
- C1: [nota]
- C2: [nota]
- C3: [nota]
- C4: [nota]
- C5: [nota]

Raciocine sobre a justificativa da sua resposta, explicando por que você fez as escolhas que realmente fez.
Pense nas etapas passo a passo.

Tema da redação: {prompt_text}
Imagem da redação: {image_data}
"""

def convert_base64(image_path: Path, max_width: int = 1500) -> str:
    """
    Converte a imagem para string Base64, redimensionando-a 
    se a largura for maior que 'max_width' para economizar tokens.
    """
    try:
        # 1. Abrir a imagem
        img = Image.open(image_path)
        
        # 2. Verificar e Redimensionar (para reduzir tokens)
        if img.width > max_width:
            # Calcular a nova altura mantendo a proporção (aspect ratio)
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            # Redimensionar
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # 3. Salvar a imagem redimensionada em um buffer de bytes
        buffer = BytesIO()
        # Salva como JPEG com qualidade padrão (pode adicionar 'quality=85' para compressão extra)
        img.save(buffer, format="JPEG") 
        
        # 4. Obter os bytes do buffer
        bytes_data = buffer.getvalue()
        
        # 5. Codificar para Base64
        return base64.b64encode(bytes_data).decode("utf-8")
        
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        # Se falhar, retorna o método original como fallback (mas pode estourar o limite)
        bytes_original = image_path.read_bytes()
        return base64.b64encode(bytes_original).decode("utf-8")

def create_essay_evaluation_from_image_chain(chat_model: BaseChatModel):
    """Create evaluation chain for processing images directly"""
    if isinstance(chat_model, ChatOpenAI):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador de redações experiente com foco nos critérios de avaliação do ENEM."),
            ("user", [
                {"type": "text", "text": PROMPT_INSTRUCTION_DIRECT_ESSAY},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                }
            ])
        ])
    elif isinstance(chat_model, ChatVertexAI):
        prompt_template = ChatPromptTemplate.from_messages([
            ("user", [
                {"type": "text", "text": PROMPT_INSTRUCTION_DIRECT_ESSAY},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                }
            ])
        ])
    elif isinstance(chat_model, ChatAnthropic):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador de redações experiente com foco nos critérios de avaliação do ENEM."),
            ("user", [
                {"type": "text", "text": PROMPT_INSTRUCTION_DIRECT_ESSAY},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                }
            ])
        ])
    elif isinstance(chat_model, ChatMistralAI):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador de redações experiente com foco nos critérios de avaliação do ENEM."),
            ("user", [
                {"type": "text", "text": PROMPT_INSTRUCTION_DIRECT_ESSAY},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                }
            ])
        ])
    elif isinstance(chat_model, ChatOllama):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador de redações experiente com foco nos critérios de avaliação do ENEM."),
            ("user", [
                {"type": "text", "text": PROMPT_INSTRUCTION_DIRECT_ESSAY},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                }
            ])
        ])
    elif isinstance(chat_model, ChatHuggingFace):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador de redações experiente com foco nos critérios de avaliação do ENEM."),
            ("user", PROMPT_INSTRUCTION_DIRECT_ESSAY + """Retorne APENAS AS NOTAS com o seguinte formato, SEM NENHUMA INTRODUÇÃO, EXPLICAÇÃO, JUSTIFICATIVA OU QUALQUER OUTRO TEXTO ADICIONAL. ENVOLVA SUAS NOTAS EXATAMENTE ENTRE OS MARCADORES '---INICIO_NOTAS---' E '---FIM_NOTAS---'.
                ---INICIO_NOTAS---
                C1: [nota]
                C2: [nota]
                C3: [nota]
                C4: [nota]
                C5: [nota]
                ---FIM_NOTAS---
             """),
        ])
        return prompt_template | chat_model  
    else:
        raise ValueError(f"Model type {type(chat_model)} not supported")
    
    return prompt_template | chat_model.with_structured_output(EssayEvaluation)

def parse_huggingface_response_from_image(response: str, essay_id: int) -> EssayEvaluation:
    """Parse HuggingFace response for image-based evaluation"""
    try:
        start = response.index("---INICIO_NOTAS---") + len("---INICIO_NOTAS---")
        end = response.index("---FIM_NOTAS---")
        response_section = response[start:end].strip()
        print("response: ", response_section)

        scores = {}
        for line in response_section.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = int(value.strip())
                scores[key] = value

        if len(scores) != 5:
            evaluation = EssayEvaluation(
                c1=0,
                c2=0,
                c3=0,
                c4=0,
                c5=0,
            )
            return evaluation
        
        else:
            evaluation = EssayEvaluation(
                c1=scores.get('c1', 0),
                c2=scores.get('c2', 0),
                c3=scores.get('c3', 0),
                c4=scores.get('c4', 0),
                c5=scores.get('c5', 0),
                total_score=scores.get('c1', 0) + scores.get('c2', 0) + scores.get('c3', 0) + scores.get('c4', 0) + scores.get('c5', 0),
                id=essay_id
            )
            
            return evaluation
    except (ValueError, IndexError) as e:
        print(f"Error parsing response: {e}")
        return EssayEvaluation(c1=0, c2=0, c3=0, c4=0, c5=0, total_score=0, id=essay_id)

def execute_essay_evaluation_from_image(
    chat_model: BaseChatModel, image_path: Path, prompt_text: str, essay_id: int
) -> Dict[str, Any]:
    """Execute essay evaluation directly from image"""
    start_time = time.time() 
    chain = create_essay_evaluation_from_image_chain(chat_model)
    image_data = convert_base64(image_path, max_width=1500)
    
    if isinstance(chat_model, ChatHuggingFace):
        response = chain.invoke({
            "image_data": image_data,
            "prompt_text": prompt_text
        })
        print("raw response: ", response.content)
        evaluation = parse_huggingface_response_from_image(response.content, essay_id)
    else:
        evaluation = chain.invoke({
            "image_data": image_data,
            "prompt_text": prompt_text
        })
        
        evaluation.total_score = evaluation.c1 + evaluation.c2 + evaluation.c3 + evaluation.c4 + evaluation.c5
        evaluation.id = essay_id
    
    elapsed_time = time.time() - start_time
    result = evaluation.model_dump()
    print(result)
    result["elapsed"] = elapsed_time
    
    return result

def evaluate_essays_from_dataset_test(
    chat_model: BaseChatModel, 
    dataset_path: str = "/home/jamillalobo/Documentos/research/image-extractor-2/image_extractor/dataset-test",
    prompt_text: str = "Tema padrão de redação ENEM"
) -> List[Dict[str, Any]]:
    """
    Evaluate all images in dataset-test folder using PROMPT_INSTRUCTION_DIRECT_ESSAY
    Returns only the scores like in essay_evaluation.py
    """
    dataset_dir = Path(dataset_path)
    results = []
    essay_id = 0
    
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_path}")
    
    # Get all subdirectories (each represents an essay)
    essay_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    essay_dirs.sort()  # Sort for consistent processing order
    
    print(f"Found {len(essay_dirs)} essay directories to process")
    
    for essay_dir in essay_dirs:
        print(f"Processing essay directory: {essay_dir.name}")
        
        # Get all image files in the directory
        image_files = list(essay_dir.glob("*.jpg")) + list(essay_dir.glob("*.jpeg")) + list(essay_dir.glob("*.png"))
        image_files.sort()  # Sort by filename
        
        if not image_files:
            print(f"No image files found in {essay_dir}")
            continue
        
        # For now, process the first image of each essay
        # You can modify this to process all images or combine them
        first_image = image_files[0]
        
        try:
            result = execute_essay_evaluation_from_image(
                chat_model, first_image, prompt_text, essay_id
            )
            result["essay_dir"] = essay_dir.name
            result["image_file"] = first_image.name
            results.append(result)
            essay_id += 1
            
        except Exception as e:
            print(f"Error processing {first_image}: {e}")
            # Add a failed result to maintain consistency
            results.append({
                "c1": 0, "c2": 0, "c3": 0, "c4": 0, "c5": 0,
                "total_score": 0, "id": essay_id, "elapsed": 0,
                "essay_dir": essay_dir.name,
                "image_file": first_image.name,
                "error": str(e)
            })
            essay_id += 1
    
    return results

class EssayEvaluatorFromImage:
    """Essay evaluator that works directly with images"""
    def __init__(self, model):
        self.model = model
        
    def evaluate_essay_from_image(self, image_path: Path, prompt_text: str, essay_id: int) -> Dict[str, Any]:
        """Evaluate a single essay from image"""
        return execute_essay_evaluation_from_image(self.model, image_path, prompt_text, essay_id)
    
    def evaluate_essays_from_dataset_test(self, dataset_path: str = None, prompt_text: str = "Tema padrão de redação ENEM") -> List[Dict[str, Any]]:
        """Evaluate all essays from dataset-test folder"""
        if dataset_path is None:
            dataset_path = "/home/jamillalobo/Documentos/research/image-extractor-2/image_extractor/dataset-test"
        return evaluate_essays_from_dataset_test(self.model, dataset_path, prompt_text)

class OpenAiEssayEvaluatorFromImage(EssayEvaluatorFromImage):
    def __init__(self):
        super().__init__(cfg.chat_openai)

class VertexAiEssayEvaluatorFromImage(EssayEvaluatorFromImage):
    def __init__(self):
        super().__init__(cfg.vertexai_gemini)

class AnthropicEssayEvaluatorFromImage(EssayEvaluatorFromImage):
    def __init__(self):
        super().__init__(cfg.chat_anthropic)

class MistralEssayEvaluatorFromImage(EssayEvaluatorFromImage):
    def __init__(self):
        super().__init__(cfg.chat_mistral)

class OllamaEssayEvaluatorFromImage(EssayEvaluatorFromImage):
    def __init__(self):
        super().__init__(cfg.chat_ollama)

class HuggingFaceEssayEvaluatorFromImage(EssayEvaluatorFromImage):
    def __init__(self):
        super().__init__(cfg.chat_huggingface)
