"""
Borrowed from https://github.com/WujiangXu/AgenticMemory/blob/main/utils.py

@article{xu2025mem,
    title={A-mem: Agentic memory for llm agents},
    author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao
           and Zhang, Yongfeng},
    journal={arXiv preprint arXiv:2502.12110},
    year={2025}
}
"""

import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import nltk
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
from sentence_transformers.util import pytorch_cos_sim

# 加载环境变量（从项目根目录的 .env 文件）
current_file_dir = Path(__file__).parent  # metrics/
evaluation_dir = current_file_dir.parent  # evaluation/
repo_root = evaluation_dir.parent          # 项目根目录 mymem/
env_file = repo_root / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    # 如果项目根目录没有 .env，尝试当前目录
    load_dotenv()

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Initialize SentenceTransformer model (this will be reused)
# 优先使用本地模型，如果不存在则从 HuggingFace 下载
# 使用与检索阶段相同的模型，保证评估指标和实际检索性能的一致性
sentence_model = None

# 从环境变量读取模型路径，优先使用 MEM0_EMBED_MODEL
embed_model_env = os.getenv("MEM0_EMBED_MODEL", "").strip()
# 去除可能的引号（.env 文件中可能包含引号）
if embed_model_env.startswith('"') and embed_model_env.endswith('"'):
    embed_model_env = embed_model_env[1:-1]
elif embed_model_env.startswith("'") and embed_model_env.endswith("'"):
    embed_model_env = embed_model_env[1:-1]
embed_model_env = embed_model_env.strip()

if embed_model_env:
    # 如果环境变量是完整路径，直接使用
    if os.path.isabs(embed_model_env) and os.path.exists(embed_model_env):
        model_path = Path(embed_model_env)
        print(f"从环境变量 MEM0_EMBED_MODEL 读取模型路径: {model_path}")
    else:
        # 如果是相对路径或模型名称，尝试在 models 目录下查找
        current_dir = Path(__file__).parent  # metrics/
        evaluation_dir = current_dir.parent  # evaluation/
        repo_root = evaluation_dir.parent    # 项目根目录 mymem/
        model_path = repo_root / "models" / embed_model_env
        if not model_path.exists():
            # 如果不存在，尝试直接使用环境变量的值（可能是 HuggingFace 模型名）
            model_path = embed_model_env
else:
    # 默认模型
    default_model_name = "multi-qa-MiniLM-L6-cos-v1"
    current_dir = Path(__file__).parent  # metrics/
    evaluation_dir = current_dir.parent  # evaluation/
    repo_root = evaluation_dir.parent    # 项目根目录 mymem/
    model_path = repo_root / "models" / default_model_name
    if not model_path.exists():
        model_path = default_model_name

# 加载模型
if isinstance(model_path, Path) and model_path.exists() and model_path.is_dir():
    # 使用本地模型路径
    try:
        print(f"正在从本地路径加载模型: {model_path}")
        sentence_model = SentenceTransformer(str(model_path))
        print(f"✓ 成功加载本地模型")
        
        # 打印embedding模型参数
        print("=" * 80)
        print("Embedding Model Configuration (for evaluation):")
        print(f"  Model Path: {model_path}")
        if hasattr(sentence_model, 'get_sentence_embedding_dimension'):
            print(f"  Embedding Dimensions: {sentence_model.get_sentence_embedding_dimension()}")
        print("=" * 80)
    except Exception as e:
        print(f"警告: 无法从本地路径加载模型 {model_path}: {e}")
        print("尝试从 HuggingFace 下载...")
        try:
            # 尝试使用路径的最后一部分作为模型名
            model_name = model_path.name if isinstance(model_path, Path) else str(model_path)
            sentence_model = SentenceTransformer(model_name)
            print(f"✓ 成功从 HuggingFace 加载模型: {model_name}")
            
            # 打印embedding模型参数
            print("=" * 80)
            print("Embedding Model Configuration (for evaluation):")
            print(f"  Model Name: {model_name}")
            if hasattr(sentence_model, 'get_sentence_embedding_dimension'):
                print(f"  Embedding Dimensions: {sentence_model.get_sentence_embedding_dimension()}")
            print("=" * 80)
        except Exception as e2:
            print(f"警告: 无法加载 SentenceTransformer 模型: {e2}")
            sentence_model = None
else:
    # 从 HuggingFace 下载或使用模型名称
    model_name = str(model_path)
    print(f"从 HuggingFace 加载模型: {model_name}")
    try:
        sentence_model = SentenceTransformer(model_name)
        print(f"✓ 成功从 HuggingFace 加载模型")
        
        # 打印embedding模型参数
        print("=" * 80)
        print("Embedding Model Configuration (for evaluation):")
        print(f"  Model Name: {model_name}")
        if hasattr(sentence_model, 'get_sentence_embedding_dimension'):
            print(f"  Embedding Dimensions: {sentence_model.get_sentence_embedding_dimension()}")
        print("=" * 80)
    except Exception as e:
        print(f"警告: 无法加载 SentenceTransformer 模型: {e}")
        sentence_model = None


def simple_tokenize(text):
    """Simple tokenization function."""
    # Convert to string if not already
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores for prediction against reference."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BLEU scores with different n-gram settings."""
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]

    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1

    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            score = 0.0
        scores[f"bleu{n}"] = score

    return scores


def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore for semantic similarity."""
    try:
        P, R, F1 = bert_score([prediction], [reference], lang="en", verbose=False)
        return {"bert_precision": P.item(), "bert_recall": R.item(), "bert_f1": F1.item()}
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}


def calculate_meteor_score(prediction: str, reference: str) -> float:
    """Calculate METEOR score for the prediction."""
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0


def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    """Calculate sentence embedding similarity using SentenceBERT."""
    if sentence_model is None:
        return 0.0
    try:
        # Encode sentences
        embedding1 = sentence_model.encode([prediction], convert_to_tensor=True)
        embedding2 = sentence_model.encode([reference], convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for a prediction."""
    # Handle empty or None values
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0,
        }

    # Convert to strings if they're not already
    prediction = str(prediction).strip()
    reference = str(reference).strip()

    # Calculate exact match
    exact_match = int(prediction.lower() == reference.lower())

    # Calculate token-based F1 score
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate all scores
    bleu_scores = calculate_bleu_scores(prediction, reference)

    # Combine all metrics
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **bleu_scores,
    }

    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]], all_categories: List[int]
) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """Calculate aggregate statistics for all metrics, split by category."""
    if not all_metrics:
        return {}

    # Initialize aggregates for overall and per-category metrics
    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))

    # Collect all values for each metric, both overall and per category
    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)

    # Calculate statistics for overall metrics
    results = {"overall": {}}

    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    # Calculate statistics for each category
    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:  # Only calculate if we have values for this category
                results[f"category_{category}"][metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

    return results
