import argparse
import concurrent.futures
import json
import os
import re
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from metrics.utils import calculate_bleu_scores, calculate_metrics
from mem0.memory.utils import extract_json

load_dotenv()


_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
_MODEL = os.getenv("EVAL_MODEL", os.getenv("MODEL", "deepseek-chat"))

if not _API_KEY:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY（或 OPENAI_API_KEY）以进行 LLM 评估")


def _get_client() -> OpenAI:
    # 每次调用创建 client，避免多线程共享对象带来的不确定性
    return OpenAI(api_key=_API_KEY, base_url=_BASE_URL)


def _safe_int_category(x: Any) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _extract_response_outside_think(text: str) -> str:
    if "<think>" not in text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


def _build_dia_index(conversation_obj: Dict[str, Any]) -> Dict[str, str]:
    """
    将单个 conversation 的所有 session 内容索引为 dia_id -> 'speaker: text'
    数据格式与 evaluation/src/memzero/search.py 中读取方式保持一致。
    """
    dia_index: Dict[str, str] = {}
    if not conversation_obj:
        return dia_index

    session_keys = sorted(
        [k for k in conversation_obj.keys() if k.startswith("session_") and not k.endswith("_date_time")]
    )
    for session_key in session_keys:
        chats = conversation_obj.get(session_key, [])
        for chat in chats:
            dia_id = chat.get("dia_id")
            if not dia_id:
                continue
            speaker = chat.get("speaker", "")
            text = chat.get("text", "")
            dia_index[str(dia_id)] = f"{speaker}: {text}".strip()
    return dia_index


def _load_evidence_snippets(
    dataset_item: Dict[str, Any], dia_ids: List[str]
) -> List[Dict[str, str]]:
    """
    按需加载原始对话片段（证据核验）。
    返回：[{ "dia_id": "...", "text": "speaker: ..." }, ...]
    """
    conversation_obj = (dataset_item.get("conversation") or {})
    dia_index = _build_dia_index(conversation_obj)
    snippets: List[Dict[str, str]] = []
    for did in dia_ids:
        did_s = str(did)
        if did_s in dia_index:
            snippets.append({"dia_id": did_s, "text": dia_index[did_s]})
    return snippets


STAGE1_PROMPT = """
Your task is to label a generated answer as CORRECT or WRONG. Be generous: if it touches the same topic as the gold answer, count it as CORRECT.
For time questions, be generous with equivalent references (relative vs absolute) and formatting differences (e.g., "May 7th" vs "7 May") if they refer to the same time period.

In addition, decide whether you need to verify details using original conversation evidence:
- Set needs_evidence=true only if you are uncertain, the memories are too vague/ambiguous, or precise details (time/number/quote) matter.
- If needs_evidence=true, select up to 6 dia_ids from available_dia_ids that would best verify the answer.

Input JSON:
{input_json}

Return ONLY this JSON (no extra text):
{{
  "label": "CORRECT" | "WRONG",
  "needs_evidence": true | false,
  "needed_dia_ids": ["D1:3"],
  "reason": "one short sentence"
}}
""".strip()


STAGE2_PROMPT = """
You are evaluating the same question again, but now you also have evidence_snippets from the original conversations (retrieved by dia_id pointers).
Use the evidence to make a final generous CORRECT/WRONG judgement consistent with the gold answer.

Input JSON:
{input_json}

Return ONLY this JSON (no extra text):
{{
  "label": "CORRECT" | "WRONG",
  "reason": "one short sentence"
}}
""".strip()


def _chat_json(client: OpenAI, prompt: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = resp.choices[0].message.content or "{}"
    parsed = json.loads(extract_json(content))
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        "total_tokens": resp.usage.total_tokens if resp.usage else 0,
    }
    return parsed, usage


def _llm_dynamic_judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    memories: List[Dict[str, Any]],
    dataset_item: Optional[Dict[str, Any]],
) -> Tuple[int, Dict[str, Any], Dict[str, int]]:
    """
    两阶段评估：
    - Stage1：只给摘要记忆与可用 dia_id 列表，判断是否需要证据核验
    - Stage2：仅在需要时按 dia_id 加载原文片段复核
    返回： (score0or1, meta, token_usage_sum)
    """
    client = _get_client()

    # 可用 dia_id 从 memories 的 metadata/dia_ids 收集（指针化）
    available_dia_ids: List[str] = []
    for m in memories or []:
        for did in (m.get("dia_ids") or []):
            did_s = str(did)
            if did_s not in available_dia_ids:
                available_dia_ids.append(did_s)

    # 仅提供记忆摘要（不直接塞 original_conversation）
    memories_brief = []
    for m in memories or []:
        memories_brief.append(
            {
                "timestamp": m.get("timestamp", ""),
                "memory": m.get("memory", ""),
                "score": m.get("score", None),
                "dia_ids": m.get("dia_ids", []),
            }
        )

    stage1_input = {
        "question": question,
        "gold_answer": gold_answer,
        "generated_answer": generated_answer,
        "memories": memories_brief,
        "available_dia_ids": available_dia_ids,
    }
    stage1_prompt = STAGE1_PROMPT.format(input_json=json.dumps(stage1_input, ensure_ascii=False))
    stage1_out, usage1 = _chat_json(client, stage1_prompt)

    label1 = str(stage1_out.get("label", "")).upper().strip()
    needs_evidence = bool(stage1_out.get("needs_evidence", False))
    needed_dia_ids = stage1_out.get("needed_dia_ids", []) or []
    needed_dia_ids = [str(x) for x in needed_dia_ids][:6]
    needed_dia_ids = [x for x in needed_dia_ids if x in available_dia_ids]

    meta: Dict[str, Any] = {
        "stage1": stage1_out,
        "used_evidence": False,
        "loaded_dia_ids": [],
        "loaded_snippets": [],
    }

    usage_sum = dict(usage1)

    if needs_evidence and needed_dia_ids and dataset_item:
        evidence_snippets = _load_evidence_snippets(dataset_item, needed_dia_ids)
        if evidence_snippets:
            stage2_input = {
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": generated_answer,
                "evidence_snippets": evidence_snippets,
            }
            stage2_prompt = STAGE2_PROMPT.format(input_json=json.dumps(stage2_input, ensure_ascii=False))
            stage2_out, usage2 = _chat_json(client, stage2_prompt)
            meta["stage2"] = stage2_out
            meta["used_evidence"] = True
            meta["loaded_dia_ids"] = [x["dia_id"] for x in evidence_snippets]
            meta["loaded_snippets"] = evidence_snippets
            # 覆盖最终 label
            label1 = str(stage2_out.get("label", label1)).upper().strip()
            usage_sum = {
                "prompt_tokens": usage_sum.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0),
                "completion_tokens": usage_sum.get("completion_tokens", 0) + usage2.get("completion_tokens", 0),
                "total_tokens": usage_sum.get("total_tokens", 0) + usage2.get("total_tokens", 0),
            }

    score = 1 if label1 == "CORRECT" else 0
    return score, meta, usage_sum


def process_item(item_data, dataset_by_idx, pbar, pbar_lock):
    k, v = item_data
    local_results = defaultdict(list)

    # k 可能是 str 或 int
    conversation_idx = None
    try:
        conversation_idx = int(k)
    except Exception:
        conversation_idx = None

    dataset_item = dataset_by_idx.get(conversation_idx) if conversation_idx is not None else None

    for item in v:
        category = _safe_int_category(item.get("category"))
        if category == 5:
            continue

        question = str(item.get("question", ""))
        gt_answer = str(item.get("answer", ""))
        pred_answer_raw = str(item.get("response", ""))
        pred_answer = _extract_response_outside_think(pred_answer_raw)

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)

        # 这里把两位 speaker 的 memories 合并，便于 LLM 做“是否需要细节”的判断
        merged_memories: List[Dict[str, Any]] = []
        merged_memories.extend(item.get("speaker_1_memories", []) or [])
        merged_memories.extend(item.get("speaker_2_memories", []) or [])

        llm_score, llm_meta, llm_token_usage = _llm_dynamic_judge(
            question=question,
            gold_answer=gt_answer,
            generated_answer=pred_answer,
            memories=merged_memories,
            dataset_item=dataset_item,
        )

        result_item: Dict[str, Any] = {
            "question": question,
            "answer": gt_answer,
            "response": pred_answer_raw,
            "category": str(item.get("category", "")),
            "bleu_score": bleu_scores["bleu1"],
            "f1_score": metrics["f1"],
            "llm_score": llm_score,
            # 兼容 generate_scores.py：保留 llm_judge_token_usage 的结构
            "llm_judge_token_usage": llm_token_usage,
            # 新增：动态评估元信息（不会影响旧聚合逻辑）
            "llm_dynamic_eval": llm_meta,
        }

        # 保留时间信息与 token 信息（与旧 evals.py 一致）
        for key in ("response_time", "speaker_1_memory_time", "speaker_2_memory_time", "token_usage"):
            if key in item:
                result_item[key] = item[key]

        local_results[k].append(result_item)

        with pbar_lock:
            pbar.update(1)

    return local_results


def main():
    print("Starting LLM-based dynamic evaluation (pointer-aware)...")
    parser = argparse.ArgumentParser(description="LLM-based dynamic evaluation (pointer-aware)")
    parser.add_argument("--input_file", type=str, required=True, help="Search results JSON (mem0_results_*.json)")
    parser.add_argument("--output_file", type=str, required=True, help="Output metrics JSON (compatible with generate_scores.py)")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=os.getenv("EVAL_DATASET_FILE", "dataset/locomo10.json"),
        help="Original dataset file to load evidence snippets (default: dataset/locomo10.json or env EVAL_DATASET_FILE)",
    )
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of worker threads")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 读取 dataset，用于 dia_id -> 原文回溯
    dataset_by_idx: Dict[int, Dict[str, Any]] = {}
    try:
        dataset_path = args.dataset_file
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_path)
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for idx, item in enumerate(dataset):
            dataset_by_idx[idx] = item
        print(f"已加载 dataset: {dataset_path} (conversations={len(dataset_by_idx)})", flush=True)
    except Exception as e:
        print(f"警告：加载 dataset 失败，将禁用按 dia_id 回溯原文。错误: {e}", flush=True)
        dataset_by_idx = {}

    total_items = sum(len([i for i in v if _safe_int_category(i.get("category")) != 5]) for v in data.values())
    print(f"待评估问答数: {total_items}", flush=True)

    results = defaultdict(list)
    results_lock = threading.Lock()
    pbar_lock = threading.Lock()

    with tqdm(total=total_items, desc="Evaluating (LLM dynamic)", unit="qa") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_item, item_data, dataset_by_idx, pbar, pbar_lock) for item_data in data.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                local_results = future.result()
                with results_lock:
                    for k, items in local_results.items():
                        results[k].extend(items)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {args.output_file}", flush=True)


if __name__ == "__main__":
    main()


