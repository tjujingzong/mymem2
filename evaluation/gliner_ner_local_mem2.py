import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gliner import GLiNER


def load_memories_from_index(index_dir: str) -> List[Dict[str, Any]]:
    pkl_path = os.path.join(index_dir, "mem0_user_indices.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"未找到索引文件: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    docstores = data.get("docstores", {}) if isinstance(data, dict) else {}
    if not isinstance(docstores, dict):
        raise ValueError("pkl 文件格式异常：docstores 不是 dict")

    memories: List[Dict[str, Any]] = []
    for user_id, store in docstores.items():
        if not isinstance(store, dict):
            continue
        for doc_id, payload in store.items():
            if not isinstance(payload, dict):
                continue

            text = payload.get("data") or payload.get("text") or payload.get("memory")
            if not text or not isinstance(text, str):
                continue

            memories.append(
                {
                    "user_id": payload.get("user_id", user_id),
                    "doc_id": doc_id,
                    "text": text,
                    "raw": payload,
                }
            )

    return memories


def _maybe_override_encoder(model_dir: str, encoder_model_path: Optional[str]) -> None:
    if not encoder_model_path:
        return
    config_path = os.path.join(model_dir, "gliner_config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("model_name") == encoder_model_path:
        return

    config["model_name"] = encoder_model_path
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def run_ner(
    memories: Iterable[Dict[str, Any]],
    model_name_or_path: str,
    labels: List[str],
    threshold: float,
    limit: int,
    local_files_only: bool,
    encoder_model_path: Optional[str],
) -> List[Dict[str, Any]]:
    _maybe_override_encoder(model_name_or_path, encoder_model_path)
    model = GLiNER.from_pretrained(model_name_or_path, local_files_only=local_files_only)

    results: List[Dict[str, Any]] = []
    for i, mem in enumerate(memories):
        if limit > 0 and i >= limit:
            break

        text = mem["text"]
        entities = model.predict_entities(text, labels, threshold=threshold)

        results.append(
            {
                "user_id": mem["user_id"],
                "doc_id": mem["doc_id"],
                "text": text,
                "entities": entities,
            }
        )

    return results


def load_queries_from_dataset(query_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"未找到 query 文件: {query_file}")

    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("query 数据格式异常：期望 JSON 数组")

    queries: List[Dict[str, Any]] = []
    qid = 0
    for conv_idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        qa_list = item.get("qa", [])
        if not isinstance(qa_list, list):
            continue

        for qa_idx, qa in enumerate(qa_list):
            if not isinstance(qa, dict):
                continue
            text = qa.get("question") or qa.get("query") or qa.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            qid += 1
            queries.append(
                {
                    "id": f"conv{conv_idx}_qa{qa_idx}_{qid}",
                    "text": text.strip(),
                    "raw": qa,
                }
            )

    return queries


def run_ner_on_queries(
    queries: Iterable[Dict[str, Any]],
    model_name_or_path: str,
    labels: List[str],
    threshold: float,
    limit: int,
    local_files_only: bool,
    encoder_model_path: Optional[str],
) -> List[Dict[str, Any]]:
    _maybe_override_encoder(model_name_or_path, encoder_model_path)
    model = GLiNER.from_pretrained(model_name_or_path, local_files_only=local_files_only)

    results: List[Dict[str, Any]] = []
    for i, q in enumerate(queries):
        if limit > 0 and i >= limit:
            break

        text = q["text"]
        entities = model.predict_entities(text, labels, threshold=threshold)

        results.append(
            {
                "id": q["id"],
                "text": text,
                "entities": entities,
            }
        )

    return results


def calc_avg_entity_count(records: Iterable[Dict[str, Any]]) -> Tuple[float, int]:
    records_list = list(records)
    if not records_list:
        return 0.0, 0

    total_entities = 0
    for r in records_list:
        ents = r.get("entities", [])
        total_entities += len(ents) if isinstance(ents, list) else 0

    count = len(records_list)
    return total_entities / count, count


def main() -> None:
    INDEX_DIR = "/root/ljz/mymem2/evaluation/local_mem2/index"
    MODEL_PATH = "/root/ljz/mymem2/models/gliner_small-v2.1"
    LOCAL_ONLY = True
    LABELS = ["person", "organization", "location", "date", "time", "event"]
    ENCODER_MODEL_PATH = "/root/ljz/mymem2/models/deberta-v3-small"
    HF_HOME = "/root/ljz/mymem2/models/hf_cache"
    OFFLINE = True
    THRESHOLD = 0.5
    MEMORY_LIMIT = 0  # 0 表示全部
    QUERY_FILE = "/root/ljz/mymem2/evaluation/dataset/locomo10.json"
    QUERY_LIMIT = 0  # 0 表示全部
    OUTPUT_FILE = "/root/ljz/mymem2/evaluation/results/gliner_mem2_ner.json"

    if OFFLINE:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", HF_HOME)

    memories = load_memories_from_index(INDEX_DIR)
    print(f"[INFO] 从 {INDEX_DIR} 读取到 {len(memories)} 条 memory")

    results = run_ner(
        memories=memories,
        model_name_or_path=MODEL_PATH,
        labels=LABELS,
        threshold=THRESHOLD,
        limit=MEMORY_LIMIT,
        local_files_only=LOCAL_ONLY,
        encoder_model_path=ENCODER_MODEL_PATH,
    )

    mem_avg, mem_count = calc_avg_entity_count(results)
    print(f"[INFO] 完成 NER，共处理 {len(results)} 条 memory")
    print(f"[STATS] memory 平均实体数量: {mem_avg:.4f} (samples={mem_count})")

    queries = load_queries_from_dataset(QUERY_FILE)
    print(f"[INFO] 从 {QUERY_FILE} 读取到 {len(queries)} 条 query")

    query_results = run_ner_on_queries(
        queries=queries,
        model_name_or_path=MODEL_PATH,
        labels=LABELS,
        threshold=THRESHOLD,
        limit=QUERY_LIMIT,
        local_files_only=LOCAL_ONLY,
        encoder_model_path=ENCODER_MODEL_PATH,
    )
    query_avg, query_count = calc_avg_entity_count(query_results)
    print(f"[INFO] 完成 query NER，共处理 {len(query_results)} 条 query")
    print(f"[STATS] query 平均实体数量: {query_avg:.4f} (samples={query_count})")

    for r in results[:10]:
        print("=" * 80)
        print(f"user_id: {r['user_id']} | doc_id: {r['doc_id']}")
        print(f"text: {r['text']}")
        print("entities:")
        print(json.dumps(r["entities"], ensure_ascii=False, indent=2))

    output_payload: Dict[str, Any] = {
        "memory": {
            "results": results,
            "avg_entity_count": mem_avg,
            "sample_count": mem_count,
        },
        "query": {
            "results": query_results,
            "avg_entity_count": query_avg,
            "sample_count": query_count,
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 结果已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
