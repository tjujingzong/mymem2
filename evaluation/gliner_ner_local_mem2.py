import argparse
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional

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


def main() -> None:
    parser = argparse.ArgumentParser(description="对 local_mem2/index 的 memory 做 GLiNER NER")
    parser.add_argument(
        "--index-dir",
        default="/root/ljz/mymem2/evaluation/local_mem2/index",
        help="包含 mem0_user_indices.pkl 的目录",
    )
    parser.add_argument(
        "--model",
        default="/root/ljz/mymem2/models/gliner_small-v2.1",
        help="GLiNER 模型名或本地模型目录",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="仅从本地加载模型，不访问网络（离线场景建议开启）",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["person", "organization", "location", "date", "time", "event"],
        help="要抽取的实体类型",
    )
    parser.add_argument(
        "--encoder-model",
        default="/root/ljz/mymem2/models/deberta-v3-small",
        help="GLiNER 内部编码器（deberta-v3-small）本地路径",
    )
    parser.add_argument(
        "--hf-home",
        default="/root/ljz/mymem2/models/hf_cache",
        help="HuggingFace 缓存目录（离线加载用）",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="启用离线模式（会设置 TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE）",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="实体置信度阈值")
    parser.add_argument("--limit", type=int, default=50, help="最多处理多少条 memory，-1 表示全部")
    parser.add_argument(
        "--output",
        default="",
        help="可选，输出 JSON 文件路径；不传则只打印到终端",
    )

    args = parser.parse_args()

    if args.offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", args.hf_home)

    memories = load_memories_from_index(args.index_dir)
    print(f"[INFO] 从 {args.index_dir} 读取到 {len(memories)} 条 memory")

    limit = args.limit
    if limit == -1:
        limit = 0

    results = run_ner(
        memories=memories,
        model_name_or_path=args.model,
        labels=args.labels,
        threshold=args.threshold,
        limit=limit,
        local_files_only=args.local_only,
        encoder_model_path=args.encoder_model,
    )

    print(f"[INFO] 完成 NER，共处理 {len(results)} 条 memory")
    for r in results[:10]:
        print("=" * 80)
        print(f"user_id: {r['user_id']} | doc_id: {r['doc_id']}")
        print(f"text: {r['text']}")
        print("entities:")
        print(json.dumps(r["entities"], ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 结果已写入: {args.output}")


if __name__ == "__main__":
    main()
