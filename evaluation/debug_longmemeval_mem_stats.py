import json
import os
import pprint
import sys
from collections import Counter, defaultdict


def setup_paths():
    """
    确保可以优先 import 当前仓库里的本地 mem0 / evaluation 代码。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(current_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return current_dir, repo_root


def build_local_memory():
    """
    复用 evaluation/src/memzero/add.py 里本地 Memory 的构建逻辑，
    这样可以自动读取 MEM0_VECTOR_PATH=/root/ljz/mymem2/evaluation/local_longmemeval/10。
    """
    from evaluation.src.memzero.add import _build_local_memory  # type: ignore

    # 确保走本地模式
    os.environ.setdefault("MEM0_LOCAL_MODE", "1")

    mem = _build_local_memory()
    return mem


def _iter_all_docstore_items(vs):
    """
    兼容两种 FAISS 存储模式：
    1) 全局 docstore: vs.docstore
    2) 分 user 索引: vs.user_docstores
    统一产出 (doc_id, payload) 迭代器。
    """
    docstore = getattr(vs, "docstore", None)
    if isinstance(docstore, dict) and docstore:
        for doc_id, payload in docstore.items():
            yield doc_id, payload

    user_docstores = getattr(vs, "user_docstores", None)
    if isinstance(user_docstores, dict) and user_docstores:
        for _, uds in user_docstores.items():
            if not isinstance(uds, dict):
                continue
            for doc_id, payload in uds.items():
                yield doc_id, payload


def analyze_vector_store(mem, print_raw: bool = False, raw_limit: int = 20):
    """
    直接读 FAISS 的 docstore，看每个 user_id 存了多少条向量级记忆。
    可选打印向量库中的原始文本内容。
    """
    vs = mem.vector_store
    # 兼容全局 docstore 与分 user docstore
    all_items = list(_iter_all_docstore_items(vs))
    print(f"[向量库统计] 总向量数: {len(all_items)}")

    user_counter = Counter()
    for _, payload in all_items:
        if not isinstance(payload, dict):
            continue
        user_id = payload.get("user_id")
        if user_id:
            user_counter[user_id] += 1

    print(f"[向量库统计] 含 user_id 的向量条数: {sum(user_counter.values())}")
    print(f"[向量库统计] 不同 user_id 数量: {len(user_counter)}")

    # 打印前若干个 user 的记忆数
    print("\n[向量库统计] 每个 user_id 的记忆条数（前 20 个）：")
    for uid, cnt in list(user_counter.items())[:20]:
        print(f"  {uid}: {cnt}")

    if print_raw:
        print(f"\n[向量库完整payload] 最多打印 {raw_limit} 条：")
        shown = 0
        for doc_id, payload in all_items:
            user_id = payload.get("user_id") if isinstance(payload, dict) else None
            print(f"\n- doc_id={doc_id}, user_id={user_id}, payload_type={type(payload).__name__}")

            try:
                if isinstance(payload, dict):
                    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
                else:
                    print(pprint.pformat(payload, width=120, compact=False))
            except Exception:
                print(str(payload))

            shown += 1
            if shown >= raw_limit:
                break

        if shown == 0:
            print("  当前向量库没有可打印的 payload。")

    return user_counter

def analyze_search_topk(mem, dataset_path: str, top_k: int = 10):
    """
    按 longmemeval_as_locomo_10.json 的结构，模拟一次搜索：
    - 对每个会话 idx 和每个问题，构造 speaker_a_user_id / speaker_b_user_id
    - 调用 mem.search(query, user_id=..., filters={\"user_id\": ...}, limit=top_k)
    - 统计每个问题实际返回的记忆条数分布
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n[检索统计] 加载数据集: {dataset_path}, 会话数: {len(data)}")

    count_dist_a = Counter()
    count_dist_b = Counter()
    detail_samples = defaultdict(list)  # 按返回条数收集少量样例，便于人工查看

    total_q = 0

    for idx, item in enumerate(data):
        conversation = item.get("conversation", {})
        qa_list = item.get("qa", [])

        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")
        if not speaker_a or not speaker_b:
            continue

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        for qa in qa_list:
            question = qa.get("question", "")
            if not question:
                continue

            total_q += 1

            # 与 MemorySearch 中本地模式 search_memory 的调用保持一致
            filters_a = {"user_id": speaker_a_user_id}
            filters_b = {"user_id": speaker_b_user_id}

            try:
                res_a = mem.search(
                    question,
                    user_id=speaker_a_user_id,
                    filters=filters_a,
                    limit=top_k,
                )
                res_b = mem.search(
                    question,
                    user_id=speaker_b_user_id,
                    filters=filters_b,
                    limit=top_k,
                )
            except Exception as e:
                print(f"[警告] 检索失败 conv_idx={idx}: {e}")
                continue

            # 兼容 search 返回格式（dict/list）
            mems_a = res_a.get("results", res_a) if isinstance(res_a, dict) else res_a
            mems_b = res_b.get("results", res_b) if isinstance(res_b, dict) else res_b

            na = len(mems_a)
            nb = len(mems_b)

            count_dist_a[na] += 1
            count_dist_b[nb] += 1

            # 只保留少量样例
            if len(detail_samples[na]) < 3:
                detail_samples[na].append(
                    {
                        "conv_idx": idx,
                        "speaker": "A",
                        "user_id": speaker_a_user_id,
                        "question": question,
                        "num_memories": na,
                    }
                )
            if len(detail_samples[nb]) < 3:
                detail_samples[nb].append(
                    {
                        "conv_idx": idx,
                        "speaker": "B",
                        "user_id": speaker_b_user_id,
                        "question": question,
                        "num_memories": nb,
                    }
                )

    print(f"\n[检索统计] 总问题数: {total_q}, top_k 期望值: {top_k}")

    def print_dist(label, dist: Counter):
        if not dist:
            print(f"{label}: 无数据")
            return
        total = sum(dist.values())
        print(f"{label}:")
        for k in sorted(dist.keys()):
            v = dist[k]
            pct = 100.0 * v / total
            print(f"  返回 {k} 条记忆: {v} 个问题 ({pct:.2f}%)")

    print_dist("[speaker_1] 每个问题实际返回记忆条数分布", count_dist_a)
    print_dist("\n[speaker_2] 每个问题实际返回记忆条数分布", count_dist_b)

    print("\n[检索统计] 部分样例（按返回条数分组）：")
    for num, samples in sorted(detail_samples.items()):
        print(f"\n=== 实际返回 {num} 条记忆的样例（最多 3 个）===")
        for s in samples:
            print(
                f"- conv_idx={s['conv_idx']}, speaker={s['speaker']}, user_id={s['user_id']}\n"
                f"  question={s['question'][:120]}..."
            )


def main():
    current_dir, _ = setup_paths()

    # 默认使用 longmemeval_as_locomo_10.json 和本地 longmemeval 向量库
    dataset_path = os.getenv(
        "DATASET_PATH_LONGMEMEVAL",
        os.path.join(current_dir, "dataset", "longmemeval", "longmemeval_as_locomo_10.json"),
    )

    # 允许通过环境变量覆盖向量库路径
    os.environ.setdefault(
        "MEM0_VECTOR_PATH",
        "/root/ljz/mymem2/evaluation/local_longmemeval/10",
    )

    print(f"[配置] DATASET_PATH = {dataset_path}")
    print(f"[配置] MEM0_VECTOR_PATH = {os.getenv('MEM0_VECTOR_PATH')}")

    mem = build_local_memory()

    # 1）统计每个 user_id 在向量库中有多少条记忆，并可选打印原始内容
    print_raw = os.getenv("PRINT_RAW_MEMORIES", "1") == "1"
    raw_limit = int(os.getenv("RAW_MEMORY_PRINT_LIMIT", "20"))
    analyze_vector_store(mem, print_raw=print_raw, raw_limit=raw_limit)

    # 2）模拟搜索，统计每个问题实际返回了多少条记忆
    analyze_search_topk(mem, dataset_path=dataset_path, top_k=10)


if __name__ == "__main__":
    main()


