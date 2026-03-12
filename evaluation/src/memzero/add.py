import json
import os
import re
import threading
import time
import sqlite3
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import Memory, MemoryClient
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.graphs.configs import GraphStoreConfig, Neo4jConfig

load_dotenv()


# Update custom instructions
# 测试：删除自定义 prompt，使用 mem0 默认的 prompt
# custom_instructions = """
# Generate personal memories that follow these guidelines:
#
# 1. Each memory should be self-contained with complete context, including:
#    - The person's name, do not use "user" while creating memories
#    - Personal details (career aspirations, hobbies, life circumstances)
#    - Emotional states and reactions
#    - Ongoing journeys or future plans
#    - Specific dates when events occurred
#
# 2. Include meaningful personal narratives focusing on:
#    - Identity and self-acceptance journeys
#    - Family planning and parenting
#    - Creative outlets and hobbies
#    - Mental health and self-care activities
#    - Career aspirations and education goals
#    - Important life events and milestones
#
# 3. Make each memory rich with specific details rather than general statements
#    - Include timeframes (exact dates when possible)
#    - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
#    - Include emotional context and personal growth elements
#
# 4. Extract memories only from user messages, not incorporating assistant responses
#
# 5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
# """


def _use_local_memory() -> bool:
    return str(os.getenv("MEM0_LOCAL_MODE", "0")).lower() in ("1", "true", "yes")


def _build_local_memory():
    """
    Build a local Memory instance that keeps vector store + embedding fully本地，
    while继续使用远程 LLM（DeepSeek/OpenAI）做抽取。
    """
    vector_provider = os.getenv("MEM0_VECTOR_PROVIDER", "faiss")
    vector_path = os.getenv("MEM0_VECTOR_PATH", "/root/ljz/mymem/evaluation/local_mem0/faiss")
    vector_collection = os.getenv("MEM0_VECTOR_COLLECTION", "mem0")
    # 对齐默认 HuggingFace 模型维度（multi-qa-MiniLM-L6-cos-v1 = 384）
    vector_dim = int(os.getenv("MEM0_VECTOR_DIM", os.getenv("MEM0_EMBED_DIM", "384")))
    # 支持通过环境变量切换相似度度量：MEM0_VECTOR_DISTANCE=euclidean|cosine|inner_product
    vector_distance = os.getenv("MEM0_VECTOR_DISTANCE", "euclidean")

    embed_provider = os.getenv("MEM0_EMBED_PROVIDER", "huggingface")
    embed_model = os.getenv("MEM0_EMBED_MODEL", "multi-qa-MiniLM-L6-cos-v1")
    # 去除可能的引号（.env 文件中可能包含引号）
    if embed_model:
        embed_model = embed_model.strip()
        if embed_model.startswith('"') and embed_model.endswith('"'):
            embed_model = embed_model[1:-1]
        elif embed_model.startswith("'") and embed_model.endswith("'"):
            embed_model = embed_model[1:-1]
        embed_model = embed_model.strip()

    llm_provider = os.getenv("MEM0_LLM_PROVIDER", "openai")
    llm_model = os.getenv("MEM0_LLM_MODEL", os.getenv("MODEL", "deepseek-chat"))
    llm_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")

    # Neo4j 图存储配置（本地模式可选开启）
    enable_graph_store = str(os.getenv("ENABLE_GRAPH_STORE", "0")).lower() in ("1", "true", "yes")
    neo4j_url = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE")

    graph_store_cfg = None
    if enable_graph_store:
        graph_store_cfg = GraphStoreConfig(
            provider="neo4j",
            config=Neo4jConfig(
                url=neo4j_url,
                username=neo4j_username,
                password=neo4j_password,
                database=neo4j_database,
            ),
            # 图谱侧可能也需要 LLM 用于实体抽取/查询
            llm=LlmConfig(
                provider=llm_provider,
                config={
                    "model": llm_model,
                    "api_key": llm_api_key,
                    "openai_base_url": llm_base_url,
                },
            ),
        )

    memory_cfg = MemoryConfig(
        vector_store=VectorStoreConfig(
            provider=vector_provider,
            config={
                "path": vector_path,
                "collection_name": vector_collection,
                "distance_strategy": vector_distance,
                "embedding_model_dims": vector_dim,
            },
        ),
        embedder=EmbedderConfig(
            provider=embed_provider,
            config={
                "model": embed_model,
                "embedding_dims": vector_dim,
            },
        ),
        llm=LlmConfig(
            provider=llm_provider,
            config={
                "model": llm_model,
                "api_key": llm_api_key,
                "openai_base_url": llm_base_url,
            },
        ),
        graph_store=graph_store_cfg or GraphStoreConfig(),
    )
    
    # 打印embedding模型参数
    print("=" * 80)
    print("Embedding Model Configuration:")
    print(f"  Provider: {embed_provider}")
    print(f"  Model: {embed_model}")
    print(f"  Embedding Dimensions: {vector_dim}")
    print(f"  Vector Store Provider: {vector_provider}")
    print(f"  Vector Store Path: {vector_path}")
    print(f"  Vector Store Collection: {vector_collection}")
    print(f"  Distance Strategy: {vector_distance}")
    print("=" * 80)
    
    return Memory(config=memory_cfg)


class MemoryADD:
    def __init__(self, data_path=None, batch_size=2, is_graph=False, use_sentence_mode=False, use_simple_mode=False):
        self.use_local = _use_local_memory()
        if self.use_local:
            self.mem0_client = _build_local_memory()
        else:
            # 使用远程Mem0 API时，打印提示信息
            print("=" * 80)
            print("Embedding Model Configuration:")
            print("  Using Remote Mem0 API (embedding model configured on server)")
            print(f"  API Key: {'*' * 20 if os.getenv('MEM0_API_KEY') else 'Not set'}")
            print(f"  Organization ID: {os.getenv('MEM0_ORGANIZATION_ID', 'Not set')}")
            print(f"  Project ID: {os.getenv('MEM0_PROJECT_ID', 'Not set')}")
            print("=" * 80)
            
            self.mem0_client = MemoryClient(
                api_key=os.getenv("MEM0_API_KEY"),
                org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                project_id=os.getenv("MEM0_PROJECT_ID"),
            )
            # 测试：删除自定义 prompt，使用 mem0 默认的 prompt
            # self.mem0_client.update_project(custom_instructions=custom_instructions)

        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        self.use_sentence_mode = use_sentence_mode
        self.use_simple_mode = use_simple_mode

        # 统计信息（token / 调用耗时）
        self.stats_lock = threading.Lock()
        self.stats = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "add_calls": 0,
            "total_elapsed_seconds": 0.0,
            # 旧口径：从 add 返回值中提取 usage（很多 provider 下会是 0）
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            # 新口径（与 search 一致）：从底层 chat.completions 原始响应 usage 统计
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
            # 严格按返回事件名统计，不预设字段、不臆造 CRUD 分类
            "crud_counts": {},
        }
        
        if self.use_local:
            self._install_llm_usage_hook()

        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data
    
    def _extract_dia_ids_from_chats(self, chats, start_idx, end_idx):
        """从chats的指定范围内提取dia_id列表"""
        dia_ids = []
        for chat in chats[start_idx:end_idx]:
            if "dia_id" in chat:
                dia_ids.append(chat["dia_id"])
        return dia_ids

    def _split_text_to_sentences(self, text: str):
        if not text:
            return []
        parts = re.split(r"[，,。.!！？?；;：:\n\r]+", str(text))
        sentences = [p.strip() for p in parts if p and p.strip()]
        return sentences

    def _extract_usage_from_raw_response(self, response):
        """从 OpenAI 兼容 raw response 中提取 usage（与 search 口径一致）。"""
        def _to_int(v):
            try:
                return int(v or 0)
            except Exception:
                return 0

        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0

        prompt_tokens = _to_int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = _to_int(getattr(usage, "completion_tokens", 0))
        total_tokens = _to_int(getattr(usage, "total_tokens", 0))
        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens
        return prompt_tokens, completion_tokens, total_tokens

    def _install_llm_usage_hook(self):
        """
        在本地 Memory 模式下，hook 底层 chat.completions.create，
        统计 raw response usage（search 同口径）。
        """
        try:
            llm = getattr(self.mem0_client, "llm", None)
            client = getattr(llm, "client", None)
            chat = getattr(client, "chat", None)
            completions = getattr(chat, "completions", None)
            create_fn = getattr(completions, "create", None)

            if create_fn is None:
                print("[ADD STATS] 未安装 llm usage hook：未找到 chat.completions.create")
                return

            # 避免重复安装
            if getattr(create_fn, "_mem0_usage_hook_installed", False):
                return

            def _wrapped_create(*args, **kwargs):
                response = create_fn(*args, **kwargs)
                p, c, t = self._extract_usage_from_raw_response(response)
                if p or c or t:
                    with self.stats_lock:
                        self.stats["llm_prompt_tokens"] += p
                        self.stats["llm_completion_tokens"] += c
                        self.stats["llm_total_tokens"] += t
                return response

            setattr(_wrapped_create, "_mem0_usage_hook_installed", True)
            completions.create = _wrapped_create
            print("[ADD STATS] 已安装 llm usage hook（raw response usage）")
        except Exception as e:
            print(f"[ADD STATS] 安装 llm usage hook 失败: {e}")

    def _collect_history_action_counts(self, history_db_path):
        """从 mem0 history.db 聚合真实动作计数（ADD/UPDATE/DELETE）。"""
        counts = {}
        if not history_db_path or not os.path.exists(history_db_path):
            return counts

        conn = None
        try:
            conn = sqlite3.connect(history_db_path)
            cur = conn.cursor()
            cur.execute("SELECT action, COUNT(*) FROM history GROUP BY action")
            rows = cur.fetchall() or []
            for action, cnt in rows:
                if action is None:
                    continue
                key = str(action).strip().upper()
                if not key:
                    continue
                counts[key] = int(cnt or 0)
        except Exception as e:
            print(f"[ADD STATS] 读取 history.db 失败: {e}")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        return counts

    def _extract_token_usage(self, result):
        """尽量从不同返回结构中提取 token 使用量，避免递归重复累计。"""
        def _to_int(v):
            try:
                return int(v or 0)
            except Exception:
                return 0

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        # 优先级1：顶层 usage（最常见且最可靠）
        if isinstance(result, dict):
            usage = result.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = _to_int(usage.get("prompt_tokens"))
                completion_tokens = _to_int(usage.get("completion_tokens"))
                total_tokens = _to_int(usage.get("total_tokens"))

            # 优先级2：顶层扁平字段（部分 provider 兼容字段）
            if prompt_tokens == 0:
                prompt_tokens = _to_int(result.get("prompt_tokens"))
            if completion_tokens == 0:
                completion_tokens = _to_int(result.get("completion_tokens"))
            if total_tokens == 0:
                total_tokens = _to_int(result.get("total_tokens"))

            # 优先级3：data[i].usage（避免全量递归造成重复统计）
            data = result.get("data")
            if isinstance(data, list) and (prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    u = item.get("usage")
                    if isinstance(u, dict):
                        prompt_tokens += _to_int(u.get("prompt_tokens"))
                        completion_tokens += _to_int(u.get("completion_tokens"))
                        total_tokens += _to_int(u.get("total_tokens"))

        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens

        return prompt_tokens, completion_tokens, total_tokens

    def _record_add_stats(self, elapsed_seconds, result):
        prompt_tokens, completion_tokens, total_tokens = self._extract_token_usage(result)
        with self.stats_lock:
            self.stats["add_calls"] += 1
            self.stats["total_elapsed_seconds"] += float(elapsed_seconds)
            self.stats["prompt_tokens"] += prompt_tokens
            self.stats["completion_tokens"] += completion_tokens
            self.stats["total_tokens"] += total_tokens

            # 事件统计：严格按返回中的 event 原值聚合，不做 CRUD 语义映射
            def _walk_events(node):
                if isinstance(node, dict):
                    event = node.get("event")
                    if event is not None:
                        event_key = str(event).strip().upper()
                        if event_key:
                            self.stats["crud_counts"][event_key] = self.stats["crud_counts"].get(event_key, 0) + 1
                    for v in node.values():
                        _walk_events(v)
                elif isinstance(node, list):
                    for item in node:
                        _walk_events(item)

            _walk_events(result)

    def add_memory(self, user_id, message, metadata, dia_ids=None, retries=3):
        # 最简单模式：不写入 dia_ids（不保存对话ID）
        if (not self.use_simple_mode) and dia_ids:
            if metadata is None:
                metadata = {}
            metadata["dia_ids"] = dia_ids
        
        for attempt in range(retries):
            try:
                start_time = time.perf_counter()
                if self.use_local:
                    result = self.mem0_client.add(message, user_id=user_id, metadata=metadata, infer=(not self.use_sentence_mode))
                else:
                    result = self.mem0_client.add(
                        message,
                        user_id=user_id,
                        version="v2",
                        metadata=metadata,
                        enable_graph=self.is_graph,
                        infer=(not self.use_sentence_mode),
                    )
                elapsed = time.perf_counter() - start_time
                self._record_add_stats(elapsed, result)
                return result
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc, chats=None):
        # 进度条：使用 `leave=False` 避免在异常时残留大量进度条输出占满控制台
        for batch_idx, i in enumerate(
            tqdm(
                range(0, len(messages), self.batch_size),
                desc=desc,
                leave=False,
                mininterval=1.0,  # 降低刷新频率，减少控制台刷屏
            )
        ):
            batch_messages = messages[i : i + self.batch_size]
            
            # 提取对应的dia_ids（messages和chats是一一对应的）
            dia_ids = None
            if chats:
                end_idx = min(i + self.batch_size, len(chats))
                dia_ids = self._extract_dia_ids_from_chats(chats, i, end_idx)

            # 短句模式：把每个短句当作一条 memory 写入向量库（infer=False 已由 add_memory 控制）
            if self.use_sentence_mode:
                sentence_idx = i * 100000
                for message_dict in batch_messages:
                    if not isinstance(message_dict, dict):
                        continue
                    content = message_dict.get("content", "")
                    role = message_dict.get("role", "")

                    # content 形如 "speaker: text"，按你的要求只切 chat['text']：取冒号后的部分
                    text_part = content
                    if isinstance(content, str) and ":" in content:
                        text_part = content.split(":", 1)[1]
                    sentences = self._split_text_to_sentences(text_part)

                    for sent in sentences:
                        per_meta = {"timestamp": timestamp, "role": role, "sentence_mode": True, "sentence_idx": sentence_idx}
                        if dia_ids and (not self.use_simple_mode):
                            per_meta["dia_ids"] = dia_ids
                        self.add_memory(speaker, sent, metadata=per_meta, dia_ids=dia_ids)
                        sentence_idx += 1
                result = None
            else:
                result = self.add_memory(
                    speaker, 
                    batch_messages, 
                    metadata={"timestamp": timestamp},
                    dia_ids=None if self.use_simple_mode else dia_ids
                )

            # 每 10 个 batch 打印一次 LLM / mem0 返回结果，便于 debug，
            # 同时避免把控制台刷爆。
            if result is not None and (batch_idx + 1) % 10 == 0:
                try:
                    print(f"\n[LLM OUTPUT][speaker={speaker}] batch #{batch_idx + 1}")
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                except TypeError:
                    # 有些后端可能返回非 JSON 序列化对象，这里做一个兜底转换
                    print(f"\n[LLM OUTPUT][speaker={speaker}] batch #{batch_idx + 1}")
                    print(str(result))

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # delete all memories for the two users
        self.mem0_client.delete_all(user_id=speaker_a_user_id)
        self.mem0_client.delete_all(user_id=speaker_b_user_id)
        # delete_all 不一定返回逐条事件，这里单独记为 API 层面的 DELETE_ALL 次数
        with self.stats_lock:
            self.stats["crud_counts"]["DELETE_ALL"] = self.stats["crud_counts"].get("DELETE_ALL", 0) + 2

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # add memories for the two users on different threads
            # 传递chats以便提取dia_ids
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A", chats),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B", chats),
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        print("Messages added successfully")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]

            for future in futures:
                future.result()

        # add 阶段全局统计输出 + 写入文件
        with self.stats_lock:
            self.stats["finished_at"] = datetime.now().isoformat(timespec="seconds")
            calls = self.stats["add_calls"]
            elapsed = self.stats["total_elapsed_seconds"]
            llm_prompt_tokens = self.stats.get("llm_prompt_tokens", 0)
            llm_completion_tokens = self.stats.get("llm_completion_tokens", 0)
            llm_total_tokens = self.stats.get("llm_total_tokens", 0)
            crud_counts = dict(self.stats.get("crud_counts", {}))

        avg_latency = (elapsed / calls) if calls > 0 else 0.0

        # mem0 默认 history.db 路径（与向量库 faiss 无关）
        history_db_path = None
        try:
            history_db_path = getattr(self.mem0_client.config, "history_db_path", None)
        except Exception:
            history_db_path = None

        # 从 history.db 获取真实动作统计（ADD/UPDATE/DELETE）
        history_action_counts = self._collect_history_action_counts(history_db_path)

        llm_avg_prompt_tokens_per_add_call = (llm_prompt_tokens / calls) if calls > 0 else 0.0
        llm_avg_completion_tokens_per_add_call = (llm_completion_tokens / calls) if calls > 0 else 0.0
        llm_avg_total_tokens_per_add_call = (llm_total_tokens / calls) if calls > 0 else 0.0

        stats_output = {
            "started_at": self.stats.get("started_at"),
            "finished_at": self.stats.get("finished_at"),
            "phase": "add",
            "llm_token_usage": {
                "total": {
                    "prompt_tokens": llm_prompt_tokens,
                    "completion_tokens": llm_completion_tokens,
                    "total_tokens": llm_total_tokens,
                },
                "avg_per_add_call": {
                    "prompt_tokens": round(llm_avg_prompt_tokens_per_add_call, 6),
                    "completion_tokens": round(llm_avg_completion_tokens_per_add_call, 6),
                    "total_tokens": round(llm_avg_total_tokens_per_add_call, 6),
                },
            },
            "time_usage": {
                "add_calls": calls,
                "total_elapsed_seconds": round(elapsed, 6),
                "avg_elapsed_seconds_per_call": round(avg_latency, 6),
            },
            "crud_counts": crud_counts,
            "history_crud_counts": history_action_counts,
            "history_db": {
                "path": history_db_path,
                "exists": bool(history_db_path and os.path.exists(history_db_path)),
            },
        }

        stats_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results_stats")
        os.makedirs(stats_dir, exist_ok=True)
        stats_file = os.path.join(stats_dir, "add_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_output, f, ensure_ascii=False, indent=2)

        print("=" * 80)
        print("[ADD STATS] Token / Time / Event Summary")
        print(f"  add 调用次数: {calls}")
        print(f"  累计耗时(秒): {elapsed:.3f}")
        print(f"  平均每次耗时(秒): {avg_latency:.3f}")
        print(f"  prompt_tokens(raw llm, total): {llm_prompt_tokens}")
        print(f"  completion_tokens(raw llm, total): {llm_completion_tokens}")
        print(f"  total_tokens(raw llm, total): {llm_total_tokens}")
        print(f"  prompt_tokens(raw llm, avg/add_call): {llm_avg_prompt_tokens_per_add_call:.6f}")
        print(f"  completion_tokens(raw llm, avg/add_call): {llm_avg_completion_tokens_per_add_call:.6f}")
        print(f"  total_tokens(raw llm, avg/add_call): {llm_avg_total_tokens_per_add_call:.6f}")
        print(f"  事件统计(event->count): {crud_counts}")
        print(f"  历史动作统计(history.db): {history_action_counts}")
        print(f"  history.db 路径: {history_db_path}")
        print(f"  history.db 是否存在: {bool(history_db_path and os.path.exists(history_db_path))}")
        print(f"  统计文件: {stats_file}")
        print("=" * 80)
