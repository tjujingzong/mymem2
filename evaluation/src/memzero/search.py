import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import math
import re
import numpy as np

from gliner import GLiNER

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm

from mem0 import Memory, MemoryClient
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.graphs.configs import GraphStoreConfig, Neo4jConfig

load_dotenv()


class MemorySearch:
    def __init__(
        self,
        output_path="results.json",
        top_k=10,
        filter_memories=False,
        is_graph=False,
        data_path=None,
        include_original_conversations=None,
        use_sentence_mode=False,
        use_hybrid_mode=False,
        use_simple_mode=False,
    ):
        self.use_local = str(os.getenv("MEM0_LOCAL_MODE", "0")).lower() in ("1", "true", "yes")
        if self.use_local:
            vector_provider = os.getenv("MEM0_VECTOR_PROVIDER", "faiss")
            vector_path = os.getenv("MEM0_VECTOR_PATH", "/root/ljz/mymem/evaluation/local_mem0/faiss")
            vector_collection = os.getenv("MEM0_VECTOR_COLLECTION", "mem0")
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

            llm_provider = os.getenv("MEM0_LLM_PROVIDER", "deepseek")
            llm_model = os.getenv("MEM0_LLM_MODEL", os.getenv("MODEL", "deepseek-chat"))
            llm_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            llm_base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")

            llm_config = {
                "model": llm_model,
                "api_key": llm_api_key,
            }
            # deepseek provider 需要 deepseek_base_url；openai provider 需要 openai_base_url
            if str(llm_provider).lower() == "deepseek":
                llm_config["deepseek_base_url"] = llm_base_url
            else:
                llm_config["openai_base_url"] = llm_base_url

            # Neo4j 图存储配置（本地模式可选开启，不影响原有非图流程）
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
                    llm=LlmConfig(
                        provider=llm_provider,
                        config=llm_config,
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
                    config=llm_config,
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
            
            self.mem0_client = Memory(config=memory_cfg)
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
        self.top_k = top_k
        # 支持自定义基座（如 DeepSeek），优先用 DEEPSEEK_API_KEY/OPENAI_BASE_URL
        self.openai_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY")),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
        )
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.use_sentence_mode = use_sentence_mode
        self.sentence_stats = {"total_sentences": 0, "num_messages": 0}
        self.is_graph = is_graph
        self.use_hybrid_mode = use_hybrid_mode
        self.use_simple_mode = use_simple_mode

        if self.use_sentence_mode and self.is_graph:
            raise ValueError("短句模式不支持图搜索，请关闭 --is_graph。")
        if self.use_sentence_mode and self.use_hybrid_mode:
            raise ValueError("USE_SENTENCE_MODE 与 USE_HYBRID_MODE 不能同时开启。")
        if include_original_conversations is None:
            # 默认值为 1 (True)，保持旧行为兼容性
            self.include_original_conversations = str(
                os.getenv("MEM0_INCLUDE_ORIGINAL_CONVERSATIONS", "1")
            ).lower() in ("1", "true", "yes")
        else:
            self.include_original_conversations = include_original_conversations

        # 在纯短句模式下，为了省 token 强制不带原文
        if self.use_sentence_mode:
            self.include_original_conversations = False

        # 是否启用两阶段按需加载原文（默认关闭，保持旧行为不变）
        self.qa_two_stage = str(os.getenv("MEM0_QA_TWO_STAGE", "0")).lower() in ("1", "true", "yes")

        # 最简单模式：强制不拼原文、关闭两阶段
        if self.use_simple_mode:
            self.include_original_conversations = False
            self.qa_two_stage = False

        # 图模式下，为了公平评估记忆能力，只使用 memory + graph_memory，不再把原始对话拼进 QA prompt
        if self.is_graph:
            self.include_original_conversations = False
            self.qa_two_stage = False

        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT
        
        # 加载原始数据文件，用于根据dia_id查找对话内容（最简单模式下跳过）
        self.original_data = None
        if (not self.use_simple_mode) and data_path and os.path.exists(data_path):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    self.original_data = json.load(f)
                print(f"已加载原始数据文件: {len(self.original_data)} 个对话")
            except Exception as e:
                print(f"加载原始数据文件失败: {e}")
                self.original_data = None

        # 是否启用“记忆指针化 + NER”的 RRF 融合检索策略（默认关闭保证兼容性）
        # 开启方式：MEM0_NER_RRF_FUSION=1（或 true/yes）
        self.use_ner_rrf_fusion = (
            str(os.getenv("MEM0_NER_RRF_FUSION", "0")).lower() in ("1", "true", "yes")
        )
        if self.use_simple_mode:
            self.use_ner_rrf_fusion = False
        # RRF 中的常数 k，可通过环境变量调整，默认 60
        try:
            self.rrf_k = int(os.getenv("MEM0_RRF_K", "60"))
        except ValueError:
            self.rrf_k = 60

        # NER 模型（本地离线）
        self.ner_model = None
        if self.use_ner_rrf_fusion:
            ner_model_path = os.getenv("MEM0_NER_MODEL_PATH", "/root/ljz/mymem2/models/gliner_small-v2.1")
            ner_encoder_path = os.getenv("MEM0_NER_ENCODER_PATH", "/root/ljz/mymem2/models/deberta-v3-small")
            ner_labels = os.getenv("MEM0_NER_LABELS", "person,organization,location,date,time,event")
            ner_threshold = float(os.getenv("MEM0_NER_THRESHOLD", "0.5"))
            self.ner_labels = [s.strip() for s in ner_labels.split(",") if s.strip()]
            self.ner_threshold = ner_threshold

            # 强制离线 + 本地模型
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

            self._maybe_override_gliner_encoder(ner_model_path, ner_encoder_path)
            self.ner_model = GLiNER.from_pretrained(ner_model_path, local_files_only=True)

    def _rank_sentences_by_query(self, query: str, sentences: list[str]):
        """Rank sentences by cosine similarity using the same embedding model as mem0."""
        if not sentences:
            return []
        try:
            embed_fn = self.mem0_client.embedding_model.embed
        except AttributeError:
            # fallback: overlap
            return sentences

        query_emb = np.array(embed_fn(query, "search"))
        sent_embs = [np.array(embed_fn(s, "search")) for s in sentences]
        # normalize
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        sims = []
        for s, emb in zip(sentences, sent_embs):
            emb_n = emb / (np.linalg.norm(emb) + 1e-8)
            sims.append(float(np.dot(query_emb, emb_n)))
        ranked = [s for _, s in sorted(zip(sims, sentences), key=lambda x: -x[0])]
        return ranked

    def _split_text_to_sentences(self, text: str):
        if not text:
            return []
        parts = re.split(r"[，,。.!！？?；;：:\n\r]+", str(text))
        sentences = [p.strip() for p in parts if p and p.strip()]
        return sentences

    def _get_conversation_by_dia_ids(self, dia_ids, conversation_idx):
        """根据dia_ids和对话索引从原始数据中获取对话内容"""
        if not dia_ids or not self.original_data or conversation_idx >= len(self.original_data):
            return None
        
        conversation = self.original_data[conversation_idx].get("conversation", {})
        if not conversation:
            return None
        
        # 收集所有相关的对话文本（保持顺序）
        conversation_texts = []
        dia_ids_set = set(dia_ids)  # 转换为set以提高查找效率
        
        # 遍历所有session查找匹配的dia_id
        # 按session顺序遍历，保持对话的时序性
        session_keys = sorted([k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")])
        
        for session_key in session_keys:
            chats = conversation.get(session_key, [])
            for chat in chats:
                if "dia_id" in chat and chat["dia_id"] in dia_ids_set:
                    speaker = chat.get("speaker", "")
                    text = chat.get("text", "")
                    conversation_texts.append(f"{speaker}: {text}")
        
        if conversation_texts:
            return "\n".join(conversation_texts)
        return None

    # ========= NER + RRF 融合（指针化原文级别） =========
    def _maybe_override_gliner_encoder(self, model_dir: str, encoder_model_path: str) -> None:
        config_path = os.path.join(model_dir, "gliner_config.json")
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            return

        if config.get("model_name") == encoder_model_path:
            return

        config["model_name"] = encoder_model_path
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _extract_ner_entities(self, text: str) -> set[str]:
        if not text or not self.ner_model:
            return set()
        try:
            entities = self.ner_model.predict_entities(
                text,
                labels=self.ner_labels,
                threshold=self.ner_threshold,
            )
        except Exception:
            return set()

        ent_set = set()
        for ent in entities or []:
            label = str(ent.get("label", "")).strip().lower()
            name = str(ent.get("text", "")).strip().lower()
            if not label or not name:
                continue
            ent_set.add(f"{label}:{name}")
        return ent_set

    def _tokenize_text(self, text: str):
        """简单分词：按非字母数字字符切分，统一转小写。"""
        if not text:
            return []
        return re.findall(r"\w+", str(text).lower())

    def _is_time_like_token(self, token: str) -> bool:
        """粗略判断是否为时间相关 token，用于对时间关键词加权。"""
        if not token:
            return False
        # 包含四位数字（年份等）
        if re.search(r"\d{4}", token):
            return True
        # 英文常见时间词
        time_words_en = {
            "yesterday",
            "today",
            "tomorrow",
            "tonight",
            "morning",
            "afternoon",
            "evening",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        }
        if token in time_words_en:
            return True
        # 中文时间词
        if any(ch in token for ch in ["年", "月", "日", "号", "昨天", "今天", "明天"]):
            return True
        return False

    def _ner_overlap_scores(self, query: str, docs: list[str]) -> list[float]:
        """基于 query 与文档的实体重叠，计算 NER 得分。"""
        if not docs:
            return []

        q_ents = self._extract_ner_entities(query)
        if not q_ents:
            return [0.0 for _ in docs]

        scores = []
        for doc in docs:
            d_ents = self._extract_ner_entities(doc)
            if not d_ents:
                scores.append(0.0)
                continue
            inter = q_ents & d_ents
            scores.append(len(inter) / max(len(q_ents), 1))
        return scores

    def _apply_rrf_fusion(self, query: str, semantic_memories: list[dict]) -> list[dict]:
        """
        对基于 embedding 的排序结果，引入“指针化原文片段”的 NER 检索，
        然后使用 RRF (Reciprocal Rank Fusion) 做多检索器融合。

        - 排序 1：Mem0 的语义检索结果（当前顺序即 rank）
        - 排序 2：在 pointer 回溯得到的 original_conversation / memory 上做 NER 重叠排序
        """
        if not semantic_memories or not query or not self.use_ner_rrf_fusion:
            return semantic_memories

        num = len(semantic_memories)
        # 语义排序：当前顺序视为 rank（从 1 开始）
        semantic_ranks = {idx: idx + 1 for idx in range(num)}

        # 为每条记忆构造用于关键词检索的文本：
        # 优先使用 pointer 回溯得到的原始对话 original_conversation，其次才用 summary memory
        docs = []
        for item in semantic_memories:
            doc_text = item.get("original_conversation") or item.get("memory", "")
            docs.append(str(doc_text))

        ner_scores = self._ner_overlap_scores(query, docs)

        # 按 NER 得分从高到低给出排名（只对得分>0的项设定 rank）
        sorted_by_ner = sorted(
            [(idx, score) for idx, score in enumerate(ner_scores)], key=lambda x: x[1], reverse=True
        )
        ner_ranks = {}
        rank_pos = 1
        for idx, score in sorted_by_ner:
            if score <= 0:
                break
            ner_ranks[idx] = rank_pos
            rank_pos += 1

        k_const = self.rrf_k
        default_rank = num + 1

        fused_scores = {}
        for idx in range(num):
            r_sem = semantic_ranks.get(idx, default_rank)
            r_ner = ner_ranks.get(idx, default_rank)
            fused_scores[idx] = 1.0 / (k_const + r_sem) + 1.0 / (k_const + r_ner)

        # 按 fused score 从大到小排序，并裁剪到 top_k
        sorted_indices = sorted(range(num), key=lambda i: fused_scores[i], reverse=True)
        top_n = min(self.top_k, len(sorted_indices))

        fused_memories = []
        for rank_idx, mem_idx in enumerate(sorted_indices[:top_n]):
            item = semantic_memories[mem_idx].copy()
            # 额外记录多检索器信息（不影响原有字段使用）
            item["semantic_rank"] = semantic_ranks.get(mem_idx)
            item["ner_score"] = float(ner_scores[mem_idx])
            item["rrf_score"] = float(fused_scores[mem_idx])
            item["fused_rank"] = rank_idx + 1
            fused_memories.append(item)

        return fused_memories

    def search_memory(self, user_id, query, conversation_idx=None, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        # mem0 v2 要求 filters 非空，这里强制携带 user_id 过滤，避免 400
        filters = {"user_id": user_id}
        # 本地模式图搜索：当启用了 Neo4j graph_store 时允许；否则保持旧行为不变
        if self.is_graph and self.use_local:
            enable_graph_store = str(os.getenv("ENABLE_GRAPH_STORE", "0")).lower() in ("1", "true", "yes")
            if not enable_graph_store:
                raise ValueError("本地模式暂不支持图搜索：请设置 ENABLE_GRAPH_STORE=1 并配置 NEO4J_URL/NEO4J_USERNAME/NEO4J_PASSWORD，或关闭 --is_graph。")
        while retries < max_retries:
            try:
                if self.use_local:
                    # 本地模式图搜索：本地 Memory.search() 不支持 enable_graph/output_format 参数
                    # 我们需要手动从其内部的 graph 实例中获取关系（如果启用了的话）
                    ner_multiplier = int(os.getenv("MEM0_NER_RRF_MULTIPLIER", "3"))
                    search_limit = self.top_k * ner_multiplier if self.use_ner_rrf_fusion else self.top_k
                    search_results = self.mem0_client.search(query, user_id=user_id, filters=filters, limit=search_limit)

                    # 兼容本地 Memory.search 返回格式：可能是 list，也可能是 dict（例如 {"results": [...], ...}）
                    if isinstance(search_results, dict) and "results" in search_results:
                        search_results_list = search_results.get("results", [])
                    else:
                        search_results_list = search_results
                    
                    if self.is_graph:
                        # 构造兼容的返回结构
                        memories = {
                            "results": search_results_list,
                            "relations": []
                        }
                        # 尝试从 graph_memory 中获取关系
                        try:
                            if hasattr(self.mem0_client, "graph") and self.mem0_client.graph:
                                graph_results = self.mem0_client.graph.search(query, filters=filters)
                                if isinstance(graph_results, dict) and "relations" in graph_results:
                                    memories["relations"] = graph_results["relations"]
                                elif isinstance(graph_results, list):
                                    memories["relations"] = graph_results
                        except Exception as ge:
                            print(f"警告：本地图搜索失败: {ge}")
                    else:
                        memories = search_results
                else:
                    if self.is_graph:
                        print("Searching with graph")
                        memories = self.mem0_client.search(
                            query,
                            user_id=user_id,
                            filters=filters,
                            top_k=self.top_k,
                            filter_memories=self.filter_memories,
                            enable_graph=True,
                            output_format="v1.1",
                        )
                    else:
                        memories = self.mem0_client.search(
                            query,
                            user_id=user_id,
                            filters=filters,
                            top_k=self.top_k,
                            filter_memories=self.filter_memories,
                        )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        # 兼容 mem0 返回格式：
        # - graph: {"results": [...], "relations": [...]}
        # - 非 graph: 可能直接 list，或 {"results": [...]}
        if not self.is_graph:
            if isinstance(memories, dict) and "results" in memories:
                memories_list = memories["results"]
            else:
                memories_list = memories
            semantic_memories = []
            for memory in memories_list:
                memory_dict = memory if isinstance(memory, dict) else {"memory": str(memory)}
                metadata = memory_dict.get("metadata", {})
                
                # 获取dia_ids并获取原始对话（最简单模式下跳过）
                dia_ids = []
                original_conversation = None
                if not self.use_simple_mode:
                    dia_ids = metadata.get("dia_ids", [])
                    if dia_ids and conversation_idx is not None:
                        original_conversation = self._get_conversation_by_dia_ids(dia_ids, conversation_idx)
                
                memory_item = {
                    "memory": memory_dict.get("memory", str(memory)),
                    "timestamp": metadata.get("timestamp", ""),
                    "score": round(memory_dict.get("score", 0), 2) if isinstance(memory_dict.get("score"), (int, float)) else None,
                }
                
                # 如果有原始对话，添加到结果中
                if original_conversation:
                    memory_item["original_conversation"] = original_conversation
                    memory_item["dia_ids"] = dia_ids

                semantic_memories.append(memory_item)

            # 在非图模式下，如果开启了指针化 NER RRF 融合，则在此对语义检索结果重排序
            if self.use_ner_rrf_fusion:
                semantic_memories = self._apply_rrf_fusion(query, semantic_memories)

            graph_memories = None
        else:
            semantic_memories = []
            for memory in memories["results"]:
                # 兼容：results 可能是 dict 列表，也可能是 str 列表
                memory_dict = memory if isinstance(memory, dict) else {"memory": str(memory)}
                metadata = memory_dict.get("metadata", {})
                
                # 获取dia_ids并获取原始对话（最简单模式下跳过）
                dia_ids = []
                original_conversation = None
                if not self.use_simple_mode:
                    dia_ids = metadata.get("dia_ids", [])
                    if dia_ids and conversation_idx is not None:
                        original_conversation = self._get_conversation_by_dia_ids(dia_ids, conversation_idx)
                
                memory_item = {
                    "memory": memory_dict.get("memory", str(memory)),
                    "timestamp": metadata.get("timestamp", ""),
                    "score": round(memory_dict.get("score", 0), 2) if isinstance(memory_dict.get("score"), (int, float)) else None,
                }
                
                # 如果有原始对话，添加到结果中
                if original_conversation:
                    memory_item["original_conversation"] = original_conversation
                    memory_item["dia_ids"] = dia_ids
                
                semantic_memories.append(memory_item)
            
            relations = memories.get("relations", []) if isinstance(memories, dict) else []
            graph_memories = [
                {
                    "source": relation.get("source", ""),
                    "relationship": relation.get("relationship", ""),
                    # mem0 graph_search 返回的是 "destination" 字段；向后兼容同时支持 "target"
                    "target": relation.get("target") or relation.get("destination", ""),
                }
                for relation in relations
                if isinstance(relation, dict)
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def _collect_original_conversations(self, speaker_memories, query=None):
        original_conversations = []
        trim_half = self.use_hybrid_mode and query
        seen_dia_ids = set()
        for item in speaker_memories:
            if "original_conversation" in item and item["original_conversation"]:
                dia_ids = item.get("dia_ids", [])
                dia_ids_key = tuple(sorted(dia_ids)) if dia_ids else None
                if dia_ids_key and dia_ids_key not in seen_dia_ids:
                    convo_text = item["original_conversation"]
                    if trim_half:
                        sents = self._split_text_to_sentences(convo_text)
                        if sents:
                            ranked = self._rank_sentences_by_query(query, sents)
                            keep_n = max(1, (len(ranked) + 1) // 2)
                            convo_text = "\n".join(ranked[:keep_n])
                    original_conversations.append(convo_text)
                    seen_dia_ids.add(dia_ids_key)
        return original_conversations

    def _call_llm(self, prompt):
        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
        t2 = time.time()
        response_time = t2 - t1
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        content = response.choices[0].message.content
        return content, response_time, token_usage

    def _need_original_conversations(self, answer_text):
        if not answer_text:
            return True
        text = str(answer_text).strip().lower()
        triggers = [
            "i don't know",
            "i do not know",
            "unknown",
            "not sure",
            "cannot determine",
            "can't determine",
            "insufficient",
            "need more",
            "no information",
            "no information provided",
            "none",
        ]
        return any(t in text for t in triggers)

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category, conversation_idx=None):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question, conversation_idx=conversation_idx
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question, conversation_idx=conversation_idx
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)

        # 第一阶段：默认不带原文（用于省 prompt_tokens）
        stage1_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            speaker_1_original_conversations="",
            speaker_2_original_conversations="",
            question=question,
        )

        response_text, response_time, token_usage = self._call_llm(stage1_prompt)

        # 兼容旧行为：
        # - 若未开启两阶段，则是否带原文完全由 include_original_conversations 控制
        # - 若开启两阶段，则优先按需加载（即第一阶段不带，必要时第二阶段才带）
        if not self.qa_two_stage:
            if self.include_original_conversations:
                original_conversations_1 = self._collect_original_conversations(speaker_1_memories, query=question)
                original_conversations_2 = self._collect_original_conversations(speaker_2_memories, query=question)

                stage_full_prompt = template.render(
                    speaker_1_user_id=speaker_1_user_id.split("_")[0],
                    speaker_2_user_id=speaker_2_user_id.split("_")[0],
                    speaker_1_memories=json.dumps(search_1_memory, indent=4),
                    speaker_2_memories=json.dumps(search_2_memory, indent=4),
                    speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
                    speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
                    speaker_1_original_conversations=json.dumps(original_conversations_1, indent=4, ensure_ascii=False)
                    if original_conversations_1
                    else "",
                    speaker_2_original_conversations=json.dumps(original_conversations_2, indent=4, ensure_ascii=False)
                    if original_conversations_2
                    else "",
                    question=question,
                )
                response_text, response_time, token_usage = self._call_llm(stage_full_prompt)

            return (
                response_text,
                speaker_1_memories,
                speaker_2_memories,
                speaker_1_memory_time,
                speaker_2_memory_time,
                speaker_1_graph_memories,
                speaker_2_graph_memories,
                response_time,
                token_usage,
            )

        # 两阶段：如果第一阶段“不确定/不知道/需要更多信息”，再加载原文重试
        if self._need_original_conversations(response_text):
            original_conversations_1 = self._collect_original_conversations(speaker_1_memories, query=question)
            original_conversations_2 = self._collect_original_conversations(speaker_2_memories, query=question)

            stage2_prompt = template.render(
                speaker_1_user_id=speaker_1_user_id.split("_")[0],
                speaker_2_user_id=speaker_2_user_id.split("_")[0],
                speaker_1_memories=json.dumps(search_1_memory, indent=4),
                speaker_2_memories=json.dumps(search_2_memory, indent=4),
                speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
                speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
                speaker_1_original_conversations=json.dumps(original_conversations_1, indent=4, ensure_ascii=False)
                if original_conversations_1
                else "",
                speaker_2_original_conversations=json.dumps(original_conversations_2, indent=4, ensure_ascii=False)
                if original_conversations_2
                else "",
                question=question,
            )

            response_text_2, response_time_2, token_usage_2 = self._call_llm(stage2_prompt)

            # 用第二阶段结果覆盖输出，同时把 token_usage 叠加记录（不破坏原结构）
            token_usage = {
                "prompt_tokens": token_usage.get("prompt_tokens", 0) + token_usage_2.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0) + token_usage_2.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0) + token_usage_2.get("total_tokens", 0),
            }
            response_time = response_time + response_time_2
            response_text = response_text_2

        return (
            response_text,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            token_usage,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id, conversation_idx=None):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            token_usage,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category, conversation_idx=conversation_idx)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
            "token_usage": token_usage,
        }

        if self.use_sentence_mode:
            denom = max(self.sentence_stats.get("num_messages", 0), 1)
            result["sentence_mode"] = True
            result["avg_sentences_per_message"] = round(self.sentence_stats.get("total_sentences", 0) / denom, 4)
            result["total_sentences"] = self.sentence_stats.get("total_sentences", 0)
            result["num_messages"] = self.sentence_stats.get("num_messages", 0)
        if self.use_hybrid_mode:
            result["hybrid_mode"] = True

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 如果还没有加载原始数据，加载它
        if not self.original_data:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.original_data = json.load(f)
                print(f"已加载原始数据文件: {len(self.original_data)} 个对话")
            except Exception as e:
                print(f"加载原始数据文件失败: {e}")

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):
                if self.use_sentence_mode:
                    conversation_obj = item.get("conversation", {})
                    total_sent = 0
                    total_msg = 0
                    for k, chats in conversation_obj.items():
                        if k in ["speaker_a", "speaker_b"] or "date" in k or "timestamp" in k:
                            continue
                        if not isinstance(chats, list):
                            continue
                        for chat in chats:
                            if not isinstance(chat, dict):
                                continue
                            text = chat.get("text", "")
                            sents = self._split_text_to_sentences(text)
                            total_sent += len(sents)
                            total_msg += 1
                    self.sentence_stats["total_sentences"] = total_sent
                    self.sentence_stats["num_messages"] = total_msg

                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id, conversation_idx=idx)
                self.results[idx].append(result)

                # Save results after each question is processed
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
