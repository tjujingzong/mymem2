import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import re
import numpy as np

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
                        "deepseek_base_url": llm_base_url,
                    },
                ),
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

        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT
        
        # 加载原始数据文件，用于根据dia_id查找对话内容
        self.original_data = None
        if data_path and os.path.exists(data_path):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    self.original_data = json.load(f)
                print(f"已加载原始数据文件: {len(self.original_data)} 个对话")
            except Exception as e:
                print(f"加载原始数据文件失败: {e}")
                self.original_data = None
    
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

    def search_memory(self, user_id, query, conversation_idx=None, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        # mem0 v2 要求 filters 非空，这里强制携带 user_id 过滤，避免 400
        filters = {"user_id": user_id}
        if self.is_graph and self.use_local:
            raise ValueError("本地模式暂不支持图搜索，请关闭 --is_graph 或使用远程 Mem0。")
        while retries < max_retries:
            try:
                if self.use_local:
                    memories = self.mem0_client.search(query, user_id=user_id, filters=filters, limit=self.top_k)
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
                
                # 获取dia_ids并获取原始对话
                dia_ids = metadata.get("dia_ids", [])
                original_conversation = None
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
            
            graph_memories = None
        else:
            semantic_memories = []
            for memory in memories["results"]:
                metadata = memory.get("metadata", {})
                
                # 获取dia_ids并获取原始对话
                dia_ids = metadata.get("dia_ids", [])
                original_conversation = None
                if dia_ids and conversation_idx is not None:
                    original_conversation = self._get_conversation_by_dia_ids(dia_ids, conversation_idx)
                
                memory_item = {
                    "memory": memory["memory"],
                    "timestamp": metadata.get("timestamp", ""),
                    "score": round(memory.get("score", 0), 2),
                }
                
                # 如果有原始对话，添加到结果中
                if original_conversation:
                    memory_item["original_conversation"] = original_conversation
                    memory_item["dia_ids"] = dia_ids
                
                semantic_memories.append(memory_item)
            
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories["relations"]
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
