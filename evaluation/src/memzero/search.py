import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False):
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
        self.is_graph = is_graph
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
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

            semantic_memories = [
                {
                    "memory": memory["memory"] if isinstance(memory, dict) else str(memory),
                    "timestamp": memory.get("metadata", {}).get("timestamp") if isinstance(memory, dict) else "",
                    "score": round(memory.get("score", 0), 2) if isinstance(memory, dict) else None,
                }
                for memory in memories_list
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories["relations"]
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )

        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL"), messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        t2 = time.time()
        response_time = t2 - t1
        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
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
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

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
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

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
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
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
