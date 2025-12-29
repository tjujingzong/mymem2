import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import Memory, MemoryClient
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

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
        # 测试：删除自定义 prompt，使用 mem0 默认的 prompt
        # custom_fact_extraction_prompt=custom_instructions,
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
    def __init__(self, data_path=None, batch_size=2, is_graph=False):
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
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                if self.use_local:
                    result = self.mem0_client.add(message, user_id=user_id, metadata=metadata, infer=True)
                else:
                    result = self.mem0_client.add(
                        message, user_id=user_id, version="v2", metadata=metadata, enable_graph=self.is_graph
                    )
                return result
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
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
            result = self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

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
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B"),
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
