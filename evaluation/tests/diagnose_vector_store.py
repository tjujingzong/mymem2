#!/usr/bin/env python3
"""
诊断脚本：检查向量库中实际存储的内容
"""
import os
import sys
from pathlib import Path

# 设置路径
CURRENT_DIR = Path(__file__).parent
EVAL_DIR = CURRENT_DIR.parent
REPO_ROOT = EVAL_DIR.parent

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

# 构建 Memory 实例（与 add/search 使用相同的配置）
vector_provider = os.getenv("MEM0_VECTOR_PROVIDER", "faiss")
vector_path = os.getenv("MEM0_VECTOR_PATH", "/root/ljz/mymem2/evaluation/local_mem2/faiss1")
vector_collection = os.getenv("MEM0_VECTOR_COLLECTION", "mem0")
vector_dim = int(os.getenv("MEM0_VECTOR_DIM", os.getenv("MEM0_EMBED_DIM", "1024")))

embed_provider = os.getenv("MEM0_EMBED_PROVIDER", "huggingface")
embed_model = os.getenv("MEM0_EMBED_MODEL", "multi-qa-MiniLM-L6-cos-v1")

llm_provider = os.getenv("MEM0_LLM_PROVIDER", "openai")
llm_model = os.getenv("MEM0_LLM_MODEL", os.getenv("MODEL", "qwen3-4b"))
llm_api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
llm_base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

print("=" * 80)
print("向量库诊断")
print("=" * 80)
print(f"向量库路径: {vector_path}")
print(f"集合名称: {vector_collection}")
print(f"向量维度: {vector_dim}")
print()

memory_cfg = MemoryConfig(
    vector_store=VectorStoreConfig(
        provider=vector_provider,
        config={
            "path": vector_path,
            "collection_name": vector_collection,
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
)

mem0_client = Memory(config=memory_cfg)

# 测试 user_id
test_user_ids = ["Calvin_9", "Dave_9"]

for user_id in test_user_ids:
    print(f"\n{'=' * 80}")
    print(f"检查 user_id: {user_id}")
    print(f"{'=' * 80}")
    
    # 1. 尝试列出所有记忆（不使用 query）
    try:
        filters = {"user_id": user_id}
        # 传入一个很大的 limit 值以获取所有记忆（默认 limit=100）
        all_memories = mem0_client.vector_store.list(filters=filters, limit=10000)
        print(f"使用 filters={{'user_id': '{user_id}'}} 列出记忆:")
        print(f"  找到 {len(all_memories[0])} 条记忆")
        
        if len(all_memories[0]) > 0:
            print(f"\n前 3 条记忆:")
            for i, mem in enumerate(all_memories[0][:3], 1):
                # OutputData 结构: {id, score, payload}
                mem_dict = mem if isinstance(mem, dict) else mem.__dict__
                payload = mem_dict.get('payload', {}) if isinstance(mem_dict, dict) else (mem.payload if hasattr(mem, 'payload') else {})
                
                mem_id = mem_dict.get('id', 'N/A') if isinstance(mem_dict, dict) else (mem.id if hasattr(mem, 'id') else 'N/A')
                memory_text = payload.get('data', payload.get('memory', payload.get('text', 'N/A'))) if payload else 'N/A'
                
                print(f"  {i}. ID: {mem_id}")
                print(f"     Memory: {memory_text[:150] if memory_text != 'N/A' else 'N/A'}...")
                print(f"     Payload keys: {list(payload.keys()) if payload else 'N/A'}")
                print(f"     User ID: {payload.get('user_id', 'N/A') if payload else 'N/A'}")
                print(f"     Timestamp: {payload.get('timestamp', 'N/A') if payload else 'N/A'}")
        else:
            print("  ⚠️  没有找到任何记忆！")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 测试 embedding 生成
    test_query = "Tokyo"
    print(f"\n测试 embedding 生成（查询: '{test_query}'）:")
    try:
        query_embedding = mem0_client.embedding_model.embed(test_query, "search")
        print(f"  Embedding 维度: {len(query_embedding)}")
        print(f"  Embedding 前5个值: {query_embedding[:5]}")
    except Exception as e:
        print(f"  ❌ Embedding 生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 尝试搜索（不传 filters，让 Memory.search 自己构建）
    print(f"\n使用查询 '{test_query}' 搜索（通过 Memory.search）:")
    try:
        # 注意：不要重复传 filters，Memory.search 会根据 user_id 自动构建
        search_results = mem0_client.search(test_query, user_id=user_id, limit=10)
        
        # 处理返回结果
        if isinstance(search_results, dict) and "results" in search_results:
            memories_list = search_results["results"]
        elif isinstance(search_results, list):
            memories_list = search_results
        else:
            memories_list = []
        
        print(f"  找到 {len(memories_list)} 条记忆")
        
        if len(memories_list) > 0:
            print(f"\n前 5 条搜索结果:")
            for i, mem in enumerate(memories_list[:5], 1):
                # Memory.search 返回的是 MemoryItem，结构是 {id, memory, score, ...}
                mem_dict = mem if isinstance(mem, dict) else mem.model_dump() if hasattr(mem, 'model_dump') else mem.__dict__
                
                memory_text = mem_dict.get('memory', 'N/A')
                score = mem_dict.get('score', 'N/A')
                mem_id = mem_dict.get('id', 'N/A')
                
                print(f"  {i}. ID: {mem_id}")
                print(f"     Memory: {memory_text[:150] if memory_text != 'N/A' else 'N/A'}...")
                print(f"     Score: {score}")
                print(f"     User ID: {mem_dict.get('user_id', 'N/A')}")
        else:
            print("  ⚠️  搜索没有返回任何结果！")
            print("  可能的原因：")
            print("    1. Embedding 模型不一致（add 和 search 用了不同的模型）")
            print("    2. 向量相似度太低，没有匹配到")
            print("    3. Filters 过滤掉了所有结果")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 直接测试向量库搜索（绕过 Memory.search，直接调用 vector_store.search）
    print(f"\n直接测试向量库搜索（绕过 Memory 层）:")
    try:
        import numpy as np
        import faiss
        
        query_embedding = mem0_client.embedding_model.embed(test_query, "search")
        filters = {"user_id": user_id}
        
        # 检查索引状态
        vs = mem0_client.vector_store
        print(f"  索引中的向量总数: {vs.index.ntotal if vs.index else 0}")
        print(f"  index_to_id 映射数量: {len(vs.index_to_id)}")
        print(f"  docstore 中的记录数: {len(vs.docstore)}")
        
        # 直接调用 Faiss 搜索，查看原始结果
        query_vectors = np.array([query_embedding], dtype=np.float32)
        if vs.normalize_L2 and vs.distance_strategy.lower() == "euclidean":
            faiss.normalize_L2(query_vectors)
        
        fetch_k = 20  # 获取更多结果用于调试
        scores, indices = vs.index.search(query_vectors, fetch_k)
        
        print(f"\n  Faiss 原始搜索结果:")
        print(f"    Scores: {scores[0][:10]}")
        print(f"    Indices: {indices[0][:10]}")
        print(f"    非 -1 的索引数量: {sum(1 for idx in indices[0] if idx != -1)}")
        
        # 检查 index_to_id 映射
        valid_indices = [int(idx) for idx in indices[0] if idx != -1]
        if valid_indices:
            print(f"\n  检查前 5 个有效索引的映射:")
            for idx in valid_indices[:5]:
                vector_id = vs.index_to_id.get(idx)
                print(f"    Index {idx} -> Vector ID: {vector_id}")
                if vector_id:
                    payload = vs.docstore.get(vector_id)
                    if payload:
                        print(f"      Payload user_id: {payload.get('user_id', 'N/A')}")
                        print(f"      Payload data: {payload.get('data', 'N/A')[:50]}...")
                    else:
                        print(f"      ⚠️  docstore 中没有找到 vector_id: {vector_id}")
                else:
                    print(f"      ⚠️  index_to_id 中没有找到 index: {idx}")
        
        # 直接调用 vector_store.search
        raw_results = vs.search(
            query=test_query,
            vectors=[query_embedding],
            limit=10,
            filters=filters
        )
        print(f"\n  经过 _parse_output 和 filters 后的结果数: {len(raw_results)}")
        
        if len(raw_results) > 0:
            print(f"\n前 3 条原始搜索结果:")
            for i, result in enumerate(raw_results[:3], 1):
                print(f"  {i}. ID: {result.id}")
                print(f"     Score: {result.score}")
                print(f"     Payload keys: {list(result.payload.keys())}")
                print(f"     Memory: {result.payload.get('data', 'N/A')[:100]}...")
                print(f"     User ID: {result.payload.get('user_id', 'N/A')}")
        else:
            print("  ⚠️  向量库搜索也没有返回任何结果！")
            print("  可能的原因：")
            print("    1. Faiss 搜索返回的所有 indices 都是 -1（没有相似向量）")
            print("    2. index_to_id 映射不完整")
            print("    3. filters 过滤掉了所有结果")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)

