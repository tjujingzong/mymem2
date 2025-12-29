#!/usr/bin/env python3
"""
调试脚本：测试 Mem0 的 add 和 search 流程
- 只处理第 0 个会话，便于调试
- 记录所有 LLM 输入输出、embedding 输入输出
- 统计 add 阶段有多少记忆被成功写入
- 测试 search 阶段是否能正确检索到记忆
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 确保能导入项目模块
CURRENT_DIR = Path(__file__).parent
EVAL_DIR = CURRENT_DIR.parent  # evaluation 目录
REPO_ROOT = EVAL_DIR.parent    # 项目根目录

# 把 evaluation 目录加到 sys.path，因为 src 在 evaluation/src/ 下
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

# 也把项目根目录加进去，因为 mem0 包在项目根目录
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from src.memzero.add import MemoryADD
from src.memzero.search import MemorySearch

load_dotenv()

# 创建日志目录
LOG_DIR = EVAL_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 运行时间标签（精确到分钟）
RUN_TAG = datetime.utcnow().strftime("%Y%m%d_%H%M")

# 日志文件路径（带时间戳，避免覆盖）
ADD_LOG_FILE = LOG_DIR / f"debug_add_{RUN_TAG}.jsonl"
SEARCH_LOG_FILE = LOG_DIR / f"debug_search_{RUN_TAG}.jsonl"
SUMMARY_LOG_FILE = LOG_DIR / f"debug_summary_{RUN_TAG}.json"


def log_add_operation(user_id, batch_idx, messages, metadata, result, mem0_client=None, log_file=ADD_LOG_FILE):
    """记录 add 操作的详细信息
    
    对于 UPDATE 事件，如果 previous_memory 和 memory 相同（LLM 返回错误），
    则从历史记录中获取实际的旧记忆内容。
    """
    # 修复 UPDATE 事件中的 previous_memory
    if result and mem0_client and result.get("results"):
        for item in result["results"]:
            if item.get("event") == "UPDATE":
                memory_id = item.get("id")
                previous_memory = item.get("previous_memory", "")
                new_memory = item.get("memory", "")
                
                # 如果 previous_memory 和 memory 相同，说明 LLM 返回有误，从历史记录中获取
                if previous_memory == new_memory and memory_id:
                    try:
                        history = mem0_client.history(memory_id)
                        if history:
                            # 找到最新的 UPDATE 事件，获取其 old_memory
                            for hist_item in reversed(history):
                                if hist_item.get("event") == "UPDATE" and hist_item.get("old_memory"):
                                    item["previous_memory"] = hist_item.get("old_memory")
                                    break
                    except Exception as e:
                        # 如果查询历史记录失败，保持原值
                        print(f"警告：无法从历史记录获取 memory_id={memory_id} 的旧值: {e}")
    
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "operation": "add",
        "user_id": user_id,
        "batch_idx": batch_idx,
        "messages_count": len(messages),
        "messages": messages,  # 完整的消息列表
        "metadata": metadata,
        "result": result,  # Mem0 返回的完整结果（已修复 UPDATE 事件的 previous_memory）
        "results_count": len(result.get("results", [])) if result else 0,
        "has_memories": len(result.get("results", [])) > 0 if result else False,
    }
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return record


def log_search_operation(user_id, query, memories, search_time, log_file=SEARCH_LOG_FILE):
    """记录 search 操作的详细信息"""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "operation": "search",
        "user_id": user_id,
        "query": query,
        "memories_count": len(memories),
        "memories": memories,  # 检索到的记忆列表
        "search_time": search_time,
    }
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return record


def test_add_phase(data_path, conversation_idx=0):
    """测试 add 阶段：只处理指定的会话"""
    print("=" * 80)
    print(f"测试 ADD 阶段 - 会话 #{conversation_idx}")
    print("=" * 80)
    
    # 清空之前的日志
    if ADD_LOG_FILE.exists():
        ADD_LOG_FILE.unlink()
    
    # 初始化 MemoryADD
    memory_add = MemoryADD(data_path=data_path, batch_size=2, is_graph=False)
    memory_add.load_data()
    
    if conversation_idx >= len(memory_add.data):
        print(f"错误：会话索引 {conversation_idx} 超出范围（总共 {len(memory_add.data)} 个会话）")
        return None
    
    item = memory_add.data[conversation_idx]
    conversation = item["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    
    speaker_a_user_id = f"{speaker_a}_{conversation_idx}"
    speaker_b_user_id = f"{speaker_b}_{conversation_idx}"
    
    print(f"Speaker A: {speaker_a} (user_id: {speaker_a_user_id})")
    print(f"Speaker B: {speaker_b} (user_id: {speaker_b_user_id})")
    print()
    
    # 删除旧记忆
    print("删除旧记忆...")
    memory_add.mem0_client.delete_all(user_id=speaker_a_user_id)
    memory_add.mem0_client.delete_all(user_id=speaker_b_user_id)
    print("完成\n")
    
    # 统计信息
    stats = {
        "speaker_a": {
            "user_id": speaker_a_user_id,
            "total_batches": 0,
            "batches_with_memories": 0,
            "total_memories_added": 0,
        },
        "speaker_b": {
            "user_id": speaker_b_user_id,
            "total_batches": 0,
            "batches_with_memories": 0,
            "total_memories_added": 0,
        },
    }
    
    # 处理对话
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
        
        # 处理 Speaker A 的记忆
        print(f"处理 {key} - Speaker A...")
        for batch_idx, i in enumerate(range(0, len(messages), memory_add.batch_size)):
            batch_messages = messages[i : i + memory_add.batch_size]
            result = memory_add.add_memory(
                speaker_a_user_id, batch_messages, metadata={"timestamp": timestamp}
            )
            
            stats["speaker_a"]["total_batches"] += 1
            if result and len(result.get("results", [])) > 0:
                stats["speaker_a"]["batches_with_memories"] += 1
                stats["speaker_a"]["total_memories_added"] += len(result.get("results", []))
            
            # 记录日志
            log_add_operation(
                speaker_a_user_id, batch_idx, batch_messages,
                {"timestamp": timestamp}, result, mem0_client=memory_add.mem0_client
            )
            
            print(f"  Batch {batch_idx + 1}: {len(result.get('results', []))} 条记忆" if result else f"  Batch {batch_idx + 1}: 0 条记忆")
        
        # 处理 Speaker B 的记忆
        print(f"处理 {key} - Speaker B...")
        for batch_idx, i in enumerate(range(0, len(messages_reverse), memory_add.batch_size)):
            batch_messages = messages_reverse[i : i + memory_add.batch_size]
            result = memory_add.add_memory(
                speaker_b_user_id, batch_messages, metadata={"timestamp": timestamp}
            )
            
            stats["speaker_b"]["total_batches"] += 1
            if result and len(result.get("results", [])) > 0:
                stats["speaker_b"]["batches_with_memories"] += 1
                stats["speaker_b"]["total_memories_added"] += len(result.get("results", []))
            
            # 记录日志
            log_add_operation(
                speaker_b_user_id, batch_idx, batch_messages,
                {"timestamp": timestamp}, result, mem0_client=memory_add.mem0_client
            )
            
            print(f"  Batch {batch_idx + 1}: {len(result.get('results', []))} 条记忆" if result else f"  Batch {batch_idx + 1}: 0 条记忆")
        
        print()
    
    # 打印统计信息
    print("=" * 80)
    print("ADD 阶段统计")
    print("=" * 80)
    print(f"Speaker A ({speaker_a_user_id}):")
    print(f"  总批次数: {stats['speaker_a']['total_batches']}")
    print(f"  有记忆的批次数: {stats['speaker_a']['batches_with_memories']}")
    print(f"  总记忆数: {stats['speaker_a']['total_memories_added']}")
    print(f"  成功率: {stats['speaker_a']['batches_with_memories'] / stats['speaker_a']['total_batches'] * 100:.1f}%")
    print()
    print(f"Speaker B ({speaker_b_user_id}):")
    print(f"  总批次数: {stats['speaker_b']['total_batches']}")
    print(f"  有记忆的批次数: {stats['speaker_b']['batches_with_memories']}")
    print(f"  总记忆数: {stats['speaker_b']['total_memories_added']}")
    print(f"  成功率: {stats['speaker_b']['batches_with_memories'] / stats['speaker_b']['total_batches'] * 100:.1f}%")
    print()
    
    return {
        "speaker_a_user_id": speaker_a_user_id,
        "speaker_b_user_id": speaker_b_user_id,
        "stats": stats,
    }


def test_search_phase(data_path, conversation_idx=0, top_k=20):
    """测试 search 阶段：遍历指定会话的全部 QA 问题"""
    print("=" * 80)
    print(f"测试 SEARCH 阶段 - 会话 #{conversation_idx}，遍历全部问题")
    print("=" * 80)
    
    # 清空之前的日志（文件名带时间戳，通常不存在旧文件）
    if SEARCH_LOG_FILE.exists():
        SEARCH_LOG_FILE.unlink()
    
    # 加载数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if conversation_idx >= len(data):
        print(f"错误：会话索引 {conversation_idx} 超出范围（总共 {len(data)} 个会话）")
        return None
    
    item = data[conversation_idx]
    qa = item["qa"]
    conversation = item["conversation"]
    
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    speaker_a_user_id = f"{speaker_a}_{conversation_idx}"
    speaker_b_user_id = f"{speaker_b}_{conversation_idx}"
    
    print(f"Speaker A: {speaker_a} (user_id: {speaker_a_user_id})")
    print(f"Speaker B: {speaker_b} (user_id: {speaker_b_user_id})")
    print(f"问题总数: {len(qa)}")
    print()
    
    # 初始化 MemorySearch
    memory_search = MemorySearch(
        output_path=f"debug_search_result_{RUN_TAG}.json",
        top_k=top_k,
        filter_memories=False,
        is_graph=False,
    )
    
    results = []
    
    for question_idx, question_item in enumerate(qa):
        question = question_item.get("question", "")
        answer = question_item.get("answer", "")
        category = question_item.get("category", -1)
        
        print("-" * 80)
        print(f"问题 #{question_idx}")
        print(f"问题: {question}")
        print(f"正确答案: {answer}")
        print(f"类别: {category}")
        
        # 搜索记忆
        print("搜索 Speaker A 的记忆...")
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = memory_search.search_memory(
            speaker_a_user_id, question
        )
        log_search_operation(speaker_a_user_id, question, speaker_1_memories, speaker_1_memory_time)
        print(f"  找到 {len(speaker_1_memories)} 条记忆，耗时 {speaker_1_memory_time:.3f}s")
        
        print("搜索 Speaker B 的记忆...")
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = memory_search.search_memory(
            speaker_b_user_id, question
        )
        log_search_operation(speaker_b_user_id, question, speaker_2_memories, speaker_2_memory_time)
        print(f"  找到 {len(speaker_2_memories)} 条记忆，耗时 {speaker_2_memory_time:.3f}s")
        print()
        
        # 显示前几条记忆
        if speaker_1_memories:
            print("Speaker A 的前 3 条记忆:")
            for i, mem in enumerate(speaker_1_memories[:3], 1):
                print(f"  {i}. [{mem.get('timestamp', 'N/A')}] {mem.get('memory', 'N/A')} (score: {mem.get('score', 'N/A')})")
        else:
            print("Speaker A: 没有找到记忆")
        print()
        
        if speaker_2_memories:
            print("Speaker B 的前 3 条记忆:")
            for i, mem in enumerate(speaker_2_memories[:3], 1):
                print(f"  {i}. [{mem.get('timestamp', 'N/A')}] {mem.get('memory', 'N/A')} (score: {mem.get('score', 'N/A')})")
        else:
            print("Speaker B: 没有找到记忆")
        print()
        
        results.append({
            "question_idx": question_idx,
            "question": question,
            "answer": answer,
            "category": category,
            "speaker_1_memories_count": len(speaker_1_memories),
            "speaker_2_memories_count": len(speaker_2_memories),
            "speaker_1_memories": speaker_1_memories[:5],  # 只保存前 5 条
            "speaker_2_memories": speaker_2_memories[:5],
        })
    
    return {
        "speaker_a_user_id": speaker_a_user_id,
        "speaker_b_user_id": speaker_b_user_id,
        "results": results,
    }


def main():
    """主函数：运行完整的测试流程"""
    print("\n" + "=" * 80)
    print("Mem0 调试脚本")
    print("=" * 80)
    print()
    
    # 数据路径
    data_path = EVAL_DIR / "dataset" / "locomo1.json"
    
    if not data_path.exists():
        print(f"错误：数据文件不存在: {data_path}")
        return
    
    # 检查环境变量
    print("检查环境变量...")
    mem0_local_mode = os.getenv("MEM0_LOCAL_MODE", "0")
    print(f"  MEM0_LOCAL_MODE: {mem0_local_mode}")
    print(f"  MEM0_VECTOR_PATH: {os.getenv('MEM0_VECTOR_PATH', '默认')}")
    print(f"  OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', '未设置')}")
    print(f"  MODEL: {os.getenv('MODEL', '未设置')}")
    print()
    
    # 测试 add 阶段
    add_result = test_add_phase(data_path, conversation_idx=0)
    
    if not add_result:
        print("ADD 阶段失败，跳过 SEARCH 测试")
        return
    
    print("\n" + "=" * 80)
    print("等待 2 秒，确保向量库写入完成...")
    print("=" * 80)
    time.sleep(2)
    
    # 测试 search 阶段（遍历所有问题）
    search_result = test_search_phase(data_path, conversation_idx=0, top_k=20)
    
    # 保存总结
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_path": str(data_path),
        "conversation_idx": 0,
        "question_indices": list(range(len(search_result["results"]))) if search_result else [],
        "add_result": add_result,
        "search_result": search_result,
    }
    
    with open(SUMMARY_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("=" * 80)
    print("测试完成！")
    print("=" * 80)
    print(f"日志文件:")
    print(f"  ADD 日志: {ADD_LOG_FILE}")
    print(f"  SEARCH 日志: {SEARCH_LOG_FILE}")
    print(f"  总结: {SUMMARY_LOG_FILE}")
    print()
    
    # 最终诊断
    if search_result:
        total_questions = len(search_result["results"])
        total_a = sum(item["speaker_1_memories_count"] for item in search_result["results"])
        total_b = sum(item["speaker_2_memories_count"] for item in search_result["results"])
        
        if total_a == 0 and total_b == 0:
            print("⚠️  警告：两个 speaker 在所有问题上都没有检索到记忆！")
            print("可能的原因：")
            print("  1. ADD 阶段没有成功写入任何记忆（检查 ADD 日志中的 results_count）")
            print("  2. SEARCH 阶段的 user_id 与 ADD 阶段不一致")
            print("  3. 向量库路径或配置不一致")
        elif total_a == 0:
            print(f"⚠️  警告：Speaker A 在 {total_questions} 个问题上均未检索到记忆（Speaker B 总计 {total_b} 条）")
        elif total_b == 0:
            print(f"⚠️  警告：Speaker B 在 {total_questions} 个问题上均未检索到记忆（Speaker A 总计 {total_a} 条）")
        else:
            print(f"✅ 成功：两个 speaker 在 {total_questions} 个问题上均有检索结果")
            print(f"  Speaker A 记忆总数: {total_a} 条")
            print(f"  Speaker B 记忆总数: {total_b} 条")


if __name__ == "__main__":
    main()

