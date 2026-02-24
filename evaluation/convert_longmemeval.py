#!/usr/bin/env python3
"""
将 longmemeval_s_cleaned.json 转换为 locomo 格式

关键转换逻辑：
1. 每个QA项转成一个独立的conversation entry
2. 只包含该QA对应的session（haystack_session_ids中的session）
3. 正确映射时间、category（question_type）、evidence格式
"""

import json
import argparse
from typing import Dict, List, Any


# category 直接保留 question_type 的原始字符串值，不进行映射


def convert_longmemeval_to_locomo(input_path: str, output_path: str, limit: int = 0, start: int = 0):
    """
    转换 longmemeval 格式到 locomo 格式
    
    Args:
        input_path: 输入的 longmemeval_s_cleaned.json 路径
        output_path: 输出的 locomo 格式文件路径
    """
    print(f"正在读取输入文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    print(f"读取完成，共 {total_items} 个QA项")
    if limit and limit > 0:
        print(f"仅转换前 {limit} 个QA项（用于快速实验）")
    
    converted_data = []
    
    if start < 0:
        start = 0
    if start > total_items:
        start = total_items

    end = total_items
    if limit and limit > 0:
        end = min(total_items, start + limit)

    for item_idx in range(start, end):
        item = data[item_idx]
        q_id = item.get("question_id", f"q_{item_idx}")
        question = item.get("question", "")
        answer = item.get("answer", "")
        question_type = item.get("question_type", "")
        question_date = item.get("question_date", "")
        
        # 为当前 QA 生成唯一的 speaker 名称
        # 说明：
        # - locomo 原始数据中每个对话都有自己的角色名（如 Caroline / Melanie）
        # - 这里我们用 question_id 或索引，为每个 QA 创建唯一的「用户/助手」标识，
        #   以便在 mem0 侧区分不同 QA 的对话上下文。
        speaker_a_name = f"user_{q_id}"
        speaker_b_name = f"assistant_{q_id}"

        # 获取该QA对应的session IDs
        haystack_session_ids = item.get("haystack_session_ids", [])
        haystack_sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates", [])
        answer_session_ids = item.get("answer_session_ids", [])
        
        # 1. 构建 conversation 对象（只包含该QA对应的session）
        conversation = {}
        
        # 记录 session_id 到 dia_id 的映射，用于 evidence
        # key: 原始session_id, value: [dia_id列表]
        session_to_dia = {}
        
        # 只处理该QA对应的session（haystack_session_ids中的session）
        # session_num 从1开始，对应转换后的session编号
        session_num = 1
        for pos, s_id in enumerate(haystack_session_ids):
            if pos >= len(haystack_sessions):
                continue
            
            session_content = haystack_sessions[pos]
            session_key = f"session_{session_num}"
            
            # 获取该session的时间（如果有对应的日期）
            date_time_key = f"{session_key}_date_time"
            # 尝试从haystack_dates中找到对应的日期
            session_date = ""
            if haystack_dates and pos < len(haystack_dates):
                session_date = haystack_dates[pos]
            
            # 构建该session的chats
            chats = []
            for turn_idx, msg in enumerate(session_content):
                # 生成 locomo 格式的 dia_id: "D{session_num}:{turn_idx+1}"
                dia_id = f"D{session_num}:{turn_idx + 1}"
                
                # 确定speaker（根据role），并使用当前 QA 独有的说话人名称
                role = msg.get("role", "user")
                if role == "user":
                    speaker = speaker_a_name
                else:
                    speaker = speaker_b_name
                
                chats.append({
                    "dia_id": dia_id,
                    "speaker": speaker,
                    "text": msg.get("content", "")
                })
                
                # 如果这个session是答案来源，记录映射
                if s_id in answer_session_ids:
                    if s_id not in session_to_dia:
                        session_to_dia[s_id] = []
                    session_to_dia[s_id].append(dia_id)
            
            conversation[session_key] = chats
            if session_date:
                conversation[date_time_key] = session_date
            
            session_num += 1
        
        # 设置speaker_a和speaker_b（locomo格式要求），使用当前 QA 独有的说话人名称
        conversation["speaker_a"] = speaker_a_name
        conversation["speaker_b"] = speaker_b_name
        
        # 2. 构建 QA 对象
        evidence = []
        for s_id in answer_session_ids:
            if s_id in session_to_dia:
                evidence.extend(session_to_dia[s_id])
        
        # 如果没有找到evidence，至少保留一个空列表
        if not evidence:
            # 如果answer_session_ids存在但没找到对应的dia_id，尝试从第一个session的第一个turn
            if answer_session_ids and session_num > 1:
                evidence = [f"D1:1"]  # 默认第一个session的第一个turn
        
        # category 直接保留 question_type 的原始字符串值
        category = question_type if question_type else ""
        
        qa_item = {
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "category": category
        }
        
        # 3. 组装成 locomo 格式的一个 entry
        converted_entry = {
            "conversation": conversation,
            "qa": [qa_item],
            "question_id": q_id,  # 保留原始ID方便追踪
            "question_date": question_date,  # 保留问题日期
            "question_type": question_type,  # 保留原始类型
        }
        converted_data.append(converted_entry)
    
    # 保存转换后的数据
    print(f"正在保存转换后的数据到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 转换完成！")
    print(f"  - 输入条目数: {total_items}")
    print(f"  - 实际转换条目数: {len(converted_data)}")
    print(f"  - 输出文件: {output_path}")


def split_to_groups(input_path: str, output_dir: str, group_size: int = 10, total: int = 500, output_prefix: str = "longmemeval_as_locomo_"):
    print(f"正在读取输入文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_items = len(data)
    if total > total_items:
        total = total_items

    if group_size <= 0:
        raise ValueError("group_size 必须 > 0")

    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"将前 {total} 个QA按每组 {group_size} 个切分，输出到: {output_dir}")

    written = 0
    for end in range(group_size, total + 1, group_size):
        start = end - group_size
        out_path = os.path.join(output_dir, f"{output_prefix}{end}.json")
        convert_longmemeval_to_locomo(input_path, out_path, limit=group_size, start=start)
        written += 1

    print(f"✓ 切分完成，共生成 {written} 个文件")


def main():
    parser = argparse.ArgumentParser(description="将 longmemeval_s_cleaned.json 转换为 locomo 格式")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/longmemeval_s_cleaned.json",
        help="输入的 longmemeval 文件路径（默认: dataset/longmemeval_s_cleaned.json）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/longmemeval_as_locomo.json",
        help="输出的 locomo 格式文件路径（默认: dataset/longmemeval_as_locomo.json）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="转换 N 个QA项（默认: 0 表示从 start 开始转换到结尾）"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="从第 start 个QA项开始转换（0-based，默认: 0）"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="是否进行批量切分（500个QA，10个一组）"
    )
    
    args = parser.parse_args()

    if args.split:
        output_dir = "/root/ljz/mymem2/evaluation/dataset/longmemeval"
        split_to_groups(args.input, output_dir)
    else:
        convert_longmemeval_to_locomo(args.input, args.output, limit=args.limit, start=args.start)


if __name__ == "__main__":
    main()
