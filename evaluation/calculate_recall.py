#!/usr/bin/env python3
"""
计算 Top-5 Accuracy 的脚本

Top-5 Accuracy: 在 Top-5 检索结果中，如果包含至少一个正确的记忆（evidence中的dia_ids），则为1，否则为0

Top-K Accuracy = {
    1, 如果前K个结果中有至少一个正确答案
    0, 如果前K个结果中没有任何正确答案
}

其中：
- evidence 中的 dia_ids 是应该检索到的（ground truth）
- speaker_1_memories 和 speaker_2_memories 中的前5个记忆是实际检索到的 Top-5
- 分别计算 speaker_1 和 speaker_2 的 Top-5 Accuracy
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


def calculate_topk_accuracy_for_question(question_data: Dict, k: int) -> Optional[Tuple[int, int]]:
    """
    计算单个问题的 Top-K Accuracy
    
    Args:
        question_data: 包含 evidence, speaker_1_memories, speaker_2_memories 的字典
        k: Top-K 中的 K 值
        
    Returns:
        (speaker_1_topk_accuracy, speaker_2_topk_accuracy) 元组，值为 0 或 1
        如果没有 evidence，返回 None
    """
    # 获取正确答案的 dia_ids（ground truth）
    evidence_ids = set(question_data.get("evidence", []))
    
    if not evidence_ids:
        # 如果没有 evidence，无法计算 Top-K Accuracy
        return None
    
    # 计算 speaker_1 的 Top-K Accuracy
    speaker_1_topk_accuracy = 0
    speaker_1_memories = question_data.get("speaker_1_memories", [])
    # 取前K个记忆（假设已经按score排序）
    top_k_speaker_1 = speaker_1_memories[:k]
    
    for memory in top_k_speaker_1:
        if isinstance(memory, dict) and "dia_ids" in memory:
            memory_dia_ids = set(memory["dia_ids"])
            # 如果这个记忆的dia_ids与evidence有交集，说明找到了正确答案
            if evidence_ids & memory_dia_ids:
                speaker_1_topk_accuracy = 1
                break  # 找到至少一个就足够了
    
    # 计算 speaker_2 的 Top-K Accuracy
    speaker_2_topk_accuracy = 0
    speaker_2_memories = question_data.get("speaker_2_memories", [])
    # 取前K个记忆（假设已经按score排序）
    top_k_speaker_2 = speaker_2_memories[:k]
    
    for memory in top_k_speaker_2:
        if isinstance(memory, dict) and "dia_ids" in memory:
            memory_dia_ids = set(memory["dia_ids"])
            # 如果这个记忆的dia_ids与evidence有交集，说明找到了正确答案
            if evidence_ids & memory_dia_ids:
                speaker_2_topk_accuracy = 1
                break  # 找到至少一个就足够了
    
    return (speaker_1_topk_accuracy, speaker_2_topk_accuracy)


def calculate_top5_accuracy_for_question(question_data: Dict) -> Optional[Tuple[int, int]]:
    """计算单个问题的 Top-5 Accuracy（兼容旧接口）"""
    return calculate_topk_accuracy_for_question(question_data, 5)


def calculate_topk_accuracy_from_file(json_file_path: str, k: int) -> Dict:
    """
    从JSON文件计算 Top-K Accuracy
    
    Args:
        json_file_path: JSON文件路径
        k: Top-K 中的 K 值
        
    Returns:
        包含统计信息的字典
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    speaker_1_accuracies = []
    speaker_2_accuracies = []
    speaker_1_accuracies_by_category = {}
    speaker_2_accuracies_by_category = {}
    total_questions = 0
    questions_without_evidence = 0
    excluded_category_5 = 0
    
    # 遍历所有问题组
    for group_key, questions in data.items():
        if not isinstance(questions, list):
            continue
            
        for question_data in questions:
            total_questions += 1
            
            # 排除类别5
            category = question_data.get("category", "unknown")
            if category == 5:
                excluded_category_5 += 1
                continue
            
            accuracy_result = calculate_topk_accuracy_for_question(question_data, k)
            
            if accuracy_result is None:
                questions_without_evidence += 1
                continue
            
            speaker_1_accuracy, speaker_2_accuracy = accuracy_result
            
            speaker_1_accuracies.append(speaker_1_accuracy)
            speaker_2_accuracies.append(speaker_2_accuracy)
            
            # 按类别统计
            if category not in speaker_1_accuracies_by_category:
                speaker_1_accuracies_by_category[category] = []
                speaker_2_accuracies_by_category[category] = []
            speaker_1_accuracies_by_category[category].append(speaker_1_accuracy)
            speaker_2_accuracies_by_category[category].append(speaker_2_accuracy)
    
    # 计算统计信息
    results = {
        "total_questions": total_questions,
        "excluded_category_5": excluded_category_5,
        "questions_without_evidence": questions_without_evidence,
        "valid_questions": len(speaker_1_accuracies),
        "speaker_1": {
            f"overall_top{k}_accuracy": sum(speaker_1_accuracies) / len(speaker_1_accuracies) if speaker_1_accuracies else 0.0,
            f"top{k}_accuracy_count": sum(speaker_1_accuracies),
            f"top{k}_accuracy_by_category": {}
        },
        "speaker_2": {
            f"overall_top{k}_accuracy": sum(speaker_2_accuracies) / len(speaker_2_accuracies) if speaker_2_accuracies else 0.0,
            f"top{k}_accuracy_count": sum(speaker_2_accuracies),
            f"top{k}_accuracy_by_category": {}
        }
    }
    
    # 计算每个类别的平均 Top-K Accuracy
    for category in speaker_1_accuracies_by_category.keys():
        s1_accuracies = speaker_1_accuracies_by_category[category]
        s2_accuracies = speaker_2_accuracies_by_category[category]
        
        results["speaker_1"][f"top{k}_accuracy_by_category"][category] = {
            "count": len(s1_accuracies),
            f"top{k}_accuracy": sum(s1_accuracies) / len(s1_accuracies) if s1_accuracies else 0.0,
            f"top{k}_accuracy_count": sum(s1_accuracies)
        }
        
        results["speaker_2"][f"top{k}_accuracy_by_category"][category] = {
            "count": len(s2_accuracies),
            f"top{k}_accuracy": sum(s2_accuracies) / len(s2_accuracies) if s2_accuracies else 0.0,
            f"top{k}_accuracy_count": sum(s2_accuracies)
        }
    
    return results


def calculate_top5_accuracy_from_file(json_file_path: str) -> Dict:
    """
    从JSON文件计算 Top-5 Accuracy（兼容旧接口）
    
    Args:
        json_file_path: JSON文件路径
        
    Returns:
        包含统计信息的字典
    """
    return calculate_topk_accuracy_from_file(json_file_path, 5)
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    speaker_1_accuracies = []
    speaker_2_accuracies = []
    speaker_1_accuracies_by_category = {}
    speaker_2_accuracies_by_category = {}
    total_questions = 0
    questions_without_evidence = 0
    excluded_category_5 = 0
    
    # 遍历所有问题组
    for group_key, questions in data.items():
        if not isinstance(questions, list):
            continue
            
        for question_data in questions:
            total_questions += 1
            
            # 排除类别5
            category = question_data.get("category", "unknown")
            if category == 5:
                excluded_category_5 += 1
                continue
            
            accuracy_result = calculate_top5_accuracy_for_question(question_data)
            
            if accuracy_result is None:
                questions_without_evidence += 1
                continue
            
            speaker_1_accuracy, speaker_2_accuracy = accuracy_result
            
            speaker_1_accuracies.append(speaker_1_accuracy)
            speaker_2_accuracies.append(speaker_2_accuracy)
            
            # 按类别统计
            if category not in speaker_1_accuracies_by_category:
                speaker_1_accuracies_by_category[category] = []
                speaker_2_accuracies_by_category[category] = []
            speaker_1_accuracies_by_category[category].append(speaker_1_accuracy)
            speaker_2_accuracies_by_category[category].append(speaker_2_accuracy)
    
    # 计算统计信息
    results = {
        "total_questions": total_questions,
        "excluded_category_5": excluded_category_5,
        "questions_without_evidence": questions_without_evidence,
        "valid_questions": len(speaker_1_accuracies),
        "speaker_1": {
            "overall_top5_accuracy": sum(speaker_1_accuracies) / len(speaker_1_accuracies) if speaker_1_accuracies else 0.0,
            "top5_accuracy_count": sum(speaker_1_accuracies),
            "top5_accuracy_by_category": {}
        },
        "speaker_2": {
            "overall_top5_accuracy": sum(speaker_2_accuracies) / len(speaker_2_accuracies) if speaker_2_accuracies else 0.0,
            "top5_accuracy_count": sum(speaker_2_accuracies),
            "top5_accuracy_by_category": {}
        }
    }
    
    # 计算每个类别的平均 Top-5 Accuracy
    for category in speaker_1_accuracies_by_category.keys():
        s1_accuracies = speaker_1_accuracies_by_category[category]
        s2_accuracies = speaker_2_accuracies_by_category[category]
        
        results["speaker_1"]["top5_accuracy_by_category"][category] = {
            "count": len(s1_accuracies),
            "top5_accuracy": sum(s1_accuracies) / len(s1_accuracies) if s1_accuracies else 0.0,
            "top5_accuracy_count": sum(s1_accuracies)
        }
        
        results["speaker_2"]["top5_accuracy_by_category"][category] = {
            "count": len(s2_accuracies),
            "top5_accuracy": sum(s2_accuracies) / len(s2_accuracies) if s2_accuracies else 0.0,
            "top5_accuracy_count": sum(s2_accuracies)
        }
    
    return results


def main():
    """主函数"""
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # 默认文件路径
        json_file = "results_search/mem0_results_top_10-8b.json"
    
    json_path = Path(__file__).parent / json_file
    
    if not json_path.exists():
        print(f"错误：文件不存在: {json_path}")
        sys.exit(1)
    
    print(f"正在计算 Top-5 和 Top-10 Accuracy...")
    print(f"文件: {json_path}")
    print("=" * 60)
    
    # 计算 Top-5 Accuracy
    print(f"\n{'='*60}")
    print(f"Top-5 Accuracy 结果:")
    print(f"{'='*60}")
    results_top5 = calculate_topk_accuracy_from_file(str(json_path), 5)
    
    # 打印 Top-5 结果
    print(f"\n总体统计:")
    print(f"  总问题数: {results_top5['total_questions']}")
    print(f"  排除的类别5问题数: {results_top5['excluded_category_5']}")
    print(f"  无evidence的问题数: {results_top5['questions_without_evidence']}")
    print(f"  有效问题数: {results_top5['valid_questions']}")
    
    print(f"\nSpeaker 1 (Top-5 Accuracy):")
    print(f"  整体 Top-5 Accuracy: {results_top5['speaker_1']['overall_top5_accuracy']:.4f}")
    print(f"  成功检索到正确答案的问题数: {results_top5['speaker_1']['top5_accuracy_count']} / {results_top5['valid_questions']}")
    
    print(f"\nSpeaker 2 (Top-5 Accuracy):")
    print(f"  整体 Top-5 Accuracy: {results_top5['speaker_2']['overall_top5_accuracy']:.4f}")
    print(f"  成功检索到正确答案的问题数: {results_top5['speaker_2']['top5_accuracy_count']} / {results_top5['valid_questions']}")
    
    print(f"\n按类别统计 - Speaker 1 (Top-5):")
    for category in sorted(results_top5['speaker_1']['top5_accuracy_by_category'].keys()):
        stats = results_top5['speaker_1']['top5_accuracy_by_category'][category]
        print(f"  类别 {category}:")
        print(f"    问题数: {stats['count']}")
        print(f"    Top-5 Accuracy: {stats['top5_accuracy']:.4f}")
        print(f"    成功检索数: {stats['top5_accuracy_count']} / {stats['count']}")
    
    print(f"\n按类别统计 - Speaker 2 (Top-5):")
    for category in sorted(results_top5['speaker_2']['top5_accuracy_by_category'].keys()):
        stats = results_top5['speaker_2']['top5_accuracy_by_category'][category]
        print(f"  类别 {category}:")
        print(f"    问题数: {stats['count']}")
        print(f"    Top-5 Accuracy: {stats['top5_accuracy']:.4f}")
        print(f"    成功检索数: {stats['top5_accuracy_count']} / {stats['count']}")
    
    # 计算 Top-10 Accuracy
    print(f"\n{'='*60}")
    print(f"Top-10 Accuracy 结果:")
    print(f"{'='*60}")
    results_top10 = calculate_topk_accuracy_from_file(str(json_path), 10)
    
    # 打印 Top-10 结果
    print(f"\n总体统计:")
    print(f"  总问题数: {results_top10['total_questions']}")
    print(f"  排除的类别5问题数: {results_top10['excluded_category_5']}")
    print(f"  无evidence的问题数: {results_top10['questions_without_evidence']}")
    print(f"  有效问题数: {results_top10['valid_questions']}")
    
    print(f"\nSpeaker 1 (Top-10 Accuracy):")
    print(f"  整体 Top-10 Accuracy: {results_top10['speaker_1']['overall_top10_accuracy']:.4f}")
    print(f"  成功检索到正确答案的问题数: {results_top10['speaker_1']['top10_accuracy_count']} / {results_top10['valid_questions']}")
    
    print(f"\nSpeaker 2 (Top-10 Accuracy):")
    print(f"  整体 Top-10 Accuracy: {results_top10['speaker_2']['overall_top10_accuracy']:.4f}")
    print(f"  成功检索到正确答案的问题数: {results_top10['speaker_2']['top10_accuracy_count']} / {results_top10['valid_questions']}")
    
    print(f"\n按类别统计 - Speaker 1 (Top-10):")
    for category in sorted(results_top10['speaker_1']['top10_accuracy_by_category'].keys()):
        stats = results_top10['speaker_1']['top10_accuracy_by_category'][category]
        print(f"  类别 {category}:")
        print(f"    问题数: {stats['count']}")
        print(f"    Top-10 Accuracy: {stats['top10_accuracy']:.4f}")
        print(f"    成功检索数: {stats['top10_accuracy_count']} / {stats['count']}")
    
    print(f"\n按类别统计 - Speaker 2 (Top-10):")
    for category in sorted(results_top10['speaker_2']['top10_accuracy_by_category'].keys()):
        stats = results_top10['speaker_2']['top10_accuracy_by_category'][category]
        print(f"  类别 {category}:")
        print(f"    问题数: {stats['count']}")
        print(f"    Top-10 Accuracy: {stats['top10_accuracy']:.4f}")
        print(f"    成功检索数: {stats['top10_accuracy_count']} / {stats['count']}")
    
    output_file_top5 = json_path.parent / f"{json_path.stem}_top5_accuracy.json"
    with open(output_file_top5, 'w', encoding='utf-8') as f:
        json.dump(results_top5, f, indent=2, ensure_ascii=False)
    
    output_file_top10 = json_path.parent / f"{json_path.stem}_top10_accuracy.json"
    with open(output_file_top10, 'w', encoding='utf-8') as f:
        json.dump(results_top10, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到:")
    print(f"  Top-5: {output_file_top5}")
    print(f"  Top-10: {output_file_top10}")


if __name__ == "__main__":
    main()

