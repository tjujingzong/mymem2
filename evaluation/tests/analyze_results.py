#!/usr/bin/env python3
"""
分析评估结果，找出问题所在
"""
import json
import os
from collections import defaultdict

def analyze_results():
    # 读取评估结果
    with open("/root/ljz/mymem/evaluation/evaluation_metrics2.json", "r") as f:
        data = json.load(f)
    
    # 统计信息
    stats = {
        "total": 0,
        "correct_llm": 0,
        "wrong_llm": 0,
        "date_errors": [],
        "missing_memory_errors": [],
        "format_errors": [],
        "category_stats": defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0})
    }
    
    # 分析每个问题
    for key, items in data.items():
        for item in items:
            stats["total"] += 1
            category = item["category"]
            stats["category_stats"][category]["total"] += 1
            
            question = item["question"]
            answer = item["answer"]
            response = item["response"]
            llm_score = item["llm_score"]
            f1_score = item["f1_score"]
            bleu_score = item["bleu_score"]
            
            if llm_score == 1:
                stats["correct_llm"] += 1
                stats["category_stats"][category]["correct"] += 1
            else:
                stats["wrong_llm"] += 1
                stats["category_stats"][category]["wrong"] += 1
                
                # 分析错误类型
                response_lower = response.lower()
                answer_lower = answer.lower()
                
                # 检查是否是日期错误
                if any(word in question.lower() for word in ["when", "date", "time"]):
                    if "january" in answer_lower or "february" in answer_lower or "march" in answer_lower or \
                       "april" in answer_lower or "may" in answer_lower or "june" in answer_lower or \
                       "july" in answer_lower or "august" in answer_lower or "september" in answer_lower or \
                       "october" in answer_lower or "november" in answer_lower or "december" in answer_lower or \
                       "2023" in answer or "2022" in answer or "2024" in answer:
                        stats["date_errors"].append({
                            "question": question,
                            "answer": answer,
                            "response": response,
                            "f1": f1_score,
                            "bleu": bleu_score
                        })
                
                # 检查是否是记忆缺失
                if "not mention" in response_lower or "do not mention" in response_lower or \
                   "no mention" in response_lower or "neither" in response_lower and "mentioned" in response_lower:
                    stats["missing_memory_errors"].append({
                        "question": question,
                        "answer": answer,
                        "response": response,
                        "f1": f1_score,
                        "bleu": bleu_score
                    })
                
                # 检查格式错误（语义正确但格式不对）
                if f1_score > 0.3 or bleu_score > 0.3:
                    stats["format_errors"].append({
                        "question": question,
                        "answer": answer,
                        "response": response,
                        "f1": f1_score,
                        "bleu": bleu_score
                    })
    
    # 打印统计结果
    print("=" * 80)
    print("评估结果分析")
    print("=" * 80)
    print(f"\n总体统计:")
    print(f"  总问题数: {stats['total']}")
    print(f"  LLM Judge 正确: {stats['correct_llm']} ({stats['correct_llm']/stats['total']*100:.2f}%)")
    print(f"  LLM Judge 错误: {stats['wrong_llm']} ({stats['wrong_llm']/stats['total']*100:.2f}%)")
    
    print(f"\n按类别统计:")
    for cat in sorted(stats['category_stats'].keys()):
        cat_stats = stats['category_stats'][cat]
        if cat_stats['total'] > 0:
            accuracy = cat_stats['correct'] / cat_stats['total'] * 100
            print(f"  类别 {cat}: {cat_stats['correct']}/{cat_stats['total']} ({accuracy:.2f}%)")
    
    print(f"\n错误类型分析:")
    print(f"  日期错误: {len(stats['date_errors'])} 个")
    print(f"  记忆缺失错误: {len(stats['missing_memory_errors'])} 个")
    print(f"  格式错误（语义正确但格式不对）: {len(stats['format_errors'])} 个")
    
    # 显示一些典型错误例子
    if stats['date_errors']:
        print(f"\n日期错误示例（前5个）:")
        for i, error in enumerate(stats['date_errors'][:5], 1):
            print(f"\n  {i}. 问题: {error['question']}")
            print(f"     正确答案: {error['answer']}")
            print(f"     生成回答: {error['response']}")
            print(f"     F1: {error['f1']:.3f}, BLEU: {error['bleu']:.3f}")
    
    if stats['missing_memory_errors']:
        print(f"\n记忆缺失错误示例（前5个）:")
        for i, error in enumerate(stats['missing_memory_errors'][:5], 1):
            print(f"\n  {i}. 问题: {error['question']}")
            print(f"     正确答案: {error['answer']}")
            print(f"     生成回答: {error['response']}")
            print(f"     F1: {error['f1']:.3f}, BLEU: {error['bleu']:.3f}")
    
    if stats['format_errors']:
        print(f"\n格式错误示例（语义正确但格式不对，前5个）:")
        for i, error in enumerate(stats['format_errors'][:5], 1):
            print(f"\n  {i}. 问题: {error['question']}")
            print(f"     正确答案: {error['answer']}")
            print(f"     生成回答: {error['response']}")
            print(f"     F1: {error['f1']:.3f}, BLEU: {error['bleu']:.3f}")
    
    print("\n" + "=" * 80)
    print("可能的问题:")
    print("=" * 80)
    print("1. 记忆检索问题: 很多问题显示'记忆中没有提到'，说明检索到的记忆不够准确")
    print("2. 日期计算问题: 很多日期回答错误，可能是相对时间转换有问题")
    print("3. 模型问题: 如果使用了DeepSeek而不是GPT-4o-mini，回答质量可能下降")
    print("4. LLM Judge问题: 如果LLM Judge使用的是DeepSeek，评分标准可能与论文不同")
    print("5. 提示词遵循问题: 模型可能没有很好地遵循'回答应该少于5-6个词'的指示")

if __name__ == "__main__":
    analyze_results()
