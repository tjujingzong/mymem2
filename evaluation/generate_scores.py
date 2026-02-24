import argparse
import json
import re

import pandas as pd
from metrics.utils import calculate_bleu_scores, calculate_metrics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate evaluation scores from JSON data")
    parser.add_argument("input_file", help="Path to the input JSON file (e.g., evaluation_metrics.json)")
    parser.add_argument("-o", "--output", default="scores_output.txt", 
                        help="Path to the output text file (default: scores_output.txt)")
    
    args = parser.parse_args()
    
    # Load the evaluation metrics data
    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    # Flatten the data into a list of question items
    all_items = []
    for key in data:
        all_items.extend(data[key])
    
    # Process items: for responses with <think>, recalculate scores using only content outside the tags
    for item in all_items:
        response = str(item.get("response", ""))
        answer = str(item.get("answer", ""))
        
        # Check if response contains <think> tag
        if "<think>" in response:
            # Extract content outside the <think>...</think> tags
            # Remove the entire tag and its content
            cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            cleaned_response = cleaned_response.strip()
            
            # Recalculate bleu_score and f1_score using cleaned response
            if cleaned_response and answer:
                bleu_scores = calculate_bleu_scores(cleaned_response, answer)
                metrics = calculate_metrics(cleaned_response, answer)
                item["bleu_score"] = bleu_scores["bleu1"]
                item["f1_score"] = metrics["f1"]
    
    # Convert to DataFrame
    df = pd.DataFrame(all_items)
    
    # 保持 category 为字符串（例如 single-session-user），不再强制转为数值类型
    
    # 处理token_usage字段（如果是字典格式，展开为单独的列）
    if "token_usage" in df.columns:
        # 将token_usage字典展开为单独的列
        token_df = pd.json_normalize(df["token_usage"])
        if not token_df.empty:
            df["prompt_tokens"] = token_df.get("prompt_tokens", 0)
            df["completion_tokens"] = token_df.get("completion_tokens", 0)
            df["total_tokens"] = token_df.get("total_tokens", 0)
    
    # 处理llm_judge_token_usage字段（评估过程的token）
    if "llm_judge_token_usage" in df.columns:
        judge_token_df = pd.json_normalize(df["llm_judge_token_usage"])
        if not judge_token_df.empty:
            df["llm_judge_prompt_tokens"] = judge_token_df.get("prompt_tokens", 0)
            df["llm_judge_completion_tokens"] = judge_token_df.get("completion_tokens", 0)
            df["llm_judge_total_tokens"] = judge_token_df.get("total_tokens", 0)
    
    # 准备聚合的列
    agg_dict = {"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}
    
    # 添加时间指标
    if "response_time" in df.columns:
        agg_dict["response_time"] = "mean"
    if "speaker_1_memory_time" in df.columns:
        agg_dict["speaker_1_memory_time"] = "mean"
    if "speaker_2_memory_time" in df.columns:
        agg_dict["speaker_2_memory_time"] = "mean"
    
    # 添加token指标（QA过程的token）
    if "total_tokens" in df.columns:
        agg_dict["total_tokens"] = "mean"
    if "prompt_tokens" in df.columns:
        agg_dict["prompt_tokens"] = "mean"
    if "completion_tokens" in df.columns:
        agg_dict["completion_tokens"] = "mean"
    
    # 添加LLM judge评估过程的token指标
    if "llm_judge_total_tokens" in df.columns:
        agg_dict["llm_judge_total_tokens"] = "mean"
    if "llm_judge_prompt_tokens" in df.columns:
        agg_dict["llm_judge_prompt_tokens"] = "mean"
    if "llm_judge_completion_tokens" in df.columns:
        agg_dict["llm_judge_completion_tokens"] = "mean"
    
    # Calculate mean scores by category
    result = df.groupby("category").agg(agg_dict).round(4)
    
    # Add count of questions per category
    result["count"] = df.groupby("category").size()
    
    # Calculate overall means
    overall_means = df.agg(agg_dict).round(4)
    
    # Write results to output file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("Mean Scores Per Category:\n")
        f.write(str(result))
        f.write("\n\nOverall Mean Scores:\n")
        f.write(str(overall_means))
        f.write("\n")
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
