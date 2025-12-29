import argparse
import concurrent.futures
import json
import os
import threading
from collections import defaultdict

from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

def process_item(item_data, pbar, pbar_lock):
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])

        # Skip category 5
        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

        local_results[k].append(
            {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": category,
                "bleu_score": bleu_scores["bleu1"],
                "f1_score": metrics["f1"],
                "llm_score": llm_score,
            }
        )

        # 更新进度条（线程安全）
        with pbar_lock:
            pbar.update(1)

    return local_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    parser.add_argument(
        "--input_file", type=str, default="results/rag_results_500_k1.json", help="Path to the input dataset file"
    )
    parser.add_argument(
        "--output_file", type=str, default="evaluation_metrics.json", help="Path to save the evaluation results"
    )
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads")

    args = parser.parse_args()

    size_mb = os.path.getsize(args.input_file) / (1024 * 1024)
    print(f"开始读取输入文件: {args.input_file} (~{size_mb:.2f} MB)", flush=True)
    with open(args.input_file, "r") as f:
        data = json.load(f)
    print(f"读取完成: {len(data)} 个会话块", flush=True)

    results = defaultdict(list)
    results_lock = threading.Lock()

    # 预计算总任务量用于进度显示（跳过 category 5）
    total_items = sum(len([i for i in v if str(i.get("category")) != "5"]) for v in data.values())
    print(f"待评估问答数: {total_items}", flush=True)

    pbar_lock = threading.Lock()
    with tqdm(total=total_items, desc="Evaluating", unit="qa") as pbar:
        # Use ThreadPoolExecutor with specified workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_item, item_data, pbar, pbar_lock) for item_data in data.items()]

            for future in concurrent.futures.as_completed(futures):
                local_results = future.result()
                with results_lock:
                    for k, items in local_results.items():
                        results[k].extend(items)

    # Save results to JSON file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
