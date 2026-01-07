#!/bin/bash

# 脚本：依次执行实验、评估和生成分数的完整流程（执行三遍）
# 使用方法: bash run_all.sh 或 ./run_all.sh

set -e  # 如果任何命令失败，立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 定义三个不同的 MEM0_VECTOR_PATH（可以根据需要修改路径）
VECTOR_PATHS=(
    "/root/ljz/mymem2/evaluation/local_mem2/faiss_1"
    "/root/ljz/mymem2/evaluation/local_mem2/faiss_2"
    "/root/ljz/mymem2/evaluation/local_mem2/faiss_3"
)

# 定义输出文件夹
OUTPUT_SEARCH="results_search"
OUTPUT_METRICS="results_metrics"
OUTPUT_SCORES="results_scores"

# 创建输出文件夹
mkdir -p "$OUTPUT_SEARCH"
mkdir -p "$OUTPUT_METRICS"
mkdir -p "$OUTPUT_SCORES"

echo "=========================================="
echo "开始执行完整实验流程（将执行3次）"
echo "=========================================="
echo ""

# 循环执行3次完整流程
for run in 1 2 3; do
    echo "=========================================="
    echo "开始第 $run 次完整执行"
    echo "=========================================="
    echo ""
    
    # 获取对应的 MEM0_VECTOR_PATH
    VECTOR_PATH="${VECTOR_PATHS[$((run-1))]}"
    
    echo "使用 MEM0_VECTOR_PATH: $VECTOR_PATH"
    echo ""
    
    # 步骤1: 运行实验 - add方法（使用对应的 MEM0_VECTOR_PATH）
    echo "[第${run}次 - 步骤1] 执行: python run_experiments.py --technique_type mem0 --method add"
    echo "环境变量: MEM0_VECTOR_PATH=$VECTOR_PATH"
    MEM0_VECTOR_PATH="$VECTOR_PATH" python run_experiments.py --technique_type mem0 --method add
    if [ $? -eq 0 ]; then
        echo "✓ 第${run}次 - 步骤1完成"
    else
        echo "✗ 第${run}次 - 步骤1失败"
        exit 1
    fi
    echo ""
    
    # 步骤2: 运行实验 - search方法（使用对应的 MEM0_VECTOR_PATH，输出到 results_search/）
    SEARCH_OUTPUT_FILE="${OUTPUT_SEARCH}/mem0_results_top_10_filter_False_graph_False_run${run}.json"
    echo "[第${run}次 - 步骤2] 执行: python run_experiments.py --technique_type mem0 --method search --top_k 10 --output_folder ${OUTPUT_SEARCH}/"
    echo "环境变量: MEM0_VECTOR_PATH=$VECTOR_PATH"
    MEM0_VECTOR_PATH="$VECTOR_PATH" python run_experiments.py --technique_type mem0 --method search --top_k 10 --output_folder "${OUTPUT_SEARCH}/"
    if [ $? -eq 0 ]; then
        echo "✓ 第${run}次 - 步骤2完成"
        # 重命名输出文件以区分不同运行
        if [ -f "${OUTPUT_SEARCH}/mem0_results_top_10_filter_False_graph_False.json" ]; then
            mv "${OUTPUT_SEARCH}/mem0_results_top_10_filter_False_graph_False.json" "$SEARCH_OUTPUT_FILE"
        fi
    else
        echo "✗ 第${run}次 - 步骤2失败"
        exit 1
    fi
    echo ""
    
    # 步骤3: 运行评估（输出到 results_metrics/）
    METRICS_OUTPUT_FILE="${OUTPUT_METRICS}/evaluation_metrics_run${run}.json"
    echo "[第${run}次 - 步骤3] 执行: python evals.py --input_file ${SEARCH_OUTPUT_FILE} --output_file ${METRICS_OUTPUT_FILE}"
    python evals.py --input_file "$SEARCH_OUTPUT_FILE" --output_file "$METRICS_OUTPUT_FILE"
    if [ $? -eq 0 ]; then
        echo "✓ 第${run}次 - 步骤3完成"
    else
        echo "✗ 第${run}次 - 步骤3失败"
        exit 1
    fi
    echo ""
    
    # 步骤4: 生成分数（输出到 results_scores/）
    SCORES_OUTPUT_FILE="${OUTPUT_SCORES}/scores_output_run${run}.txt"
    echo "[第${run}次 - 步骤4] 执行: python generate_scores.py ${METRICS_OUTPUT_FILE} -o ${SCORES_OUTPUT_FILE}"
    python generate_scores.py "$METRICS_OUTPUT_FILE" -o "$SCORES_OUTPUT_FILE"
    if [ $? -eq 0 ]; then
        echo "✓ 第${run}次 - 步骤4完成"
    else
        echo "✗ 第${run}次 - 步骤4失败"
        exit 1
    fi
    echo ""
    
    echo "第 $run 次执行完成！"
    echo "  - Search结果: ${SEARCH_OUTPUT_FILE}"
    echo "  - Metrics结果: ${METRICS_OUTPUT_FILE}"
    echo "  - Scores结果: ${SCORES_OUTPUT_FILE}"
    echo ""
done

echo "=========================================="
echo "所有执行完成！"
echo "结果保存在以下文件夹："
echo "  - ${OUTPUT_SEARCH}/ (Search结果)"
echo "  - ${OUTPUT_METRICS}/ (Metrics结果)"
echo "  - ${OUTPUT_SCORES}/ (Scores结果)"
echo "=========================================="

