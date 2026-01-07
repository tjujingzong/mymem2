#!/bin/bash

# 脚本：依次执行实验、评估和生成分数的完整流程（执行一遍）
# 使用方法: bash run_once.sh 或 ./run_once.sh

set -e  # 如果任何命令失败，立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==========================================
# 配置区域 - 可根据需要修改以下参数
# ==========================================

# MEM0_VECTOR_PATH（向量存储路径）
VECTOR_PATH="/root/ljz/mymem2/evaluation/local_mem2/faiss_8b"

# 输出文件夹
OUTPUT_SEARCH="results_search"
OUTPUT_METRICS="results_metrics"
OUTPUT_SCORES="results_scores"

# 输出文件名配置
SEARCH_FILENAME="mem0_results_top_10-8b.json"
METRICS_FILENAME="evaluation_metrics_run-8b.json"
SCORES_FILENAME="scores_output_run-8b.txt"

# run_experiments.py 参数配置
TECHNIQUE_TYPE="mem0"
TOP_K=10

# ==========================================
# 脚本执行部分
# ==========================================

# 创建输出文件夹
mkdir -p "$OUTPUT_SEARCH"
mkdir -p "$OUTPUT_METRICS"
mkdir -p "$OUTPUT_SCORES"

echo "=========================================="
echo "开始执行完整实验流程（执行1次）"
echo "=========================================="
echo ""

echo "使用 MEM0_VECTOR_PATH: $VECTOR_PATH"
echo ""

# 步骤1: 运行实验 - add方法（使用对应的 MEM0_VECTOR_PATH）
echo "[步骤1] 执行: python run_experiments.py --technique_type ${TECHNIQUE_TYPE} --method add"
echo "环境变量: MEM0_VECTOR_PATH=$VECTOR_PATH"
MEM0_VECTOR_PATH="$VECTOR_PATH" python run_experiments.py --technique_type "$TECHNIQUE_TYPE" --method add
if [ $? -eq 0 ]; then
    echo "✓ 步骤1完成"
else
    echo "✗ 步骤1失败"
    exit 1
fi
echo ""

# 步骤2: 运行实验 - search方法（使用对应的 MEM0_VECTOR_PATH，输出到 results_search/）
SEARCH_OUTPUT_FILE="${OUTPUT_SEARCH}/${SEARCH_FILENAME}"
echo "[步骤2] 执行: python run_experiments.py --technique_type ${TECHNIQUE_TYPE} --method search --top_k ${TOP_K} --output_folder ${OUTPUT_SEARCH}/"
echo "环境变量: MEM0_VECTOR_PATH=$VECTOR_PATH"
MEM0_VECTOR_PATH="$VECTOR_PATH" python run_experiments.py --technique_type "$TECHNIQUE_TYPE" --method search --top_k "$TOP_K" --output_folder "${OUTPUT_SEARCH}/"
if [ $? -eq 0 ]; then
    echo "✓ 步骤2完成"
    # 重命名输出文件以区分不同运行
    DEFAULT_SEARCH_FILE="${OUTPUT_SEARCH}/mem0_results_top_10_filter_False_graph_False.json"
    if [ -f "$DEFAULT_SEARCH_FILE" ]; then
        mv "$DEFAULT_SEARCH_FILE" "$SEARCH_OUTPUT_FILE"
    fi
else
    echo "✗ 步骤2失败"
    exit 1
fi
echo ""

# 步骤3: 运行评估（输出到 results_metrics/）
METRICS_OUTPUT_FILE="${OUTPUT_METRICS}/${METRICS_FILENAME}"
echo "[步骤3] 执行: python evals.py --input_file ${SEARCH_OUTPUT_FILE} --output_file ${METRICS_OUTPUT_FILE}"
python evals.py --input_file "$SEARCH_OUTPUT_FILE" --output_file "$METRICS_OUTPUT_FILE"
if [ $? -eq 0 ]; then
    echo "✓ 步骤3完成"
else
    echo "✗ 步骤3失败"
    exit 1
fi
echo ""

# 步骤4: 生成分数（输出到 results_scores/）
SCORES_OUTPUT_FILE="${OUTPUT_SCORES}/${SCORES_FILENAME}"
echo "[步骤4] 执行: python generate_scores.py ${METRICS_OUTPUT_FILE} -o ${SCORES_OUTPUT_FILE}"
python generate_scores.py "$METRICS_OUTPUT_FILE" -o "$SCORES_OUTPUT_FILE"
if [ $? -eq 0 ]; then
    echo "✓ 步骤4完成"
else
    echo "✗ 步骤4失败"
    exit 1
fi
echo ""

echo "=========================================="
echo "执行完成！"
echo "结果保存在以下文件："
echo "  - Search结果: ${SEARCH_OUTPUT_FILE}"
echo "  - Metrics结果: ${METRICS_OUTPUT_FILE}"
echo "  - Scores结果: ${SCORES_OUTPUT_FILE}"
echo "=========================================="

