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

# 是否在 QA 生成阶段把 original conversations 拼进 prompt
# - 1: 拼进（兼容旧行为，但 prompt_tokens 更高）
# - 0: 不拼（更省 token）
MEM0_INCLUDE_ORIGINAL_CONVERSATIONS="${MEM0_INCLUDE_ORIGINAL_CONVERSATIONS:-1}"

# 是否启用“两阶段按需加载原文”
# - 0: 关闭（默认，完全不改变旧行为）
# - 1: 开启：先不带原文生成；若模型判定不确定/需证据，再加载原文重试
MEM0_QA_TWO_STAGE="${MEM0_QA_TWO_STAGE:-0}"

# 我建议把语义明确成下面这套规则（当前代码基本已做到）：
# 规则 1：MEM0_QA_TWO_STAGE 优先级最高
# MEM0_QA_TWO_STAGE=1：启用按需加载（Stage1 不带原文，必要时 Stage2 才带）
# MEM0_QA_TWO_STAGE=0：关闭按需加载，走单阶段
# 规则 2：MEM0_INCLUDE_ORIGINAL_CONVERSATIONS 只在单阶段时生效
# 当 MEM0_QA_TWO_STAGE=0：
# MEM0_INCLUDE_ORIGINAL_CONVERSATIONS=1：总是带原文（旧行为）
# MEM0_INCLUDE_ORIGINAL_CONVERSATIONS=0：从不带原文（省 token，但可能影响正确率）


# 输出文件夹
OUTPUT_SEARCH="results_search"
OUTPUT_METRICS="results_metrics"
OUTPUT_SCORES="results_scores"

# 从指定步骤开始运行（1-4）
# 用法示例：
#   bash run_once.sh --from_step 2
# 或环境变量：
#   FROM_STEP=2 bash run_once.sh
FROM_STEP="${FROM_STEP:-1}"
if [ "$1" = "--from_step" ] && [ -n "$2" ]; then
    FROM_STEP="$2"
fi

if ! [[ "$FROM_STEP" =~ ^[1-4]$ ]]; then
    echo "✗ 参数错误：FROM_STEP 必须是 1-4，当前为: $FROM_STEP"
    exit 1
fi

# 输出文件名配置
SEARCH_FILENAME="mem0_results_top_10-8b-2.json"
METRICS_FILENAME="evaluation_metrics_run-8b-2.json"
SCORES_FILENAME="scores_output_run-8b-2.txt"

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

# 预先定义输出文件路径（用于从任意步骤开始时的依赖）
SEARCH_OUTPUT_FILE="${OUTPUT_SEARCH}/${SEARCH_FILENAME}"
METRICS_OUTPUT_FILE="${OUTPUT_METRICS}/${METRICS_FILENAME}"
SCORES_OUTPUT_FILE="${OUTPUT_SCORES}/${SCORES_FILENAME}"

# 步骤1: 运行实验 - add方法（使用对应的 MEM0_VECTOR_PATH）
if [ "$FROM_STEP" -le 1 ]; then
    echo "[步骤1] 执行: python run_experiments.py --technique_type ${TECHNIQUE_TYPE} --method add"
    echo "环境变量: MEM0_VECTOR_PATH=$VECTOR_PATH"
    MEM0_VECTOR_PATH="$VECTOR_PATH" MEM0_INCLUDE_ORIGINAL_CONVERSATIONS="$MEM0_INCLUDE_ORIGINAL_CONVERSATIONS" MEM0_QA_TWO_STAGE="$MEM0_QA_TWO_STAGE" python run_experiments.py --technique_type "$TECHNIQUE_TYPE" --method add
    if [ $? -eq 0 ]; then
        echo "✓ 步骤1完成"
    else
        echo "✗ 步骤1失败"
        exit 1
    fi
    echo ""
else
    echo "[跳过步骤1] FROM_STEP=$FROM_STEP"
    echo ""
fi

# 步骤2: 运行实验 - search方法（使用对应的 MEM0_VECTOR_PATH，输出到 results_search/）
if [ "$FROM_STEP" -le 2 ]; then
    echo "[步骤2] 执行: python run_experiments.py --technique_type ${TECHNIQUE_TYPE} --method search --top_k ${TOP_K} --output_folder ${OUTPUT_SEARCH}/"
    echo "环境变量: MEM0_VECTOR_PATH=$VECTOR_PATH"
    MEM0_VECTOR_PATH="$VECTOR_PATH" MEM0_INCLUDE_ORIGINAL_CONVERSATIONS="$MEM0_INCLUDE_ORIGINAL_CONVERSATIONS" MEM0_QA_TWO_STAGE="$MEM0_QA_TWO_STAGE" python run_experiments.py --technique_type "$TECHNIQUE_TYPE" --method search --top_k "$TOP_K" --output_folder "${OUTPUT_SEARCH}/"
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
else
    echo "[跳过步骤2] FROM_STEP=$FROM_STEP"
    echo ""
fi

# 步骤3: 运行评估（输出到 results_metrics/）
if [ "$FROM_STEP" -le 3 ]; then
    if [ ! -f "$SEARCH_OUTPUT_FILE" ]; then
        echo "✗ 找不到搜索结果文件: $SEARCH_OUTPUT_FILE"
        echo "  你选择从步骤3开始运行，但步骤3依赖步骤2的输出。"
        exit 1
    fi

    if [ "$USE_LLM_DYNAMIC_EVAL" = "1" ] || [ "$USE_LLM_DYNAMIC_EVAL" = "true" ] || [ "$USE_LLM_DYNAMIC_EVAL" = "TRUE" ]; then
        echo "[步骤3] 执行: python llm_evals.py --input_file ${SEARCH_OUTPUT_FILE} --output_file ${METRICS_OUTPUT_FILE}"
        echo "启用 LLM 动态评估（按需指针回溯原文）。如需指定评估模型，可设置 EVAL_MODEL；如需指定数据集，可设置 EVAL_DATASET_FILE。"
        python llm_evals.py --input_file "$SEARCH_OUTPUT_FILE" --output_file "$METRICS_OUTPUT_FILE"
    else
        echo "[步骤3] 执行: python evals.py --input_file ${SEARCH_OUTPUT_FILE} --output_file ${METRICS_OUTPUT_FILE}"
        python evals.py --input_file "$SEARCH_OUTPUT_FILE" --output_file "$METRICS_OUTPUT_FILE"
    fi
    if [ $? -eq 0 ]; then
        echo "✓ 步骤3完成"
    else
        echo "✗ 步骤3失败"
        exit 1
    fi
    echo ""
else
    echo "[跳过步骤3] FROM_STEP=$FROM_STEP"
    echo ""
fi

# 步骤4: 生成分数（输出到 results_scores/）
if [ "$FROM_STEP" -le 4 ]; then
    if [ ! -f "$METRICS_OUTPUT_FILE" ]; then
        echo "✗ 找不到评估结果文件: $METRICS_OUTPUT_FILE"
        echo "  你选择从步骤4开始运行，但步骤4依赖步骤3的输出。"
        exit 1
    fi

    echo "[步骤4] 执行: python generate_scores.py ${METRICS_OUTPUT_FILE} -o ${SCORES_OUTPUT_FILE}"
    python generate_scores.py "$METRICS_OUTPUT_FILE" -o "$SCORES_OUTPUT_FILE"
    if [ $? -eq 0 ]; then
        echo "✓ 步骤4完成"
    else
        echo "✗ 步骤4失败"
        exit 1
    fi
    echo ""
else
    echo "[跳过步骤4] FROM_STEP=$FROM_STEP"
    echo ""
fi

echo "=========================================="
echo "执行完成！"
echo "结果保存在以下文件："
echo "  - Search结果: ${SEARCH_OUTPUT_FILE}"
echo "  - Metrics结果: ${METRICS_OUTPUT_FILE}"
echo "  - Scores结果: ${SCORES_OUTPUT_FILE}"
echo "=========================================="

