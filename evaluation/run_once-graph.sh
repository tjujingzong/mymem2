set -e  # 如果任何命令失败，立即退出
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 是否启用“LLM 动态评估（按需指针回溯原文）”
# - 兼容性策略：默认 0，完全不改变原有流程；设为 1 才会切到新的 llm_evals.py
USE_LLM_DYNAMIC_EVAL="${USE_LLM_DYNAMIC_EVAL:-0}"

# 是否启用短句模式（按标点切分文本而非提取facts）
# - 默认 0，使用原有的facts提取模式；设为 1 则使用短句模式
USE_SENTENCE_MODE="${USE_SENTENCE_MODE:-0}"

# 是否启用 Hybrid-Sentence 模式（facts 检索 + 对话内短句再检索 + 仅前半句写入 Prompt）
# - 默认 0；设为 1 则启用
USE_HYBRID_MODE="${USE_HYBRID_MODE:-0}"

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

# 输出文件夹
OUTPUT_SEARCH="results_search/gragh"
OUTPUT_METRICS="results_metrics/gragh"
OUTPUT_SCORES="results_scores/gragh"

# MEM0_VECTOR_PATH（向量存储路径）
VECTOR_PATH="/root/ljz/mymem2/evaluation/local_gragh"

# 数据集路径（默认保持 locomo10.json 不变；如需 longmemeval 请覆盖该变量）
DATASET_PATH="${DATASET_PATH:-dataset/locomo10.json}"

# 输出文件名配置
SEARCH_FILENAME="gragh.json"
METRICS_FILENAME="gragh.json"
SCORES_FILENAME="gragh.txt"

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
if [ "$FROM_STEP" -le 1 ]; then
    echo "[步骤1] 执行: python run_experiments.py --technique_type ${TECHNIQUE_TYPE} --method add --is_graph"
    echo "环境变量: MEM0_VECTOR_PATH=$VECTOR_PATH"
    MEM0_VECTOR_PATH="$VECTOR_PATH" DATASET_PATH="$DATASET_PATH" USE_SENTENCE_MODE="$USE_SENTENCE_MODE" USE_HYBRID_MODE="$USE_HYBRID_MODE" python run_experiments.py --technique_type "$TECHNIQUE_TYPE" --method add --is_graph --data_path "$DATASET_PATH"
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
    # 预先定义输出文件路径
    SEARCH_OUTPUT_FILE="${OUTPUT_SEARCH}/${SEARCH_FILENAME}"
    
    echo "[步骤2] 执行: python run_experiments.py --technique_type ${TECHNIQUE_TYPE} --method search --is_graph --top_k ${TOP_K} --output_folder ${OUTPUT_SEARCH}/"
    echo "环境变量: MEM0_VECTOR_PATH=$VECTOR_PATH"
    MEM0_VECTOR_PATH="$VECTOR_PATH" DATASET_PATH="$DATASET_PATH" USE_SENTENCE_MODE="$USE_SENTENCE_MODE" USE_HYBRID_MODE="$USE_HYBRID_MODE" python run_experiments.py --technique_type "$TECHNIQUE_TYPE" --method search --is_graph --top_k "$TOP_K" --output_folder "${OUTPUT_SEARCH}/" --data_path "$DATASET_PATH"
    if [ $? -eq 0 ]; then
        echo "✓ 步骤2完成"
        # 重命名输出文件以区分不同运行
        DEFAULT_SEARCH_FILE="${OUTPUT_SEARCH}/mem0_results_top_${TOP_K}_filter_False_graph_True.json"
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

# 预先定义输出文件路径（用于从任意步骤开始时的依赖）
SEARCH_OUTPUT_FILE="${OUTPUT_SEARCH}/${SEARCH_FILENAME}"
METRICS_OUTPUT_FILE="${OUTPUT_METRICS}/${METRICS_FILENAME}"
SCORES_OUTPUT_FILE="${OUTPUT_SCORES}/${SCORES_FILENAME}"

# 步骤3: 运行评估（输出到 results_metrics/）
if [ "$FROM_STEP" -le 3 ]; then
    if [ ! -f "$SEARCH_OUTPUT_FILE" ]; then
        echo "✗ 找不到搜索结果文件: $SEARCH_OUTPUT_FILE"
        echo "  你选择从步骤3开始运行，但步骤3依赖步骤2的输出。"
        exit 1
    fi

    if [ "$USE_LLM_DYNAMIC_EVAL" = "1" ] || [ "$USE_LLM_DYNAMIC_EVAL" = "true" ] || [ "$USE_LLM_DYNAMIC_EVAL" = "TRUE" ]; then
        echo "[步骤3] 执行: python llm_evals.py --input_file ${SEARCH_OUTPUT_FILE} --output_file ${METRICS_OUTPUT_FILE} --dataset_file ${DATASET_PATH}"
        echo "启用 LLM 动态评估（按需指针回溯原文）。如需指定评估模型，可设置 EVAL_MODEL。"
        python llm_evals.py --input_file "$SEARCH_OUTPUT_FILE" --output_file "$METRICS_OUTPUT_FILE" --dataset_file "$DATASET_PATH"
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
