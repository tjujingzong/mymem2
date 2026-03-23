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

# mem0 add 阶段 batch size（每次提取多少条消息）
# - 默认 2；可通过环境变量覆盖，例如设为 8 表示约提升到原来的 4 倍
MEM0_ADD_BATCH_SIZE="${MEM0_ADD_BATCH_SIZE:-2}"

# 是否启用最简单模式（只使用 memory，不写入/回溯对话ID，不做RRF重排）
# - 默认 0；设为 1 则启用
USE_SIMPLE_MODE="${USE_SIMPLE_MODE:-0}"

# 从指定步骤开始运行（1-4）
# 用法示例：
#   bash run_once2.sh --from_step 2
# 或环境变量：
#   FROM_STEP=2 bash run_once2.sh
FROM_STEP="${FROM_STEP:-1}"
if [ "$1" = "--from_step" ] && [ -n "$2" ]; then
    FROM_STEP="$2"
fi

if ! [[ "$FROM_STEP" =~ ^[1-4]$ ]]; then
    echo "✗ 参数错误：FROM_STEP 必须是 1-4，当前为: $FROM_STEP"
    exit 1
fi

# 输出文件夹
OUTPUT_SEARCH="results_search/longmemeavl-index-size2"
OUTPUT_METRICS="results_metrics/longmemeavl-index-size2"
OUTPUT_SCORES="results_scores/longmemeavl-index-size2"

# 批量运行配置：SUFFIX_LIST 表示要跑哪些后缀（如 10,20,...,500）
# - 可覆盖：SUFFIX_LIST="10 20 30" bash run_once2.sh
SUFFIX_LIST="${SUFFIX_LIST:-"$(seq 10 10 500)"}"

# MEM0_VECTOR_PATH（向量存储路径）根目录（每个后缀会拼到该目录下）
VECTOR_BASE_PATH="${VECTOR_BASE_PATH:-/root/ljz/mymem2/evaluation/local_longmemeval-index-size8}"

# 数据集根目录（每个后缀会拼到该目录下）
DATASET_BASE_DIR="${DATASET_BASE_DIR:-dataset/longmemeval}"

# 输出文件名前缀
OUTPUT_PREFIX="${OUTPUT_PREFIX:-longmemeval_facts_}"

# run_experiments.py 参数配置
TECHNIQUE_TYPE="mem0"
TOP_K=10

# ==========================================
# 脚本执行部分
# ==========================================

run_one_suffix() {
    local suffix="$1"

    local vector_path dataset_path search_filename metrics_filename scores_filename

    vector_path="${VECTOR_BASE_PATH}/${suffix}"
    dataset_path="${DATASET_BASE_DIR}/longmemeval_as_locomo_${suffix}.json"

    search_filename="${OUTPUT_PREFIX}${suffix}.json"
    metrics_filename="${OUTPUT_PREFIX}${suffix}.json"
    scores_filename="${OUTPUT_PREFIX}${suffix}.txt"

    VECTOR_PATH="${VECTOR_PATH:-$vector_path}"
    DATASET_PATH="${DATASET_PATH:-$dataset_path}"
    SEARCH_FILENAME="${SEARCH_FILENAME:-$search_filename}"
    METRICS_FILENAME="${METRICS_FILENAME:-$metrics_filename}"
    SCORES_FILENAME="${SCORES_FILENAME:-$scores_filename}"

    # 创建输出文件夹
    mkdir -p "$OUTPUT_SEARCH"
    mkdir -p "$OUTPUT_METRICS"
    mkdir -p "$OUTPUT_SCORES"

    echo "=========================================="
    echo "开始执行完整实验流程（suffix=${suffix}）"
    echo "=========================================="
    echo ""

    echo "使用 MEM0_VECTOR_PATH: $VECTOR_PATH"
    echo "使用 DATASET_PATH: $DATASET_PATH"
    echo ""

    # 步骤1: 运行实验 - add方法（使用对应的 MEM0_VECTOR_PATH）
    if [ "$FROM_STEP" -le 1 ]; then
        echo "[步骤1] 执行: python run_experiments.py --technique_type ${TECHNIQUE_TYPE} --method add"
        echo "环境变量: MEM0_VECTOR_PATH=$VECTOR_PATH"
        MEM0_VECTOR_PATH="$VECTOR_PATH" DATASET_PATH="$DATASET_PATH" USE_SENTENCE_MODE="$USE_SENTENCE_MODE" USE_HYBRID_MODE="$USE_HYBRID_MODE" USE_SIMPLE_MODE="$USE_SIMPLE_MODE" MEM0_ADD_BATCH_SIZE="$MEM0_ADD_BATCH_SIZE" python run_experiments.py --technique_type "$TECHNIQUE_TYPE" --method add --data_path "$DATASET_PATH" --add_batch_size "$MEM0_ADD_BATCH_SIZE"
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

        echo "[步骤2] 执行: python run_experiments.py --technique_type ${TECHNIQUE_TYPE} --method search --top_k ${TOP_K} --output_folder ${OUTPUT_SEARCH}/"
        echo "环境变量: MEM0_VECTOR_PATH=$VECTOR_PATH"
        MEM0_VECTOR_PATH="$VECTOR_PATH" DATASET_PATH="$DATASET_PATH" USE_SENTENCE_MODE="$USE_SENTENCE_MODE" USE_HYBRID_MODE="$USE_HYBRID_MODE" USE_SIMPLE_MODE="$USE_SIMPLE_MODE" python run_experiments.py --technique_type "$TECHNIQUE_TYPE" --method search --top_k "$TOP_K" --output_folder "${OUTPUT_SEARCH}/" --data_path "$DATASET_PATH"
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
}

# 循环执行所有后缀
for s in $SUFFIX_LIST; do
    # 使用子 shell 运行，避免环境变量相互污染
    ( run_one_suffix "$s" )
done
