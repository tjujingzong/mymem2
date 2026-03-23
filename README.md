 mymem实验

 python3 evaluation/find_llm1_but_top5_miss_doublemiss_with_text.py \
  --results evaluation/results_search/mem0_results_top_10-8b.json \
  --metrics evaluation/results_metrics/evaluation_metrics_run-8b.json \
  --locomo evaluation/dataset/locomo10.json \
  --k 5 \
  --out evaluation/results_search/llm1_doublemiss_with_text.txt

  # 默认 facts
bash evaluation/run_once.sh

# 纯短句模式
USE_SENTENCE_MODE=1 bash evaluation/run_once.sh

# Hybrid-Sentence 模式
USE_HYBRID_MODE=1 bash evaluation/run_once.sh

# 最简单模式（不写入/回溯对话ID，不拼原文，不做RRF重排，仅依赖memory）
USE_SIMPLE_MODE=1 bash evaluation/run_once.sh

# 按 user_id 分索引模式（解决多用户场景下 top_k 返回不足的问题）
# 注意：需要重新运行 add 阶段，因为索引结构会改变
MEM0_FAISS_PER_USER_INDEX=1 bash evaluation/run_once.sh

# 指针化 + 多检索器 RRF 融合模式（在 Mem0 语义检索上，再叠加基于原文指针的关键词检索）
# 默认关闭以保持兼容，按需打开：
MEM0_KEYWORD_RRF_FUSION=1 bash evaluation/run_once.sh

| 模式 | USE_SENTENCE_MODE | USE_HYBRID_MODE | USE_SIMPLE_MODE | MEM0_KEYWORD_RRF_FUSION | 效果 |
|------|------------------|-----------------|-----------------|--------------------------|-------|
| 默认 facts | 0 | 0 | 0 | 0 | 与之前完全一致 |
| 纯短句 | 1 | 0 | 0 | 0 | 只入库短句，QA 时按短句检索 |
| Hybrid-Sentence | 0 | 1 | 0 | 0 | 先按 facts/summary 检索 → 拿到原对话 → 对话内再做短句相似度 → 仅把「前一半」短句写入 Prompt |
| 最简单 | 0/1 | 0/1 | 1 | 0 | 不写入/回溯对话ID、不拼原文、不做RRF重排，仅依赖memory |
| 指针化 RRF 融合 | 0/1 | 0/1 | 0 | 1 | 在 Mem0 语义检索排序基础上，引入“原文指针片段”的关键词/时间检索，并用 RRF 做多检索器融合 |


USE_HYBRID_MODE=1 bash run_once.sh


  {
    "question_id": "object",
    "question_type": "object",
    "question": "object",
    "question_date": "object",
    "answer": "object",
    "answer_session_ids": "object",
    "haystack_dates": "object",
    "haystack_session_ids": "object",
    "haystack_sessions": "object"
  },


  python convert_longmemeval.py --input dataset/longmemeval_s_cleaned.json     --output dataset/longmemeval_as_locomo_10.json     --limit 10


  python3 evaluation/convert_firstagent_to_locomo.py \
  --input evaluation/dataset/FirstAgentDataHighLevel.json \
  --output evaluation/dataset/FirstAgentDataHighLevel_as_locomo.json

python3 evaluation/summarize_scores.py

python evaluation/run_experiments.py --technique_type mem0 --method add --is_graph

bash evaluation/run_once.sh --from_step 2

docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:latest


export MEM0_LOCAL_MODE=1
export ENABLE_GRAPH_STORE=1

export NEO4J_URL=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=password123
export NEO4J_DATABASE=neo4j

MEM0_KEYWORD_RRF_FUSION=1 MEM0_RRF_K=30 bash evaluation/run_once.sh

MEM0_KEYWORD_RRF_FUSION=1 bash evaluation/run_once.sh


MEM0_FAISS_PER_USER_INDEX=1 time bash run_once.sh --only_step 1

USE_HYBRID_MODE=1 MEM0_LOCAL_MODE=1 MEM0_FAISS_PER_USER_INDEX=1 MEM0_ADD_BATCH_SIZE=8 bash evaluation/run_once-longmemeval.sh --from_step 2