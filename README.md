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

| 模式 | USE_SENTENCE_MODE | USE_HYBRID_MODE | 效果 |
|------|------------------|-----------------|-------|
| 默认 facts | 0 | 0 | 与之前完全一致 |
| 纯短句 | 1 | 0 | 只入库短句，QA 时按短句检索 |
| Hybrid-Sentence | 0 | 1 | 先按 facts/summary 检索 → 拿到原对话 → 对话内再做短句相似度 → 仅把「前一半」短句写入 Prompt |


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