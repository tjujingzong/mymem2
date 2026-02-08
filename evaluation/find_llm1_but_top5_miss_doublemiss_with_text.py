import argparse
import json
from typing import Any, Dict, List, Optional, Set, Tuple


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _collect_items(results_json: Any) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    if isinstance(results_json, dict):
        for _, arr in results_json.items():
            if isinstance(arr, list):
                for it in arr:
                    if isinstance(it, dict):
                        all_items.append(it)
    elif isinstance(results_json, list):
        for it in results_json:
            if isinstance(it, dict):
                all_items.append(it)
    return all_items


def _flatten_dia_ids_from_topk(topk: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for m in topk:
        if not isinstance(m, dict):
            continue
        dia_ids = m.get("dia_ids")
        if isinstance(dia_ids, list):
            out.extend([str(d) for d in dia_ids])
    return out


def _build_dia_to_original_conversation(results_items: List[Dict[str, Any]]) -> Dict[str, str]:
    dia_map: Dict[str, str] = {}

    def visit_mem(mem: Dict[str, Any]):
        dia_ids = mem.get("dia_ids")
        if not isinstance(dia_ids, list):
            return
        oc = mem.get("original_conversation")
        if not isinstance(oc, str) or not oc.strip():
            return
        for d in dia_ids:
            ds = str(d)
            if ds not in dia_map:
                dia_map[ds] = oc

    for it in results_items:
        for mem in it.get("speaker_1_memories", []) or []:
            if isinstance(mem, dict):
                visit_mem(mem)
        for mem in it.get("speaker_2_memories", []) or []:
            if isinstance(mem, dict):
                visit_mem(mem)

    return dia_map


def _topk_hit(memories: List[Dict[str, Any]], evidence: List[str], k: int) -> bool:
    ev = set(str(e) for e in (evidence or []))
    top = memories[:k] if isinstance(memories, list) else []
    for m in top:
        if not isinstance(m, dict):
            continue
        dia_ids = m.get("dia_ids")
        if isinstance(dia_ids, list):
            for d in dia_ids:
                if str(d) in ev:
                    return True
    return False


def _unique_preserve_order(xs: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _build_dia_to_oc_for_item(item: Dict[str, Any]) -> Dict[str, str]:
    dia_map: Dict[str, str] = {}

    def visit_mem(mem: Dict[str, Any]):
        dia_ids = mem.get("dia_ids")
        if not isinstance(dia_ids, list):
            return
        oc = mem.get("original_conversation")
        if not isinstance(oc, str) or not oc.strip():
            return
        for d in dia_ids:
            ds = str(d)
            if ds not in dia_map:
                dia_map[ds] = oc

    for mem in item.get("speaker_1_memories", []) or []:
        if isinstance(mem, dict):
            visit_mem(mem)
    for mem in item.get("speaker_2_memories", []) or []:
        if isinstance(mem, dict):
            visit_mem(mem)

    return dia_map


def _build_dia_to_text_from_locomo_conv(conv: Dict[str, Any]) -> Dict[str, str]:
    """把 locomo10.json 的单个 conversation 展开成 dia_id -> "Speaker: text"。"""
    dia_map: Dict[str, str] = {}
    if not isinstance(conv, dict):
        return dia_map
    for k, v in conv.items():
        if not isinstance(k, str) or not k.startswith("session_"):
            continue
        # 跳过 session_1_date_time 这类字段
        if k.endswith("_date_time"):
            continue
        if not isinstance(v, list):
            continue
        for turn in v:
            if not isinstance(turn, dict):
                continue
            dia_id = turn.get("dia_id")
            speaker = turn.get("speaker")
            text = turn.get("text")
            if isinstance(dia_id, str) and isinstance(text, str) and text.strip():
                if isinstance(speaker, str) and speaker.strip():
                    dia_map[dia_id] = f"{speaker}: {text}"
                else:
                    dia_map[dia_id] = text
    return dia_map


def _get_evidence_original_from_locomo(locomo_entry: Dict[str, Any], evidence: List[str]) -> List[str]:
    """从 locomo10 的 conversation 中取 evidence 对应的原始对话行（按 dia_id）。去重后返回。"""
    conv = locomo_entry.get("conversation") if isinstance(locomo_entry, dict) else None
    dia_to_text = _build_dia_to_text_from_locomo_conv(conv) if isinstance(conv, dict) else {}
    lines: List[str] = []
    for d in evidence or []:
        ds = str(d)
        lines.append(dia_to_text.get(ds, "[locomo conversation not found]") )
    return _unique_preserve_order(lines)


def main():
    parser = argparse.ArgumentParser(
        description="From raw jsons: pick llm_score==1 and double-miss (both speaker_1 and speaker_2 top-k miss evidence), then dump evidence (from locomo10) + top-k memories (with original_conversation)."
    )
    parser.add_argument("--results", required=True, help="Path to mem0_results_top_10-8b.json")
    parser.add_argument("--metrics", required=True, help="Path to evaluation_metrics_run-8b.json (contains llm_score)")
    parser.add_argument("--locomo", required=True, help="Path to evaluation/dataset/locomo10.json (for evidence original conversation)")
    parser.add_argument("--k", type=int, default=5, help="Top-k for miss check (default: 5)")
    parser.add_argument("--out", required=True, help="Output txt path")
    args = parser.parse_args()

    results_json = _load_json(args.results)
    results_items = _collect_items(results_json)

    locomo_json = _load_json(args.locomo)
    # locomo10.json 是 list，每个 entry 对应一个对话
    locomo_entries: List[Dict[str, Any]] = [e for e in locomo_json if isinstance(e, dict)] if isinstance(locomo_json, list) else []
    locomo_by_question: Dict[str, Dict[str, Any]] = {}
    for entry in locomo_entries:
        qa = entry.get("qa")
        if not isinstance(qa, list):
            continue
        for qait in qa:
            if isinstance(qait, dict) and isinstance(qait.get("question"), str):
                locomo_by_question[qait["question"]] = entry

    metrics_json = _load_json(args.metrics)
    metrics_items = _collect_items(metrics_json)
    llm_by_question: Dict[str, float] = {}
    for mi in metrics_items:
        q = mi.get("question")
        if not isinstance(q, str) or not q.strip():
            continue
        if "llm_score" not in mi:
            continue
        try:
            llm_by_question[q] = float(mi.get("llm_score"))
        except Exception:
            continue

    kept: List[Dict[str, Any]] = []
    for it in results_items:
        q = it.get("question")
        if not isinstance(q, str) or not q.strip():
            continue
        llm_score = llm_by_question.get(q)
        if llm_score is None or float(llm_score) != 1.0:
            continue

        evidence = it.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            continue

        sp1_mems = it.get("speaker_1_memories")
        sp2_mems = it.get("speaker_2_memories")
        sp1_mems = sp1_mems if isinstance(sp1_mems, list) else []
        sp2_mems = sp2_mems if isinstance(sp2_mems, list) else []

        sp1_hit = _topk_hit(sp1_mems, evidence, args.k)
        sp2_hit = _topk_hit(sp2_mems, evidence, args.k)
        if sp1_hit or sp2_hit:
            continue

        kept.append(it)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"Total double-miss cases: {len(kept)}\n")

        for idx, it in enumerate(kept, 1):
            # Build dia_id -> original_conversation map per item to avoid collision
            dia_to_oc = _build_dia_to_oc_for_item(it)

            q = _safe_str(it.get("question"))
            answer = _safe_str(it.get("answer"))
            response = _safe_str(it.get("response"))
            evidence = [str(e) for e in (it.get("evidence") or [])]

            f.write("=" * 80 + "\n")
            f.write(f"[#{idx}]\n")
            f.write(f"question: {q}\n")
            f.write(f"answer: {answer}\n")
            f.write(f"response: {response}\n")
            f.write(f"evidence_dia_ids: {json.dumps(evidence, ensure_ascii=False)}\n")

            # evidence 原始对话：严格从 locomo10.json 获取（按 dia_id -> speaker:text）
            f.write("\n[evidence_original_conversation_unique]\n")
            locomo_entry = locomo_by_question.get(q)
            if locomo_entry is None:
                ev_lines = ["[locomo entry not found for question]"]
            else:
                ev_lines = _get_evidence_original_from_locomo(locomo_entry, evidence)
            for line in ev_lines:
                f.write(line + "\n")

            def dump_topk(title: str, mems: List[Dict[str, Any]]):
                f.write(f"\n[{title}]\n")
                for rank, m in enumerate(mems[: args.k], 1):
                    if not isinstance(m, dict):
                        continue
                    mem_text = _safe_str(m.get("memory"))
                    dia_ids = m.get("dia_ids")
                    dia_ids_list = [str(d) for d in dia_ids] if isinstance(dia_ids, list) else []
                    dia_ids_list = _unique_preserve_order(dia_ids_list)
                    ocs = _unique_preserve_order([dia_to_oc.get(d, "[original_conversation not found]") for d in dia_ids_list])

                    f.write(f"\n- rank: {rank}\n")
                    f.write(f"  score: {_safe_str(m.get('score'))}\n")
                    f.write(f"  timestamp: {_safe_str(m.get('timestamp'))}\n")
                    f.write(f"  memory: {mem_text}\n")
                    f.write(f"  dia_ids: {json.dumps(dia_ids_list, ensure_ascii=False)}\n")
                    f.write("  original_conversation_unique:\n")
                    for oc in ocs:
                        f.write("    " + oc.replace("\n", "\\n") + "\n")

            dump_topk("speaker_1_topk_with_text", it.get("speaker_1_memories") or [])
            dump_topk("speaker_2_topk_with_text", it.get("speaker_2_memories") or [])

            f.write("\n")

    print(f"Saved: {args.out} (count={len(kept)})")


if __name__ == "__main__":
    main()

