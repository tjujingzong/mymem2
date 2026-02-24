#!/usr/bin/env python3

import argparse
import json
from typing import Any, Dict, List, Tuple


def _parse_time_place(raw_time: Any, raw_place: Any) -> Tuple[str, str]:
    t = "" if raw_time is None else str(raw_time)
    p = "" if raw_place is None else str(raw_place)
    return t, p


def _build_conversation(message_list: Any, speaker_a: str, speaker_b: str) -> Tuple[Dict[str, Any], Dict[int, str]]:
    """Build locomo conversation and a global mid->dia_id map.

    message_list: expected as List[List[Dict]] where inner list is a session.

    Returns:
      - conversation dict
      - mid_to_dia_id mapping across all sessions
    """
    conversation: Dict[str, Any] = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
    }

    mid_to_dia: Dict[int, str] = {}

    if not isinstance(message_list, list):
        return conversation, mid_to_dia

    session_num = 1
    for session in message_list:
        if not isinstance(session, list):
            continue

        session_key = f"session_{session_num}"
        date_time_key = f"{session_key}_date_time"

        chats: List[Dict[str, Any]] = []
        session_date_time = ""

        for turn_idx, msg in enumerate(session):
            if not isinstance(msg, dict):
                continue

            dia_id = f"D{session_num}:{len(chats) + 1}"

            mid = msg.get("mid")
            try:
                mid_int = int(mid) if mid is not None else None
            except Exception:
                mid_int = None

            user_text = msg.get("user", "")
            assistant_text = msg.get("assistant", "")
            raw_time, raw_place = msg.get("time"), msg.get("place")
            t, p = _parse_time_place(raw_time, raw_place)
            if not session_date_time and t:
                session_date_time = t

            if user_text is not None and str(user_text) != "":
                chats.append({
                    "dia_id": dia_id,
                    "speaker": speaker_a,
                    "text": str(user_text),
                })
                if mid_int is not None:
                    mid_to_dia[mid_int] = dia_id

            if assistant_text is not None and str(assistant_text) != "":
                dia_id_assistant = f"D{session_num}:{len(chats) + 1}"
                chats.append({
                    "dia_id": dia_id_assistant,
                    "speaker": speaker_b,
                    "text": str(assistant_text),
                })

            # place currently not represented in locomo format; keep it unused.
            _ = p

        conversation[session_key] = chats
        if session_date_time:
            conversation[date_time_key] = session_date_time

        session_num += 1

    return conversation, mid_to_dia


def _extract_questions(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    qs = obj.get("questions", [])
    if isinstance(qs, list):
        return [q for q in qs if isinstance(q, dict)]
    return []


def _build_qa_items(questions: List[Dict[str, Any]], mid_to_dia: Dict[int, str], category: str) -> List[Dict[str, Any]]:
    qa_items: List[Dict[str, Any]] = []

    for q in questions:
        question = q.get("question", "")
        answer = q.get("answer", "")

        evidence: List[str] = []
        raw_mid = q.get("mid")
        if raw_mid is not None:
            try:
                mid_int = int(raw_mid)
                if mid_int in mid_to_dia:
                    evidence = [mid_to_dia[mid_int]]
            except Exception:
                evidence = []

        qa_items.append({
            "question": "" if question is None else str(question),
            "answer": "" if answer is None else str(answer),
            "evidence": evidence,
            "category": "" if category is None else str(category),
        })

    return qa_items


def convert_firstagent_to_locomo(input_path: str, output_path: str, limit: int = 0):
    print(f"正在读取输入文件: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted: List[Dict[str, Any]] = []

    # input is expected as: {Category: {Category: [ {tid, message_list, questions, ...}, ...] }, ...}
    for top_k, top_v in (data.items() if isinstance(data, dict) else []):
        if not isinstance(top_v, dict):
            continue

        # prefer the list under same key; fallback to first list value
        items = top_v.get(top_k)
        if items is None:
            for vv in top_v.values():
                if isinstance(vv, list):
                    items = vv
                    break

        if not isinstance(items, list):
            continue

        for item_idx, obj in enumerate(items):
            if limit and len(converted) >= limit:
                break
            if not isinstance(obj, dict):
                continue

            tid = obj.get("tid", item_idx)
            q_id = f"{top_k}_{tid}"

            speaker_a = f"user_{q_id}"
            speaker_b = f"assistant_{q_id}"

            conversation, mid_to_dia = _build_conversation(obj.get("message_list"), speaker_a, speaker_b)
            questions = _extract_questions(obj)
            qa_items = _build_qa_items(questions, mid_to_dia, category=top_k)

            converted.append({
                "conversation": conversation,
                "qa": qa_items,
                "question_id": q_id,
                "category": top_k,
                "tid": tid,
            })

        if limit and len(converted) >= limit:
            break

    print(f"正在保存转换后的数据到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print("✓ 转换完成！")
    print(f"  - 输出条目数: {len(converted)}")
    print(f"  - 输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="将 FirstAgentDataHighLevel.json 转换为 locomo 格式")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/FirstAgentDataHighLevel.json",
        help="输入文件路径（默认: dataset/FirstAgentDataHighLevel.json）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/FirstAgentDataHighLevel_as_locomo.json",
        help="输出文件路径（默认: dataset/FirstAgentDataHighLevel_as_locomo.json）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅转换前 N 个 entry（默认: 0 表示全部）",
    )

    args = parser.parse_args()
    convert_firstagent_to_locomo(args.input, args.output, limit=args.limit)


if __name__ == "__main__":
    main()

