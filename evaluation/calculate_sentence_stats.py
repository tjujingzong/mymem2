import json
import re

def _split_short_sentences(text: str):
    # 按中英文常见分隔符拆短句（包含换行）
    sentence_delimiters = re.compile(r"[\.,;!?。，；！？\n]")
    parts = sentence_delimiters.split(text)
    return [p.strip() for p in parts if p.strip()]


def _percentile(sorted_vals, p: float):
    """无 numpy 的 percentile：线性插值，p in [0, 100]"""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def calculate_sentence_stats(file_path: str, per_turn: bool = True):
    """统计短句数。

    - per_turn=True：统计每个 turn（每个小对话/每条 dia）的短句数量（你这次想要的）
    - per_turn=False：统计整段 conversation 拼接后的短句数量
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return

    counts = []
    skipped = 0

    for entry in data:
        conversation = entry.get("conversation", {})
        session_keys = sorted([k for k in conversation.keys() if k.startswith("session_")])

        if per_turn:
            for session_key in session_keys:
                session_dialogs = conversation.get(session_key)
                if not isinstance(session_dialogs, list):
                    continue
                for turn in session_dialogs:
                    if not isinstance(turn, dict):
                        continue
                    text = turn.get("text", "")
                    if not isinstance(text, str) or not text.strip():
                        skipped += 1
                        continue
                    counts.append(len(_split_short_sentences(text)))
        else:
            full_text = ""
            for session_key in session_keys:
                session_dialogs = conversation.get(session_key)
                if not isinstance(session_dialogs, list):
                    continue
                for turn in session_dialogs:
                    if isinstance(turn, dict) and isinstance(turn.get("text"), str):
                        full_text += turn["text"] + " "
            if not full_text.strip():
                skipped += 1
                continue
            counts.append(len(_split_short_sentences(full_text)))

    if not counts:
        print("No valid text entries found.")
        return

    n = len(counts)
    total = sum(counts)
    mean = total / n
    sorted_counts = sorted(counts)
    median = _percentile(sorted_counts, 50)
    q1 = _percentile(sorted_counts, 25)
    q3 = _percentile(sorted_counts, 75)
    mn = sorted_counts[0]
    mx = sorted_counts[-1]

    # std（总体标准差）
    var = sum((x - mean) ** 2 for x in counts) / n
    std = var ** 0.5

    scope = "per-turn" if per_turn else "per-conversation"
    print(f"--- Short Sentence Statistics ({scope}) for locomo10 ---")
    print(f"Total items analyzed: {n}")
    print(f"Skipped items (no/invalid text): {skipped}")
    print(f"Total short sentences: {total}")
    print("\n--- Distribution ---")
    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"Median: {median}")
    print(f"Min: {mn}")
    print(f"Max: {mx}")
    print(f"Q1 (25%): {q1}")
    print(f"Q3 (75%): {q3}")


if __name__ == "__main__":
    dataset_path = "dataset/locomo10.json"
    # 默认按每条 dia/turn 统计（你截图那种）
    calculate_sentence_stats(dataset_path, per_turn=True)

