import os
import re
import csv
from collections import defaultdict


def _iter_txt_files(directory: str):
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith('.txt'):
                yield os.path.join(root, name)


def _parse_overall_mean_scores(lines):
    """从 txt 文本中解析 Overall Mean Scores 区块里的 key/value。"""
    scores = {}
    in_overall = False

    # 允许：key + 多空格 + 数字（支持科学计数法）
    num_re = r'[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?'
    kv_re = re.compile(rf'^([A-Za-z0-9_]+)\s+({num_re})\s*$')

    for raw in lines:
        line = raw.rstrip('\n')

        if line.strip() == 'Overall Mean Scores:':
            in_overall = True
            continue

        if not in_overall:
            continue

        if not line.strip():
            break
        if line.strip().startswith('dtype:'):
            break

        m = kv_re.match(line.strip())
        if m:
            k, v = m.group(1), m.group(2)
            scores[k] = float(v)

    return scores


def summarize_scores(directory: str):
    all_rows = []
    all_keys = set()

    for path in _iter_txt_files(directory):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"读取文件失败: {path} ({e})")
            continue

        scores = _parse_overall_mean_scores(lines)
        if not scores:
            # 没有 Overall Mean Scores 区块则跳过
            continue

        row = {'filename': os.path.basename(path)}
        row.update(scores)
        all_keys.update(scores.keys())
        all_rows.append(row)

    if not all_rows:
        print("未发现包含 'Overall Mean Scores:' 的有效 txt 文件。")
        return

    keys = sorted(all_keys)

    sums = defaultdict(float)
    counts = defaultdict(int)
    for row in all_rows:
        for k in keys:
            if k in row:
                sums[k] += row[k]
                counts[k] += 1

    print("\n" + "=" * 60)
    print(f"数据汇总自目录: {directory}")
    print(f"处理有效文件数: {len(all_rows)}")
    print("=" * 60)
    print("\n各指标平均值 (Mean of Overall Mean Scores):")
    for k in keys:
        if counts[k]:
            print(f"{k:<30} {sums[k] / counts[k]:>12.4f}")
    print("\n" + "=" * 60)

    output_csv = os.path.join(os.path.dirname(directory), "longmemeval_summary.csv")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename'] + keys)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        print(f"逐文件明细已导出至: {output_csv}")
    except Exception as e:
        print(f"导出 CSV 失败: {e}")


if __name__ == "__main__":
    target_dir = "/root/ljz/mymem2/evaluation/results_scores/longmemeavl-simple"
    if os.path.exists(target_dir):
        summarize_scores(target_dir)
    else:
        print(f"错误: 目录 {target_dir} 不存在。")
