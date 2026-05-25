import json
import argparse
from pathlib import Path


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_idx}: {e}")

            data.append(item)
    return data


def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def compute_preference_strength(item, line_idx=None):
    chosen_scores = item.get("chosen_scores")
    rejected_scores = item.get("rejected_scores")

    if chosen_scores is None or rejected_scores is None:
        raise KeyError(f"Missing chosen_scores or rejected_scores at line {line_idx}")

    if len(chosen_scores) != len(rejected_scores):
        raise ValueError(
            f"Length mismatch at line {line_idx}: "
            f"len(chosen_scores)={len(chosen_scores)}, "
            f"len(rejected_scores)={len(rejected_scores)}"
        )

    if len(chosen_scores) == 0:
        raise ValueError(f"Empty score list at line {line_idx}")

    preference_strength = [
        float(c) - float(r)
        for c, r in zip(chosen_scores, rejected_scores)
    ]

    avg_preference_strength = sum(preference_strength) / len(preference_strength)

    item["preference_strength"] = preference_strength
    item["avg_preference_strength"] = avg_preference_strength

    return item


def flip_preference_label(item, swap_scores=True):
    item["chosen"], item["rejected"] = item["rejected"], item["chosen"]

    if swap_scores:
        item["chosen_scores"], item["rejected_scores"] = (
            item["rejected_scores"],
            item["chosen_scores"],
        )

        # 翻转后重新计算 preference_strength，使其与新的 chosen/rejected 一致
        preference_strength = [
            float(c) - float(r)
            for c, r in zip(item["chosen_scores"], item["rejected_scores"])
        ]
        item["preference_strength"] = preference_strength
        item["avg_preference_strength"] = (
            sum(preference_strength) / len(preference_strength)
        )

    item["flipped"] = True
    return item


def process_jsonl(input_path, output_path, swap_scores=True):
    data = read_jsonl(input_path)

    processed = []
    for idx, item in enumerate(data, start=1):
        item = compute_preference_strength(item, line_idx=idx)
        item["flipped"] = False
        processed.append(item)

    # 按平均偏好强度从小到大排序
    processed.sort(key=lambda x: x["avg_preference_strength"])

    # 找出平均偏好强度为负数的数据
    negative_indices = [
        idx for idx, item in enumerate(processed)
        if item["avg_preference_strength"] < 0
    ]

    # bottom 50%：负数样本中平均偏好强度最小的一半
    num_to_flip = len(negative_indices) // 2
    indices_to_flip = set(negative_indices[:num_to_flip])

    for idx in indices_to_flip:
        processed[idx] = flip_preference_label(processed[idx], swap_scores=swap_scores)

    write_jsonl(processed, output_path)

    print(f"Total samples: {len(processed)}")
    print(f"Negative samples: {len(negative_indices)}")
    print(f"Flipped samples: {num_to_flip}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input jsonl file path")
    parser.add_argument("--output", required=True, help="Output jsonl file path")
    parser.add_argument(
        "--no_swap_scores",
        action="store_true",
        help="Only swap chosen/rejected text, without swapping chosen_scores/rejected_scores",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    process_jsonl(
        input_path=input_path,
        output_path=output_path,
        swap_scores=not args.no_swap_scores,
    )


if __name__ == "__main__":
    main()
