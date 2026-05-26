import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any


ScoreKey = Tuple[str, str, str]


def read_score_jsonl_dir(score_dir: str) -> Dict[ScoreKey, List[float]]:
    """
    遍历 score_dir 下所有 .jsonl 文件，构建：
    (image_path, question, answer) -> [score1, score2, ...]
    """
    score_dict = defaultdict(list)

    for root, _, files in os.walk(score_dir):
        for filename in files:
            if not filename.endswith(".jsonl"):
                continue

            file_path = os.path.join(root, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON in {file_path}, line {line_idx}: {e}"
                        )

                    required_keys = ["image_path", "question", "answer", "score"]
                    for key in required_keys:
                        if key not in obj:
                            raise KeyError(
                                f"Missing key `{key}` in {file_path}, line {line_idx}"
                            )

                    score_key = (
                        obj["image_path"],
                        obj["question"],
                        obj["answer"],
                    )
                    score_dict[score_key].append(float(obj["score"]))

    return score_dict


def read_input_jsonl(input_jsonl: str) -> List[dict]:
    data = []

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {input_jsonl}, line {line_idx}: {e}")

            required_keys = ["image_path", "question", "chosen", "rejected"]
            for key in required_keys:
                if key not in obj:
                    raise KeyError(f"Missing key `{key}` in input line {line_idx}")

            data.append(obj)

    return data


def add_scores_and_preference_strength(
    data: List[dict],
    score_dict: Dict[ScoreKey, List[float]],
    missing_output_jsonl: str = None,
) -> List[dict]:
    """
    给每条 preference 数据补充 chosen_scores / rejected_scores，
    并计算 preference_strength 和 avg_preference_strength。
    缺少分数的数据默认跳过，并可写入 missing_output_jsonl。
    """
    processed = []
    missing_records = []

    for idx, obj in enumerate(data):
        image_path = obj["image_path"]
        question = obj["question"]
        chosen = obj["chosen"]
        rejected = obj["rejected"]

        chosen_key = (image_path, question, chosen)
        rejected_key = (image_path, question, rejected)

        chosen_scores = score_dict.get(chosen_key)
        rejected_scores = score_dict.get(rejected_key)

        if not chosen_scores or not rejected_scores:
            missing_records.append(
                {
                    "index": idx,
                    "image_path": image_path,
                    "question": question,
                    "chosen": chosen,
                    "rejected": rejected,
                    "missing_chosen_scores": not bool(chosen_scores),
                    "missing_rejected_scores": not bool(rejected_scores),
                }
            )
            continue

        if len(chosen_scores) != len(rejected_scores):
            raise ValueError(
                f"Score length mismatch at input index {idx}: "
                f"len(chosen_scores)={len(chosen_scores)}, "
                f"len(rejected_scores)={len(rejected_scores)}"
            )

        preference_strength = [
            float(c) - float(r)
            for c, r in zip(chosen_scores, rejected_scores)
        ]

        avg_preference_strength = sum(preference_strength) / len(preference_strength)

        new_obj = dict(obj)
        new_obj["chosen_scores"] = chosen_scores
        new_obj["rejected_scores"] = rejected_scores
        new_obj["preference_strength"] = preference_strength
        new_obj["avg_preference_strength"] = avg_preference_strength
        new_obj["flipped"] = False

        processed.append(new_obj)

    if missing_output_jsonl is not None:
        with open(missing_output_jsonl, "w", encoding="utf-8") as f:
            for item in missing_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Input samples: {len(data)}")
    print(f"Samples with scores: {len(processed)}")
    print(f"Samples missing scores: {len(missing_records)}")

    return processed


def flip_bottom_50_negative(data: List[dict]) -> List[dict]:
    """
    按 avg_preference_strength 从小到大排序。
    对 avg_preference_strength < 0 的 bottom 50% 翻转 chosen/rejected。
    """
    data = sorted(data, key=lambda x: x["avg_preference_strength"])

    negative_indices = [
        idx for idx, obj in enumerate(data)
        if obj["avg_preference_strength"] < 0
    ]

    num_to_flip = len(negative_indices) // 2
    indices_to_flip = set(negative_indices[:num_to_flip])

    for idx in indices_to_flip:
        obj = data[idx]

        obj["chosen"], obj["rejected"] = obj["rejected"], obj["chosen"]
        obj["chosen_scores"], obj["rejected_scores"] = (
            obj["rejected_scores"],
            obj["chosen_scores"],
        )

        # 翻转后重新计算 preference_strength，使其和新的 chosen/rejected 一致
        preference_strength = [
            float(c) - float(r)
            for c, r in zip(obj["chosen_scores"], obj["rejected_scores"])
        ]
        obj["preference_strength"] = preference_strength
        obj["avg_preference_strength"] = sum(preference_strength) / len(preference_strength)
        obj["flipped"] = True

    print(f"Negative samples: {len(negative_indices)}")
    print(f"Flipped bottom 50% negative samples: {num_to_flip}")

    return data


def add_image_token(question: str) -> str:
    """
    输出 conversations 中的 human value。
    如果 question 已经以 <image> 开头，就不重复添加。
    """
    if question.startswith("<image>"):
        print("Error: this question should not startswith <image>")
    return "<image>" + question


def convert_to_training_format(data: List[dict]) -> List[dict]:
    """
    转换成：
    {
        "conversations": [
            {
                "from": "human",
                "value": "<image>" + question
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": chosen
        },
        "rejected": {
            "from": "gpt",
            "value": rejected
        },
        "images": [image_path]
    }
    """
    output = []

    for obj in data:
        converted = {
            "conversations": [
                {
                    "from": "human",
                    "value": add_image_token(obj["question"]),
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": obj["chosen"],
            },
            "rejected": {
                "from": "gpt",
                "value": obj["rejected"],
            },
            "images": [obj["image_path"]],
        }
        output.append(converted)

    return output


def write_json(data: List[dict], output_json: str) -> None:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Input preference jsonl file with image_path, question, chosen, rejected.",
    )
    parser.add_argument(
        "--score_dir",
        required=True,
        help="Directory containing reward score jsonl files.",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Output json file in training format.",
    )
    parser.add_argument(
        "--missing_output_jsonl",
        default=None,
        help="Optional jsonl file to save samples whose scores are missing.",
    )

    args = parser.parse_args()

    score_dict = read_score_jsonl_dir(args.score_dir)
    print(f"Loaded score keys: {len(score_dict)}")

    input_data = read_input_jsonl(args.input_jsonl)

    processed_data = add_scores_and_preference_strength(
        data=input_data,
        score_dict=score_dict,
        missing_output_jsonl=args.missing_output_jsonl,
    )

    flipped_data = flip_bottom_50_negative(processed_data)

    output_data = convert_to_training_format(flipped_data)

    write_json(output_data, args.output_json)

    print(f"Final output samples: {len(output_data)}")
    print(f"Saved to: {args.output_json}")


if __name__ == "__main__":
    main()
