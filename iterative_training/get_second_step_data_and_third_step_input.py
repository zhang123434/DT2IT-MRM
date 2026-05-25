import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any, List


ScoreKey = Tuple[str, str, str]


def read_jsonl_score_files(score_dir: str) -> Dict[ScoreKey, float]:
    """
    遍历 score_dir 下所有 .jsonl 文件，建立：
    (image_path, question, answer) -> score
    """
    score_dict: Dict[ScoreKey, float] = {}
    duplicate_count = 0

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
                        raise ValueError(f"Invalid JSON in {file_path}, line {line_idx}: {e}")

                    try:
                        key = (
                            obj["image_path"],
                            obj["question"].strip(),
                            obj["answer"],
                        )
                        score = float(obj["score"])
                    except KeyError as e:
                        raise KeyError(f"Missing key {e} in {file_path}, line {line_idx}")

                    if key in score_dict:
                        duplicate_count += 1

                    # 如果有重复 key，默认后出现的覆盖前面的
                    score_dict[key] = score

    print(f"Loaded {len(score_dict)} unique score records.")
    if duplicate_count > 0:
        print(f"Warning: found {duplicate_count} duplicate keys. Later scores overwrote earlier ones.")

    return score_dict


def load_json_list(input_json_path: str) -> List[dict]:
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON file should be a list of dictionaries.")

    return data


def write_json(data: List[dict], output_json_path: str) -> None:
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def append_jsonl_line(f, obj: dict) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_question(obj: dict) -> str:
    return obj["conversations"][0]["value"]


def get_clean_question(obj: dict) -> str:
    return obj["conversations"][0]["value"].replace("<image>", "").strip()


def process_data(
    score_dict: Dict[ScoreKey, float],
    input_json_path: str,
    output_json_path: str,
    inconsistent_jsonl_path: str,
    missing_jsonl_path: str = None,
) -> None:
    data = load_json_list(input_json_path)

    kept_data = []
    kept_count = 0
    inconsistent_count = 0
    missing_count = 0

    inconsistent_f = open(inconsistent_jsonl_path, "w", encoding="utf-8")
    missing_f = open(missing_jsonl_path, "w", encoding="utf-8") if missing_jsonl_path else None

    try:
        for idx, obj in enumerate(data):
            try:
                image_path = obj["images"][0]
                question = get_question(obj)
                clean_question = get_clean_question(obj)
                chosen_answer = obj["chosen"]["value"]
                rejected_answer = obj["rejected"]["value"]
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(f"Invalid data format at item index {idx}: {e}")

            chosen_key = (image_path, question, chosen_answer)
            rejected_key = (image_path, question, rejected_answer)

            if chosen_key not in score_dict or rejected_key not in score_dict:
                missing_count += 1

                missing_record = {
                    "index": idx,
                    "image_path": image_path,
                    "question": clean_question,
                    "chosen": chosen_answer,
                    "rejected": rejected_answer,
                    "missing_chosen_score": chosen_key not in score_dict,
                    "missing_rejected_score": rejected_key not in score_dict,
                }

                if missing_f is not None:
                    append_jsonl_line(missing_f, missing_record)

                continue

            chosen_score = score_dict[chosen_key]
            rejected_score = score_dict[rejected_key]

            if chosen_score > rejected_score:
                kept_data.append(obj)
                kept_count += 1
            else:
                inconsistent_count += 1

                # 原始顺序：chosen, rejected
                record_1 = {
                    "image_path": image_path,
                    "question": clean_question,
                    "origin_chosen": chosen_answer,
                    "origin_rejected": rejected_answer,
                    "answer": [chosen_answer, rejected_answer],
                }

                # 反转顺序：rejected, chosen
                record_2 = {
                    "image_path": image_path,
                    "question": clean_question,
                    "origin_chosen": chosen_answer,
                    "origin_rejected": rejected_answer,
                    "answer": [rejected_answer, chosen_answer],
                }

                append_jsonl_line(inconsistent_f, record_1)
                append_jsonl_line(inconsistent_f, record_2)

        write_json(kept_data, output_json_path)

    finally:
        inconsistent_f.close()
        if missing_f is not None:
            missing_f.close()

    print("Done.")
    print(f"Total input pairs: {len(data)}")
    print(f"Kept pairs: {kept_count}")
    print(f"Inconsistent pairs: {inconsistent_count}")
    print(f"Missing-score pairs: {missing_count}")
    print(f"Saved kept JSON to: {output_json_path}")
    print(f"Saved inconsistent JSONL to: {inconsistent_jsonl_path}")
    if missing_jsonl_path:
        print(f"Saved missing-score records to: {missing_jsonl_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--score_dir",
        required=True,
        help="Directory containing many .jsonl files with image_path, question, answer, score.",
    )
    parser.add_argument(
        "--input_json",
        required=True,
        help="Input .json file. It should be a list of preference dictionaries.",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Output .json file for kept preference data.",
    )
    parser.add_argument(
        "--inconsistent_jsonl",
        required=True,
        help="Output .jsonl file for pairs whose chosen_score <= rejected_score.",
    )
    parser.add_argument(
        "--missing_jsonl",
        default=None,
        help="Optional .jsonl file for records whose scores cannot be found.",
    )

    args = parser.parse_args()

    score_dict = read_jsonl_score_files(args.score_dir)

    process_data(
        score_dict=score_dict,
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        inconsistent_jsonl_path=args.inconsistent_jsonl,
        missing_jsonl_path=args.missing_jsonl,
    )


if __name__ == "__main__":
    main()
