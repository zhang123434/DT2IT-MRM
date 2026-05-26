import json
from pathlib import Path


input_file = r"D:\workspace\MMPR-v1.1\bt_stage_2_before_filter_data.jsonl"# 这里输入的也可以是.json文件
output_file = r"D:\workspace\MMPR-v1.1\bt_stage_2_before_filter_data_nooverlap_answer.jsonl"


def load_data(input_path):
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    data = []

    if suffix == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                question = obj["question"]

                if "<image>" in question:
                    print(f"error: <image> found in jsonl line {line_idx}")

                data.append({
                    "image_path": obj["image_path"],
                    "question": question,
                    "chosen": obj["chosen"],
                    "rejected": obj["rejected"],
                })

    elif suffix == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError("The input .json file should be a list of dictionaries.")

        for idx, obj in enumerate(raw_data):
            try:
                question = obj["conversations"][0]["value"].replace("<image>", "").strip()

                data.append({
                    "image_path": obj["images"][0],
                    "question": question,
                    "chosen": obj["chosen"]["value"],
                    "rejected": obj["rejected"]["value"],
                })

            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(f"Invalid format at json item {idx}: {e}")

    else:
        raise ValueError(f"Unsupported input file type: {suffix}. Please use .jsonl or .json.")

    return data


data = load_data(input_file)
print("Input data length:", len(data))

seen = set()

with open(output_file, "w", encoding="utf-8") as outfile:
    for obj in data:
        image_path = obj["image_path"]
        question = obj["question"]
        chosen = obj["chosen"]
        rejected = obj["rejected"]

        elem = (image_path, question, chosen)
        if elem not in seen:
            seen.add(elem)
            temp = {
                "image_path": image_path,
                "question": question,
                "answer": chosen,
            }
            json.dump(temp, outfile, ensure_ascii=False)
            outfile.write("\n")

        elem1 = (image_path, question, rejected)
        if elem1 not in seen:
            seen.add(elem1)
            temp1 = {
                "image_path": image_path,
                "question": question,
                "answer": rejected,
            }
            json.dump(temp1, outfile, ensure_ascii=False)
            outfile.write("\n")

print("Unique answer count:", len(seen))
print("Saved to:", output_file)
