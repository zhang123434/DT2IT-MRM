import json
import random
import argparse
from pathlib import Path


TEMPLATE = """Image 1: <image>\nImage 2: <image>\nYou are given a text caption and two generated images based on that caption. Your task is to evaluate and compare these images based on two key criteria:\n1. Alignment with the Caption: Assess how well each image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of each image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nCompare both images using the above criteria and select the one that better aligns with the caption while exhibiting superior visual quality.\nProvide a clear conclusion such as "Image 1 is better than Image 2.", "Image 2 is better than Image 1." and "Both images are equally good."\nYour task is provided as follows:\nText Caption: [{caption}]"""

SYSTEM_PROMPT = "You are an impartial evaluator. Please assess the quality of the AI assistant's response to the user's question."

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_idx}: {e}")

            data.append(obj)

    return data


def validate_item(obj, line_idx):
    if "chosen" not in obj or "rejected" not in obj:
        raise KeyError(f"Line {line_idx}: missing `chosen` or `rejected`.")

    if "value" not in obj["chosen"] or "value" not in obj["rejected"]:
        raise KeyError(
            f"Line {line_idx}: missing `chosen['value']` or `rejected['value']`."
        )

    chosen_caption = obj["chosen"]["value"]
    rejected_caption = obj["rejected"]["value"]

    if chosen_caption != rejected_caption:
        raise ValueError(
            f"Line {line_idx}: chosen value != rejected value.\n"
            f"chosen: {chosen_caption}\n"
            f"rejected: {rejected_caption}"
        )

    if "images" not in obj or not isinstance(obj["images"], list) or len(obj["images"]) != 2:
        raise ValueError(
            f"Line {line_idx}: `images` should be a list with exactly two image paths."
        )

    # obj["images"][0] 是 chosen image，obj["images"][1] 是 rejected image
    return chosen_caption, obj["images"][0], obj["images"][1]


def convert_item(caption, chosen_image, rejected_image, put_chosen_first):
    conversation_value = TEMPLATE.format(caption=caption)

    if put_chosen_first:
        images = [chosen_image, rejected_image]
        chosen_answer = "Image 1 is better than Image 2."
        rejected_answer = "Image 2 is better than Image 1."
    else:
        images = [rejected_image, chosen_image]
        chosen_answer = "Image 2 is better than Image 1."
        rejected_answer = "Image 1 is better than Image 2."

    return {
        "conversations": [
            {
                "from": "human",
                "value": conversation_value,
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": chosen_answer,
        },
        "rejected": {
            "from": "gpt",
            "value": rejected_answer,
        },
        "images": images,
        "system": SYSTEM_PROMPT,
    }


def build_position_flags(n, seed=42):
    """
    True  表示 chosen image 放在 Image 1
    False 表示 chosen image 放在 Image 2

    如果 n 是偶数，则两类数量完全相等。
    如果 n 是奇数，则两类数量相差 1，多出来的那个位置随机决定。
    """
    random.seed(seed)

    half = n // 2
    position_flags = [True] * half + [False] * half

    if n % 2 == 1:
        position_flags.append(random.choice([True, False]))

    random.shuffle(position_flags)
    return position_flags


def convert_jsonl_to_json(input_path, output_path, seed=42):
    data = read_jsonl(input_path)
    position_flags = build_position_flags(len(data), seed=seed)

    converted_data = []
    chosen_first_count = 0
    chosen_second_count = 0

    for line_idx, (obj, put_chosen_first) in enumerate(zip(data, position_flags), start=1):
        caption, chosen_image, rejected_image = validate_item(obj, line_idx)

        new_obj = convert_item(
            caption=caption,
            chosen_image=chosen_image,
            rejected_image=rejected_image,
            put_chosen_first=put_chosen_first,
        )

        converted_data.append(new_obj)

        if put_chosen_first:
            chosen_first_count += 1
        else:
            chosen_second_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=4)

    print(f"Input samples: {len(data)}")
    print(f"Output samples: {len(converted_data)}")
    print(f"Chosen image as Image 1: {chosen_first_count}")
    print(f"Chosen image as Image 2: {chosen_second_count}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, help="Input jsonl file.")
    parser.add_argument("--output_json", required=True, help="Output json file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    convert_jsonl_to_json(
        input_path=args.input_jsonl,
        output_path=args.output_json,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
