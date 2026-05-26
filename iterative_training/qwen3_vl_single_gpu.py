from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor,set_seed
from qwen_vl_utils import process_vision_info
import torch
import os
import json
import math
import torch_npu;
from tqdm import tqdm
import argparse
from trl import AutoModelForCausalLMWithValueHead
from typing import TYPE_CHECKING, Dict
from transformers.utils import cached_file

random_seed = 42
set_seed(random_seed)

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"
V_HEAD_WEIGHTS_NAME = "value_head.bin"

def load_valuehead_params(path_or_repo_id: str) -> Dict[str, torch.Tensor]:
    r"""
    Loads value head parameters from Hugging Face Hub or local disk.

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
    """
    err_text = ""
    try:
        from safetensors import safe_open
        vhead_file = os.path.join(path_or_repo_id, V_HEAD_SAFE_WEIGHTS_NAME)
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)
        print("error info:",err_text)
    return None

import regex
prompt_origin_ch="您是一位公正的评审员，请评估AI助手对用户问题的回答质量。"
prompt_origin_en="You are an impartial evaluator. Please assess the quality of the AI assistant's response to the user's question."
def contains_chinese(text):
	pattern = regex.compile(r'\p{IsHan}')
	return bool(pattern.search(text))

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='/cache/zzh/critic_model_v6p19_only_mine_and_spa_vl_t2i_133k_processed_rlaif_v_r1_reward_v6_bs_512_low_resolution/checkpoint-1750', type=str)
# parser.add_argument("--image_dir", default='/cache/zzh/DPO', type=str)
parser.add_argument("--question_file", default='/cache/bt_stage_2_before_filter_data_nooverlap_answer.jsonl', type=str)
# parser.add_argument("--output_path", default='/cache/zzh/critic_qa_data/test_output.jsonl', type=str)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("id", type=int)
args = parser.parse_args()

# device_id=int(args.id/2)
# print(args.id,device_id)
#设置显卡号必须要在import torch和import torch_npu之前，否则是没用
# os.environ["ASCEND_VISIBLE_DEVICES"] = f"{device_id}"
# device=torch.device(f"npu:{device_id}")
model1 = AutoModelForImageTextToText.from_pretrained(args.model_path, local_files_only=True, device_map="npu", dtype="auto")
model = AutoModelForCausalLMWithValueHead.from_pretrained(model1)
vhead_params = load_valuehead_params(args.model_path)
model.load_state_dict(vhead_params, strict=False)
# model=model.to("npu")
# model.requires_grad_(False)
model.eval()

# default processer
# min_pixels = 0
# max_pixels = 1280 * 32 * 32
processor = AutoProcessor.from_pretrained(args.model_path) #, min_pixels=min_pixels, max_pixels=max_pixels)
# print(processor) #Qwen2_5_VLProcessor
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# print(tokenizer) #Qwen2TokenizerFast, 151644是<|im_start|> 151645是<|im_end|>

GPU_NUM=16
data = [json.loads(item.strip()) for item in open(args.question_file, "r", encoding="utf-8")][9573*3:]
chunk_size=int(len(data)/GPU_NUM)+1
start_index=args.id*chunk_size;
end_index=(args.id+1)*chunk_size
print(f"from {start_index} to {end_index}")
data=data[start_index: end_index]
print("input data num:",len(data))
num_batches=(len(data) + args.batch_size - 1) // args.batch_size
batch_size=args.batch_size
count=0;
file_w = open(f"/cache/zzh/llava_rlhf_dtit_mrm_output/reward_output{args.id}.jsonl", 'w', encoding='utf-8')

for item in tqdm(data):
    # image = item["image_path"]
    # image_path = "/cache/zzh/critic_qa_data/" + item["image_path"]
    image_path = item["image_path"]
    #if "MM-RLHF/" in image_path:
    #    continue;
    if contains_chinese(item["question"]):
        system_info=prompt_origin_ch
    else:
        system_info=prompt_origin_en
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_info
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": 0,
                    "max_pixels": 1280*32*32,
                },
                {
                    "type": "text",
                    "text": item["question"]
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": item["answer"]
                },
            ]
        }
    ]
    # Preparation for inference #, add_generation_prompt=True
    text = processor.apply_chat_template(messages, tokenize=False)
    if count==0:
        print(messages)
        print(text)
    
    try:
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    except Exception as e:
        print(f"图片损坏")
        print(e)
        continue
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        do_resize=False,
        # padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model1.device)

    # Inference: Generation of the output
    with torch.no_grad():
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        reward_score = values.gather(dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        # print(reward_score)#(batch_size,1)
        tmp = {
                'image_path': item["image_path"],
                'question': item['question'],
                'answer': item["answer"],
                "score": reward_score[0].item(),
            }
        file_w.write(json.dumps(tmp, ensure_ascii=False) + '\n')
    if count%50==0:
        file_w.flush()
    count+=1;
    #如果不注释下面的内容，会导致推理的速度变慢
    # del outputs
    # torch.cuda.empty_cache()
file_w.flush()
file_w.close()
