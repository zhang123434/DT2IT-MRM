import json;
import os;
data=[json.loads(line.strip()) for line in open(r"D:\workspace\data\train_data\bt_rm_stage_2\r1_reward_rlaif_v_v6p19_only_mine_and_spa_vl_third_step_input_rest.jsonl",'r',encoding='utf-8')]
print(len(data))
dict1={}
for obj in data:
	elem=(obj["image_path"],obj["question"],obj["origin_chosen"],obj["origin_rejected"])
	dict1[elem]=[0,0]
print(len(dict1.keys())) # voting的结果；

data1=[json.loads(line.strip()) for line in open(r"D:\workspace\data\train_data\bt_rm_stage_2\r1_reward_rlaif_v_v6p19_only_mine_and_spa_vl_third_step_input_rest_gemini_output.jsonl",'r',encoding='utf-8')]
data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\data\train_data\bt_rm_stage_2\r1_reward_rlaif_v_v6p19_only_mine_and_spa_vl_third_step_input_rest_gpt_5_output.jsonl",'r',encoding='utf-8')])
# data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\MM-RLHF\mmrlhf_third_step_input_gemini_output.jsonl",'r',encoding='utf-8')])
# data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\MM-RLHF\mmrlhf_third_step_input_gpt_5_output.jsonl",'r',encoding='utf-8')])
# data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\MM-RLHF\mmrlhf_third_step_input_v2_rest_gemini_output.jsonl",'r',encoding='utf-8')])
# data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\MM-RLHF\mmrlhf_third_step_input_v2_rest_gpt_5_output.jsonl",'r',encoding='utf-8')])
# data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\data\train_data\bt_rm_stage_2\r1_reward_rlaif_v_v6p19_only_mine_and_spa_vl_third_step_input_rest_gpt_5_output.jsonl",'r',encoding='utf-8')])
# data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\data\train_data\bt_rm_stage_2\r1_reward_rlaif_v_v6p19_only_mine_and_spa_vl_third_step_input_rest_gemini_output.jsonl",'r',encoding='utf-8')])
# data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\r1-reward_rl\r1_reward_except_mmrlhf_third_step_input_gemini_output.jsonl",'r',encoding='utf-8')])
# data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\r1-reward_rl\r1_reward_except_mmrlhf_third_step_input_gpt_5_output.jsonl",'r',encoding='utf-8')])
# data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\data\train_data\bt_rm_stage_2\r1_reward_rlaif_v_v6p19_only_mine_and_spa_vl_third_step_input_rest_v1_gpt_5_output.jsonl",'r',encoding='utf-8')])
# data1.extend([json.loads(line.strip()) for line in open(r"D:\workspace\data\train_data\bt_rm_stage_2\r1_reward_rlaif_v_v6p19_only_mine_and_spa_vl_third_step_input_rest_v1_gemini_output.jsonl",'r',encoding='utf-8')])
print(len(data1))
noscore=0;
for obj in data1:
	if set(obj["answer"])!=set([obj["origin_chosen"],obj["origin_rejected"]]):
		print("erorr")
	if obj["score"]==[] or obj['score'] is None:
		noscore+=1;
		continue;
	elem=(obj["image_path"],obj["question"],obj["origin_chosen"],obj["origin_rejected"])
	if elem not in dict1:
		print("error111")
		continue;
	try:
		if obj["score"][0]>obj["score"][1]:
			if obj["answer"][0]==obj["origin_chosen"]:
				dict1[elem][0]+=1;
			elif obj["answer"][0]==obj["origin_rejected"]:
				dict1[elem][1]+=1;
			else:
				print("erorr")
		elif obj["score"][1]>obj["score"][0]:
			if obj["answer"][1]==obj["origin_chosen"]:
				dict1[elem][0]+=1;
			elif obj["answer"][1]==obj["origin_rejected"]:
				dict1[elem][1]+=1;
			else:
				print("erorr")
		else:
			dict1[elem][0]+=0.5
			dict1[elem][1]+=0.5
	except Exception as e:
		print(e)
		continue;

print("no score num:",noscore)
count=0;
tie=0;
for key,value in dict1.items():
	if count==0:
		print(value)
	if value[0]>value[1]:
		count+=1;
	elif value[0]==value[1]:
		tie+=1;
print("consist with gpt&gemini :",count,len(dict1.keys()),count/len(dict1.keys()))# 和gpt-5和gemini一致的数据很少,只有6847条，这些数据还是有很大的噪声的
print("tie case:",tie)

import regex
prompt_origin_ch="您是一位公正的评审员，请评估AI助手对用户问题的回答质量。"
prompt_origin_en="You are an impartial evaluator. Please assess the quality of the AI assistant's response to the user's question."

def contains_chinese(text):
	pattern = regex.compile(r'\p{IsHan}')
	return bool(pattern.search(text))

final_data=[]
skip=0;
# from collections import defaultdict;
# dict3=defaultdict(int)
for key,value in dict1.items():
	if value[0]==value[1]:
		continue;
	elif value[0]>value[1]:
		chosen=key[2]
		rejected=key[3]
	elif value[1]>value[0]:
		chosen=key[3]
		rejected=key[2]
		# continue;
	if value[0]+value[1]!=2:
		skip+=1;
		continue;
	# dict3[value[0]+value[1]]+=1;
	if "<image>" in key[1]:
		print("erorr")
	json_obj={}
	json_obj["conversations"]=[{"from":"human","value": "<image>"+key[1]}]
	json_obj["chosen"]= {"from":"gpt","value": chosen}
	json_obj["rejected"]={"from":"gpt","value": rejected}
	json_obj["images"]=[key[0]]
	if contains_chinese(key[1]):
		json_obj["system"]=prompt_origin_ch
	else:
		json_obj["system"]=prompt_origin_en
	final_data.append(json_obj)

print("all data num:",len(final_data))
outfilename=r"D:\workspace\data\train_data\bt_rm_stage_2\mmpr_second_sample_data_third_step_input_gpt_processed_all.json"
outfile=open(outfilename, 'w', encoding='utf-8')
json.dump(final_data, outfile, ensure_ascii=False, indent=4)
outfile.flush()
outfile.close()
print("skip data num:",skip)
# print(dict3)
