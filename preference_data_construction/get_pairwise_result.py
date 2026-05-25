#coding=utf-8
from openai import OpenAI
import ssl
import httpx
import base64
import json
import copy
from tqdm import tqdm
import os
import pandas as pd
import concurrent.futures
import time;
import random;
from json_repair import repair_json
from prompt.prompt_pairwise_general_11_3 import prompt_criticV3_en1,prompt_criticV3_ch1;

import re
def get_score1(text,num):
	text=text.replace('\\', '\\\\')
	text=text.replace('```json\n', '')
	text=text.replace('\n```', '')
	if not contain_chinese(text):
		text=text.encode('utf-8').decode('unicode_escape')
	try:
		result=json.loads(text.strip(), strict=False)
	except Exception as e:
		text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
		try:
			result=json.loads(text.strip(), strict=False)
		except Exception as e1:
			print(e1)
			return True,[]
	try:
		overall=[]
		for value in list(result.values())[1:]:
			valuelist=list(value.values())
			if len(valuelist)==0:
				continue;
			last_value = valuelist[-1]
			overall.append(last_value)
		if len(overall)==num:
			return False,overall;
		elif len(overall)>num:
			return False,overall[:num]
		else:
			return True,[]
	except Exception as e:
		print(e)
		return True,[]

def check_response1(text):
	# if "standard human" in text or "human-generated" in text or "human generated" in text or "standard answer" in text or "标准答案" in text or "人类生成" in text or "标准人类" in text or "生成答案" in text or "生成的答案" in text  or "生成的标准" in text or "人类的答案" in text or "人类的标准" in text or "人类答案" in text or "标准的人类" in text or "参考答案" in text: # 这个已经暂时不用了
	# 	return True;
	if "standard human" in text or "human-generated" in text or "human generated" in text or "standard answer" in text or "标准答案" in text or "人类生成" in text or "标准人类" in text or "生成的标准" in text or "人类的答案" in text or "人类的标准" in text or "人类答案" in text or "标准的人类" in text:
		print("mention standard answer")
		return True;
	text=text.replace('\\', '\\\\')
	text=text.replace('```json\n', '')
	text=text.replace('\n```', '')
	if not contain_chinese(text):
		text=text.encode('utf-8').decode('unicode_escape')
	try:
		result=json.loads(text.strip(), strict=False)
	except Exception as e:
		text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
		try:
			result=json.loads(text.strip(), strict=False)
		except Exception as e1:
			print(e1)
			return True
	try:
		count=0;
		result_list=list(result.values())[1:]
		temp_num=len(result_list)
		for value in result_list:
			lst=[i+1 for i in range(temp_num)]# 除了assistant i之外的字符
			count+=1;
			lst.remove(count)
			check_content=str(value).lower()
			for elem in lst:
				if f"assistant {elem}" in check_content or f"助手{elem}" in check_content or f"助手 {elem}" in check_content or f"AI {elem}" in check_content:
					print("mention another assistant")
					return True;
		return False
	except Exception as e:
		print(e)
		return True;

def check_response2(text):
	text=text.replace('\\', '\\\\')
	text=text.replace('```json\n', '')
	text=text.replace('\n```', '')
	if not contain_chinese(text):
		text=text.encode('utf-8').decode('unicode_escape')
	try:
		result=json.loads(text.strip(), strict=False)
		return False
	except Exception as e:
		text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
		try:
			result=json.loads(text.strip(), strict=False)
			return False
		except Exception as e1:
			print(e1)
			return True

def check_response(text):
	try:
		content = repair_json(text)
		result=json.loads(fr"{content}", strict=False)
		return False
	except Exception as e:
		print("check response error:", e)
		return True;

def get_score(text,num):
	try:
		content = repair_json(text)
		result=json.loads(fr"{content}", strict=False)
	except Exception as e:
		print(e)
		return True,[]
	try:
		overall=[]
		for value in list(result.values())[1:]:
			valuelist=list(value.values())
			if len(valuelist)==0:
				continue;
			last_value = valuelist[-1]
			overall.append(last_value)
		if len(overall)==num:
			return False,overall;
		elif len(overall)>num:
			return False,overall[:num]
		else:
			return True,[]
	except Exception as e:
		print(e)
		return True,[]

def get_item(prompt, image_path, item):
	base64_image = encode_image(image_path=image_path)
	new_prompt = [
		{
			"role": "user",
			"content": [
				{   
					"type": "text", 
					"text": prompt
				},
				{
					"type": "image_url",
					"image_url": {"url": f"data:image/png;base64,{base64_image}"}
				}
			],
		}
	]
	item["new_prompt"]=new_prompt
	return item

def get_item1(prompt, image_path, item):
	base64_image = encode_image(image_path=image_path)
	new_prompt = [
		{
			"role": "user",
			"content": [
				{   
					"type": "input_text", 
					"text": prompt
				},
				{
					"type": "input_image",
					"image_url": f"data:image/png;base64,{base64_image}",
				}
			],
		}
	]
	item["new_prompt"]=new_prompt
	return item

def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')

def get_output(item):
	MAX_TRY_TIMES = 0
	response_lst=[]
	while True:
		try:
			MAX_TRY_TIMES += 1
			if MAX_TRY_TIMES>3:
				content=""
				break;
			client = OpenAI(api_key='<KEY>', http_client=httpx.Client(verify=False))
			
			# client.api_key = 'sk-rHvVv6vqzOjWBrSn25A76d7f0d1d4b5eBfB7928284D6Cd7a'
			# client.api_key = 'sk-2gHJWhnV9hAMz98t4255F84c86F04aC9BcE38643E63a912f'
			# client.api_key = 'sk-4sq2G8qXuHAzESP2CLqGdU2qTD3ygh0TIzeKUUm9QQX110Cc'
			# client.api_key = 'sk-hgiYYZZ6HorJHjxr7a5eE172Ce7041DcA08b651d113bC69f'
			# client.base_url = 'http://rerverseapi.workergpt.cn/v1'
			client.api_key = '<your key>'
			client.base_url= 'https://xxxx'
			# gpt-5-chat-latest
			# response = client.chat.completions.create(
			# 	model='gpt-5-chat-latest',
			# 	messages=item['new_prompt'],
			# 	temperature=0,  
			# 	max_tokens=5120,
			# )

			#gpt-5 infer 1:
			response = client.chat.completions.create(
				model='gemini-3-pro-preview',
				messages=item['new_prompt'],
				temperature=1,
				max_completion_tokens=8192,
			)
			response_lst.append(response.to_dict())
			content = response.choices[0].message.content
			# gpt-5 infer 2:
			# response = client.responses.create(
			# 	model='gpt-5',
			# 	input=item['new_prompt'],
			# 	# temperature=1,
			# 	max_output_tokens=8192,
			# 	text={"verbosity": "medium"},
			# 	reasoning={"effort": "medium"},
			# )
			# response_lst.append(response.to_dict())
			# content = response.output_text
			if check_response(content):#retry两次
				raise ValueError("Check response failed")
			if content is None or content=="":
				raise ValueError("empty response")
			break
		except Exception as e:
			time.sleep(0.1)
			print(e)

	item["response"] = content
	item["error"],item["score"]=get_score(item["response"],num=len(item["answer"]))
	item["ifrevise"]=check_response(item["response"])
	del item["new_prompt"]
	item["raw_response"]=response_lst
	item["key"]="fH37gCle"
	return item

import regex
def contain_chinese(text):
	pattern = regex.compile(r'\p{IsHan}')
	return bool(pattern.search(text))

def get_ch_answer(answers):
	temp=""
	for i in range(len(answers)):
		temp+=f"### AI助手{i+1}的回答：\n"
		temp+=answers[i]+"\n\n"
	return temp;

def get_en_answer(answers):
	temp=""
	for i in range(len(answers)):
		temp+=f"### AI Assistant {i+1}'s Response:\n"
		temp+=answers[i]+"\n\n"
	return temp;

def get_exising_data(path):
	if not os.path.exists(path):
		return set()
	data=[json.loads(line.strip()) for line in open(path,'r',encoding='utf-8')]
	print("origin data num:",len(data))
	a=set()
	final_data=[]
	for obj in data:
		if obj["raw_response"]==[] and obj["response"]=="":
			continue;
		answer=copy.deepcopy(obj["answer"])
		answer.sort()
		elem=(obj["data_id"],obj["image_path"],obj["question"],tuple(answer))
		a.add(elem)
		final_data.append(obj)
	print("have completed data num:",len(a),len(final_data))
	temp_outfile=open(path,'w',encoding='utf-8')
	for obj in final_data:
		json.dump(obj,temp_outfile,ensure_ascii=False)
		temp_outfile.write("\n")
	temp_outfile.flush()
	temp_outfile.close()
	return a;

def shuffle_different(lst1):
	shuffled=copy.deepcopy(lst1)
	while True:
		random.shuffle(shuffled)
		if shuffled != lst1:
			return shuffled

if __name__ == '__main__':
	# gt_answer=json.load(open(r"D:\workspace\data\train_data\other\origin\gt_answer.json",'r',encoding='utf-8'))

	root_path=r"D:\static"
	# task_path=[r"D:\workspace\process\output1\hallucination_en_pairwise_output.jsonl", r"D:\workspace\process\output1\hallucination_ch_pairwise_output.jsonl"]
	# output_paths=[r"D:\workspace\process\pointwise_output3\pairwise_ablation\hallucination_en_pairwise_output_shuffle_answer.jsonl",r"D:\workspace\process\pointwise_output3\pairwise_ablation\hallucination_ch_pairwise_output_shuffle_answer.jsonl"]
	# task_path = [r"D:\workspace\process\pointwise_output3\type013_gpt_5_1_input.jsonl"]
	# output_paths =[r"D:\workspace\process\pointwise_output3\pairwise_ablation\type013_pairwise_output_shuffle_answer.jsonl"]
	task_path = [r"D:\workspace\process\output1\hallucination_en_pairwise_output.jsonl"]
	output_paths =[r"D:\workspace\process\pointwise_output3\gemini_pairwise_ablation\hallucination_en_gemini_pairwise_output.jsonl"]

	print(output_paths)
	for i in range(len(task_path)):
		count=0;
		task_list = []
		existing_set=get_exising_data(output_paths[i])
		with open(task_path[i], 'r', encoding='utf-8') as file:
			for id, line in enumerate(file):
				count+=1
				if count>=6500:
					break;
				item = json.loads(line.strip())
				if "gt" not in item:
					continue;
				if len(item["answer"])==1:
					continue;
				temp_answer=copy.deepcopy(item["answer"])
				temp_answer.sort()
				elem=(item["data_id"],item["image_path"],item["question"],tuple(temp_answer))
				if elem in existing_set:
					continue;
				# if not item["error"] and not item["ifrevise"]:
				# 	continue;
				# item["previous_score"]=item["score"]
				# if "origin_score" in item:
				# 	del item["origin_score"]
				# del item["raw_response"]
				if "final_score" in item:
					del item["final_score"]
				if "previous_score" in item:
					del item["previous_score"]
				del item["response"]
				del item["score"]
				del item["error"]
				del item["ifrevise"]
				image_path = os.path.join(root_path,item['image_path'])
				origin_answer=copy.deepcopy(item["answer"])
				# 打乱待评估的回答
				if len(set(item["answer"]))!=1:
					item["answer"]=shuffle_different(item["answer"])
					if item["answer"]==origin_answer:
						print("erorr")
				if contain_chinese(item['question']):
					prompt=prompt_criticV3_ch1.format(question=item['question'],gt=item["gt"],answer=get_ch_answer(item["answer"]))
				else:
					prompt=prompt_criticV3_en1.format(question=item['question'], gt=item["gt"], answer=get_en_answer(item["answer"]))
					#是否需要再次随机打乱待评估的回答->暂时不需要
				if count==1:
					print(prompt)
				task_list.append(copy.deepcopy(get_item(prompt=prompt, image_path=image_path, item=item)))

		batch_size = 1  # 每1000个结果保存一次
		output = []
		num_workers = 180  # 线程池中的工作线程数量

		# 打开文件进行写入
		with open(output_paths[i], 'a', encoding='utf-8') as f:
			with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
				futures = [executor.submit(get_output, item) for item in task_list]
				for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
					try:
						res = future.result()
						output.append(copy.deepcopy(res))
						
						# 当累计完成的任务达到batch_size时，保存到文件
						if len(output) >= batch_size:
							# 将多个JSON对象写入文件
							lines = [json.dumps(sample, ensure_ascii=False) + '\n' for sample in output]
							f.writelines(lines)
							# 清空输出列表
							f.flush()
							output = []
					except Exception as e:
						print(f"Error processing a task: {e}")
			
			# 保存剩余的结果
			if output:
				lines = [json.dumps(sample, ensure_ascii=False) + '\n' for sample in output]
				f.writelines(lines)
				f.flush()
		print("所有任务已完成并保存。")
