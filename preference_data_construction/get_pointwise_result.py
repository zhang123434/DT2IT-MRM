#coding=utf-8
from openai import OpenAI
import base64
import json
import copy
from tqdm import tqdm
import os
import time;
import httpx
import pandas as pd
import concurrent.futures
from prompt.prompt_pointwise import prompt_criticV3_en#,prompt_criticV3_ch

import re;
def get_score(text):
	# 对于gemini的输出可能需要找到json的位置
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
			return None
	try:
		return list(result.values())[-1]
	except Exception as e:
		print(e)
		return None

def check_response(text):# pointwise评估中不需要管提到了人类的标准答案，也不会提到其他助手的回答
	# if "standard human" in text or "human-generated" in text or "human generated" in text or "standard answer" in text or "标准答案" in text or "人类生成" in text or "标准人类" in text or "生成的标准" in text or "人类的答案" in text or "人类的标准" in text or "人类答案" in text or "标准的人类" in text:
	# 	print("mention standard answer")
	# 	return True;
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

def get_score1(text: str):
	"""
	提取文本中 <score>...</score> 之间的浮点数，返回一个浮点数列表。
	"""
	# 匹配 <score> 与 </score> 之间的内容，允许有空格
	pattern = r"<score>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*</score>"
	matches = re.findall(pattern, text)
	# 转换为 float
	ret=[float(m) for m in matches]
	if len(ret)==0:
		return None;
	else:
		return ret[-1]

def check_response1(text):# 调用gemini需要把参考答案放在<answer>和</answer>之间
	pattern = r"<answer>\s*.+?\s*</answer>"
	return re.search(pattern, text, flags=re.DOTALL) is not None

def get_item1(prompt, image_path, item):
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

def get_item(prompt, image_path, item):
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
			#gpt-5-chat-latest
			# response = client.chat.completions.create(
			# 	model='gpt-5-chat-latest',
			# 	messages=item['new_prompt'],
			# 	temperature=0,  
			# 	max_tokens=5120,
			# )

			#gpt-5 infer 1:
			# response = client.chat.completions.create(
			# 	model='gpt-5',
			# 	messages=item['new_prompt'],
			# 	temperature=1,
			# 	max_completion_tokens=8192,
			# )
			# response_lst.append(response.to_dict())
			# content = response.choices[0].message.content
			# gpt-5 infer 2:
			response = client.responses.create(
				model='gpt-5',
				input=item['new_prompt'],
				# temperature=1,
				max_output_tokens=8192,
				text={"verbosity": "medium"},
				reasoning={"effort": "medium"},
			)
			response_lst.append(response.to_dict())
			content = response.output_text
			if check_response(content):#retry两次
				raise ValueError("Check response failed")
			if content is None or content=="":
				raise ValueError("empty response")
			break
		except Exception as e:
			time.sleep(0.1)
			print(e)

	item["response"] = content
	item["score"]=get_score(item["response"])
	item["ifrevise"]=check_response(item["response"])
	del item["new_prompt"]
	item["raw_response"]=response_lst
	return item

import regex
def contain_chinese(text):
	pattern = regex.compile(r'\p{IsHan}')
	return bool(pattern.search(text))

if __name__ == '__main__':

	root_path=r"D:\static"
	task_path=[r"D:\workspace\process\output1\hallucination_en_pairwise_output.jsonl"]
	output_paths=[r"D:\workspace\process\pointwise_output3\hallucination_en_pointwise_output.jsonl"]
	print(output_paths)
	for i in range(len(task_path)):
		task_list = []
		count=0;
		with open(task_path[i], 'r', encoding='utf-8') as file:
			for id, line in enumerate(file):
				if id>9000:
					break;
				origin_item = json.loads(line.strip())
				for answer in origin_item["answer"]:
					item={"image_path": origin_item["image_path"],"question": origin_item["question"],"answer": answer,"gt": origin_item["gt"]}
					image_path = os.path.join(root_path, item['image_path'])
					if contain_chinese(item['question']):
						continue;
						# prompt=prompt_criticV3.format(question=item['question'], answer=item['answer'],gt=item["gt"])
					else:
						prompt=prompt_criticV3_en.format(question=item['question'], answer=item['answer'], gt=item["gt"])
					if count==0:
						print(prompt)
						count+=1;
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


"""
for i in range(len(task_path)):
		print("process ",task_path[i])
		outfile=open(output_paths[i], 'w', encoding='utf-8')
		chunk_size=10000
		item_lst= [json.loads(line.strip()) for line in open(task_path[i], 'r', encoding='utf-8')]
		for j in range(int(len(item_lst)/chunk_size)+1):#一次性加载不进去，使用分片操作
			task_list = []
			count=0;
			for origin_item in item_lst[j*chunk_size:(j+1)*chunk_size]:
				for answer in origin_item["answer"]:
					item={"image_path": origin_item["image_path"],"question": origin_item["question"],"answer":answer,"gt": origin_item["gt"]}
					image_path = os.path.join(root_path,item['image_path'])
					if contain_chinese(item['question']):
						prompt=prompt_criticV3.format(question=item['question'], answer=item['answer'],gt=item["gt"])
					else:
						prompt=prompt_criticV3_en.format(question=item['question'], answer=item['answer'],gt=item["gt"])
					if count==0:
						print(prompt)
						count+=1
					task_list.append(copy.deepcopy(get_item(prompt=prompt, image_path=image_path, item=item)))

			batch_size = 100  # 每1000个结果保存一次
			output = []
			num_workers = 180  # 线程池中的工作线程数量

			# 打开文件进行写入
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
							outfile.writelines(lines)
							# 清空输出列表
							outfile.flush()
							output = []
					except Exception as e:
						print(f"Error processing a task: {e}")
			
			# 保存剩余的结果
			if output:
				lines = [json.dumps(sample, ensure_ascii=False) + '\n' for sample in output]
				outfile.writelines(lines)
				outfile.flush()
		outfile.close()
		print("所有任务已完成并保存。")
"""
