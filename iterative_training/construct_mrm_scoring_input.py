import json;
data=[json.loads(line.strip()) for line in open(r"D:\workspace\MMPR-v1.1\bt_stage_2_before_filter_data.jsonl",'r',encoding='utf-8')]
print(len(data))
a=set()
outfile=open(r"D:\workspace\MMPR-v1.1\bt_stage_2_before_filter_data_nooverlap_answer.jsonl",'w',encoding='utf-8')
for obj in data:
	if "<image>" in obj["question"]:
		print("erorr")
	elem=(obj["image_path"],obj["question"],obj["chosen"])
	if elem not in a:
		a.add(elem)
		temp={"image_path": obj["image_path"],"question": obj["question"], "answer": obj["chosen"]}
		json.dump(temp,outfile,ensure_ascii=False)
		outfile.write("\n")
	elem1=(obj["image_path"],obj["question"],obj["rejected"])
	if elem1 not in a:
		a.add(elem1)
		temp1={"image_path": obj["image_path"],"question": obj["question"], "answer": obj["rejected"]}
		json.dump(temp1,outfile,ensure_ascii=False)
		outfile.write("\n")
outfile.flush()
outfile.close()
print(len(a))
