from collections import defaultdict;
import json;
import regex
output_path=[r"D:\workspace\data\test_data\result\qwen3_vl_8b_instruct_three_bench.jsonl"]
print("output file:",output_path)

gt_path=[r"D:\reward_data\MM-RLHF-RewardBench\mm_reward_bench.jsonl",r"D:\workspace\data\test_data\test_input\vlrewardbench_newest_raw.jsonl",r"D:\workspace\data\test_data\test_input\gt\multimodel-rewardbench_with-Category.jsonl"]

def compute_mmrewardbench(lst,lst1):
	category_keywords = ["mcq", "long", "short", "safety", "video"]
	# 初始化统计
	category_stats = {keyword: {"accuracy": 0, "acc_plus": 0, "total": 0} for keyword in category_keywords}
	overall_stats = {"accuracy": 0, "acc_plus": 0, "total": 0}

	# 用于存储每个id的items
	id_to_items = defaultdict(list)

	# 读取数据并分类
	for i in range(len(lst1)):
		item=lst1[i]
		item1=lst[i]
		item["rewards"]=[item1["chosen"],item1["rejected"]]
		image_path = item.get("image", "") or item.get("video", "")
		item_id = item.get("id", "")
		id_to_items[item_id].append(item)

		# 分类到相应类别
		for keyword in category_keywords:
			if keyword in image_path:
				category_stats[keyword]["total"] += 1
				break

		# 更新总计
		overall_stats["total"] += 1

	# 计算accuracy和acc_plus
	for item_id, items in id_to_items.items():
		# 统计单个id是否满足acc+
		all_correct = True
		for item in items:
			reward_0 = item["rewards"][0]
			reward_1 = item["rewards"][1]
			correct = reward_0 > reward_1

			# 分类统计accuracy
			for keyword in category_keywords:
				if keyword in item.get("image", "") or keyword in item.get("video", ""):
					if correct:
						category_stats[keyword]["accuracy"] += 1
					else:
						all_correct = False
					break

			# 总体统计accuracy
			if correct:
				overall_stats["accuracy"] += 1
			else:
				all_correct = False

		# 更新acc+统计
		if all_correct:
			for keyword in category_keywords:
				if any(keyword in item.get("image", "") or keyword in item.get("video", "") for item in items):
					category_stats[keyword]["acc_plus"] += 1
					break
			overall_stats["acc_plus"] += 1

	# 计算每个类别的accuracy和acc+
	for keyword, stats in category_stats.items():
		if stats["total"] > 0:
			stats["accuracy"] = stats["accuracy"] / stats["total"]
			stats["acc_plus"] = stats["acc_plus"] / len(
				[item_id for item_id in id_to_items if any(keyword in (item.get("image", "") + item.get("video", "")) for item in id_to_items[item_id])]
			)

	# 计算总体accuracy和acc+
	if overall_stats["total"] > 0:
		overall_stats["accuracy"] = overall_stats["accuracy"] / overall_stats["total"]
		overall_stats["acc_plus"] = overall_stats["acc_plus"] / len(id_to_items)

	# 输出结果
	def print_metrics():
		print("\nCategory-wise Metrics:")
		for keyword, stats in category_stats.items():
			print(f"Category: {keyword}")
			print(f"  Accuracy: {stats['accuracy']:.5f}")
			print(f"  ACC+: {stats['acc_plus']:.5f}")
			print(f"  Total: {stats['total']}")

		print("\nOverall Metrics:")
		print(f"Overall Accuracy: {overall_stats['accuracy']:.5f}")
		print(f"Overall ACC+: {overall_stats['acc_plus']:.5f}")
		print(f"Total Items: {overall_stats['total']}")

	# 输出
	print_metrics()

def contains_chinese(text):
    pattern = regex.compile(r'\p{IsHan}')
    return bool(pattern.search(text))

def compute_accuracy(lst):
    count1=0;
    count2=0;
    count3=0
    for i in range(len(lst)):
        obj=lst[i] 
        # obj1=lst1[i]
        # if obj1["task"]!="caption":
        #     continue;
        if obj["chosen"]>obj["rejected"]:
            count1+=1
        elif obj["chosen"]==obj["rejected"]:
            count2+=1
        else:
            count3+=1;
    print("total:",len(lst),"correct:",count1,"tie:",count2,"false:",count3)
    print("**accuracy**:",count1/len(lst))
    return count1

def get_dataset_from_id(id:str):
    dataset=["povid","reasoning_tasks","rlaif-v","rlhf-v","vlfeedback","wildvision-battle"]
    def get_id_prefix(id_value:str):
        split_index = min((id_value.find('_'), id_value.find('-')), key=lambda x: x if x != -1 else float('inf'))
        if split_index != -1:
            id_prefix = id_value[:split_index]
        else:
            id_prefix = id_value
            
        return id_prefix
    
    id_prefix=get_id_prefix(id)
    if id_prefix == "RLAIF":
        return "rlaif-v"
    elif id_prefix == "RLHF":
        return "rlhf-v"
    elif id_prefix == "mathverse" or id_prefix == "mmmu":
        return "reasoning_tasks"
    elif id_prefix == "wildvision":
        return "wildvision-battle"
    elif id_prefix == 'hallucination':
        return "povid" # fix povid bug 
    else:
        return "vlfeedback"

group_mapping = {
    "vlfeedback": 0,
    "povid": 1,
    "reasoning_tasks": 2,
    "rlhf-v": 1,
    "rlaif-v": 1,
    "wildvision-battle": 0
}

def compute_vlrewardbench(lst,lst1):
    correct_info=[0,0,0]#general,hallucination,reasoning
    total_num_info=[0,0,0]
    category=["general","hallucination","reasoning"]
    for i in range(int(len(lst))):
        category_index=group_mapping[get_dataset_from_id(lst1[i]["id"])]
        total_num_info[category_index]+=1
        if lst[i]["chosen"]>lst[i]["rejected"]:
                correct_info[category_index]+=1
    print(correct_info,total_num_info)
    accuracy=[]
    for i in range(3):
        if total_num_info[i]==0:
            continue;
        acc=correct_info[i]/total_num_info[i]
        print(category[i],":",acc)
        accuracy.append(acc)
    all_correct_num=sum(correct_info)
    total_num=sum(total_num_info)
    print("overall accuracy:",all_correct_num/total_num)
    print("macro accuracy:", sum(accuracy)/len(accuracy))

def compute_multimodal_rewardbench(lst,lst1):
    correct=defaultdict(int)
    total_num=defaultdict(int)
    for i in range(len(lst1)):
        obj=lst1[i]
        obj_output=lst[i]
        total_num[obj["category"]]+=1;
        datasetname=obj["image_path"].split("/")[1]
        if datasetname=="mathvista":
            total_num["math"]+=1;
        elif datasetname in ['EMMA-Coding', 'image2struct']:
            total_num["coding"]+=1;
        if obj_output["chosen"]>obj_output["rejected"]:
            correct[obj["category"]]+=1;
            if datasetname=="mathvista":
                correct["math"]+=1;
            elif datasetname in ['EMMA-Coding', 'image2struct']:
                correct["coding"]+=1;
    test_set_name=["open_generation/correctness_task","open_generation/preference_task","knowledge","math","coding","safety","vqa"]
    all_correct_num=0;
    all_num=0;
    for name in test_set_name:
        print(name,correct[name]/total_num[name],correct[name],total_num[name])
        all_correct_num+=correct[name]
        all_num+=total_num[name]
    print("overall acc:",all_correct_num/all_num,"correct:",all_correct_num,"total:",all_num)
    print("overall acc(without safety):",(all_correct_num-correct["safety"])/(all_num-total_num["safety"]),"correct:",all_correct_num-correct["safety"],"total:",all_num-total_num["safety"])

def get_min_max_score(lst):
    temp_set=set()
    for elem in lst:
        temp_set.add(elem["chosen"])
        temp_set.add(elem["rejected"])
    print("max_score:",max(temp_set),"min_score:",min(temp_set))

if __name__ == '__main__':
    correct_num=0
    all_result=[]
    accuracy=0;
    for filename in output_path:
        all_result.extend([json.loads(line.strip()) for line in open(filename,'r',encoding='utf-8')])
    get_min_max_score(all_result)
    total_sample_num=len(all_result)
    data_num=[170,1247,4711]
    test_name=["MM-Reward-Bench(170):","VL-RewardBench(1247):","Multimodal-RewardBench(4771):"]
    index=0;
    for i in range(len(data_num)):
        print("-------------",test_name[i],"-------------")
        #读取输出的结果
        lst_output=all_result[index: index+data_num[i]]
        if test_name[i]=="MM-Reward-Bench(170):":
            lst_gt=[json.loads(line.strip()) for line in open(gt_path[i],'r',encoding='utf-8')]
            compute_mmrewardbench(lst_output, lst_gt)
        elif test_name[i]=="VL-RewardBench(1247):":
            lst_gt=[json.loads(line.strip()) for line in open(gt_path[i],'r',encoding='utf-8')]
            compute_vlrewardbench(lst_output,lst_gt)
        elif test_name[i]=="Multimodal-RewardBench(4771):":
            lst_gt=[json.loads(line.strip()) for line in open(gt_path[i],'r',encoding='utf-8')]
            compute_multimodal_rewardbench(lst_output,lst_gt)
        correct_count=compute_accuracy(lst_output)
        accuracy+=correct_count/data_num[i]
        correct_num+=correct_count
        print()
        index+=data_num[i]
    print("all correct num:",correct_num,"all data num:",total_sample_num)
    print("accuracy on whole test set:",correct_num/total_sample_num)
    print("average accuracy:",accuracy/3.0)
