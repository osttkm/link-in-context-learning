import json
import numpy as np

def get_dict():
    return {
        "class_id":"",
        "class_name":"",
        "samples":[],
        }

def write(all_train_dict, test_path):
    train_path = "/home/oshita/vlm/Link-Context-Learning/docs/Imagenet_classify_train.jsonl"
    test_path = "/home/oshita/vlm/Link-Context-Learning/docs/Imagenet_classify_test.jsonl"

    # all_train_dictとall_test_dictをjsonl形式で保存
    with open(train_path, "w") as f:
        for key,items in all_train_dict.items():
            f.write(json.dumps(items) + "\n")
    with open(test_path, "w") as f:
        for key,items in all_test_dict.items():
            f.write(json.dumps(items) + "\n")

all_train_dict = {}
all_test_dict = {}




train_jsonl_path = "/home/oshita/vlm/Link-Context-Learning/docs/train900_pairs.jsonl"
with open(train_jsonl_path, "r") as f:
    for line in f:
        data = json.loads(line)
        class_id = data["class_id"]
        class_name = data["class_name"]
        samples = data["samples"]
        neighbors = data["neighbors"]

        if class_name not in all_train_dict:
            all_train_dict[class_name] = get_dict()
            all_train_dict[class_name]["class_id"] = class_id
            all_train_dict[class_name]["class_name"] = class_name
        if class_name not in all_test_dict:
            all_test_dict[class_name] = get_dict()
            all_test_dict[class_name]["class_id"] = class_id
            all_test_dict[class_name]["class_name"] = class_name

        for sample in samples:
            if "train" in sample: 
                if sample not in all_train_dict[class_name]["samples"]:   
                    all_train_dict[class_name]["samples"].append(sample)
                else:
                    pass
            elif "train" not in sample:
                if sample not in all_test_dict[class_name]["samples"]:
                    all_test_dict[class_name]["samples"].append(sample)
                else:
                    pass

        for neighbor in neighbors:
            class_id = neighbor[0]
            class_name = neighbor[1]
            sample = neighbor[2]
            if class_name not in all_train_dict:
                all_train_dict[class_name] = get_dict()
                all_train_dict[class_name]["class_id"] = class_id
                all_train_dict[class_name]["class_name"] = class_name
            if class_name not in all_test_dict:
                all_test_dict[class_name] = get_dict()
                all_test_dict[class_name]["class_id"] = class_id
                all_test_dict[class_name]["class_name"] = class_name

            if "train" in sample: 
                if sample not in all_train_dict[class_name]["samples"]:
                    all_train_dict[class_name]["samples"].append(sample)
                else:
                    pass
                
            if "train" not in sample:
                if sample not in all_test_dict[class_name]["samples"]:
                    all_test_dict[class_name]["samples"].append(sample)
                else:
                    pass


# all_train_dictとall_test_dictのキーの長さを図る
print(len(all_train_dict.keys()))
print(len(all_test_dict.keys()))
test_jsonl_path = "/home/oshita/vlm/Link-Context-Learning/docs/test100_pairs.jsonl"
with open(test_jsonl_path, "r") as f:
    for line in f:
            data = json.loads(line)
            class_id = data["class_id"]
            class_name = data["class_name"]
            context_samples = data["context_samples"]
            test_samples = data["test_samples"]

            if class_name not in all_train_dict:
                all_train_dict[class_name] = get_dict()
                all_train_dict[class_name]["class_id"] = class_id
                all_train_dict[class_name]["class_name"] = class_name
            if class_name not in all_test_dict:
                all_test_dict[class_name] = get_dict()
                all_test_dict[class_name]["class_id"] = class_id
                all_test_dict[class_name]["class_name"] = class_name


            for context_sample in context_samples:
                if "train" in context_sample:
                    if context_sample not in all_train_dict[class_name]["samples"]:   
                        all_train_dict[class_name]["samples"].append(context_sample)
                    else:
                        pass
                elif "train" not in context_sample:
                    if context_sample not in all_test_dict[class_name]["samples"]:   
                        all_test_dict[class_name]["samples"].append(context_sample)
                    else:
                        pass

            for test_sample in test_samples:
                if "train" in test_sample:
                    if test_sample not in all_train_dict[class_name]["samples"]:   
                        all_train_dict[class_name]["samples"].append(test_sample)
                    else:
                        pass
                elif "train" not in test_sample:
                    if test_sample not in all_test_dict[class_name]["samples"]:   
                        all_test_dict[class_name]["samples"].append(test_sample)
                    else:
                        pass


print(len(all_train_dict.keys()))
print(len(all_test_dict.keys()))

write(all_train_dict, all_test_dict)


        