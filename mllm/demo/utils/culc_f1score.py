import re
import numpy as np
import argparse
from pathlib import Path

argparse = argparse.ArgumentParser()
argparse.add_argument('--txt_path', type=str, default='/dataset/mvtec')
args = argparse.parse_args()


def extract_accuracy_rates(text,speacis):
    pattern = rf"{speacis} : (\w+) =.*?正常品に対する正答率 : (\d+)/(\d+).*?欠陥品に対する正答率 : (\d+)/(\d+)"
    matches = re.findall(pattern, text, re.DOTALL)
    results = {}
    for match in matches:
        defect_type = match[0]

        # Calculate tp, fn for defective products
        tp = int(match[3])
        total_defective = int(match[4])
        fn = total_defective - tp

        # Calculate fp, tf for normal products
        tf = int(match[1])
        total_normal = int(match[2])
        fp = total_normal - tf
        results[defect_type] = {"tp": tp, "fn": fn, "fp": fp, "tf": tf}
    return results

def calculate_f1_score_from_accuracy_and_recall(tp,fp,fn):
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp/(tp+fn)
    
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp/(tp+fp)
        
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)

def write_txt(path,text):
    with open(path, "a") as f:
        f.write(f'{text}\n')

# txt_paths = ['/home/oshita/vlm/Link-Context-Learning/mllm/demo/result/mvtec/{s}/'  \
# +Path(args.txt_path).name+f'_checkpoint-epoch-{i*10}.txt' for i in range(1,6)]

txt_paths = ['/home/oshita/vlm/Link-Context-Learning/mllm/demo/result/mvtec/{s}/'  \
+Path(args.txt_path).name+f'_checkpoint-epoch-{i}.txt' for i in range(1,9)]
for s in ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]:
    for txt_path in txt_paths:
        if not Path(txt_path.format(s=s)).exists():
            txt_paths.remove(txt_path)
            print(f"Removed {txt_path.format(s=s)} from the list of paths because it does not exist.")
        else:
            pass

# txt_paths.remove('/home/oshita/vlm/Link-Context-Learning/mllm/demo/result/mvtec/{s}/LCL_VI_FIX_30_checkpoint-epoch-50.txt')
mvtec_speacis = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


for speacis in mvtec_speacis:
    print(f'==={speacis}===')
    for idx,txt_path in enumerate(txt_paths):
        save_path = txt_path[:-4]+'_f1_score.txt'
        # defect_nameをキーにもつ辞書を作成
        with open(txt_path.format(s=speacis), "r") as f:
            txt = f.read()
        defect_names = re.findall(rf"{speacis} : (\w+)".format(s=speacis), txt)
        defect_name_dict = {}
        for defect_name in defect_names:
            defect_name_dict[defect_name] = []
        defect_name_dict['class_average'] = []
        print(txt.format(s=speacis))
        with open(txt_path.format(s=speacis), "r") as f:
            txt = f.read()
        extracted_data = extract_accuracy_rates(txt,speacis)
        f1_scores = []
        for i,key in enumerate(extracted_data.keys()):
            tp, fn, fp, _ = extracted_data[key].values()
            print(key)
            f1_score = calculate_f1_score_from_accuracy_and_recall(tp,fp,fn)
            f1_scores.append(f1_score)
            defect_name_dict[defect_names[i]].append(f1_scores[i])
        defect_name_dict['class_average'].append(np.mean(f1_scores))

        
        class_ave = round(np.mean(defect_name_dict['class_average']),2)
        class_std = round(np.std(defect_name_dict['class_average']),2)
        print(f'''class average: {class_ave} ± {class_std}''')
        with open(f'{save_path.format(s=speacis)}', "w") as f:
            f.close()
        write_txt(save_path.format(s=speacis),f'''class average: {class_ave} ± {class_std}''')
        for key in defect_name_dict.keys():
            if key != 'class_average':
                defect_ave = round(np.mean(defect_name_dict[key]), 2)
                defect_std = round(np.std(defect_name_dict[key]), 2)
                print(f'''{key}: {defect_ave} ± {defect_std}''')
                write_txt(save_path.format(s=speacis),f'''{key}: {defect_ave} ± {defect_std}''')

