import argparse
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument('--txt_path', type=str, default='/dataset/mvtec')
args = argparser.parse_args()

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

# import pdb;pdb.set_trace()
data_path = Path('/dataset/mvtec')
species = [p.name for p in data_path.iterdir() if p.is_dir()]
species.remove('bottle_test')
for txt_path in txt_paths:
    for s in species:
        file_path = txt_path.format(s=s)
        with open(file_path, 'r') as file:
            file_contents = file.readlines()
        defect_name = []
        result = {}
        for f in file_contents:
            if f.split('/')[-1][0:3]!="000":
                print(f.split(' ')[0].split('/'))
                if f.split(' ')[0].split('/')[-2] != 'good':
                    defect_name = f.split(' ')[0].split('/')[-2]
                    good_name = ""
                elif f.split(' ')[0].split('/')[-2] == 'good':
                    good_name = f.split(' ')[0].split('/')[-2]

                if defect_name not in result.keys():
                    result[defect_name] = {'good': 0,'good_total':0 ,'defect_total':0,'defect': 0}

                if good_name == "":
                    result[defect_name]['defect_total'] += 1
                    if f.split(' ')[1] == "Yes.":
                        result[defect_name]['defect'] += 1

                elif good_name == "good":
                    result[defect_name]['good_total'] += 1
                    if f.split(' ')[1] == "No.":
                        result[defect_name]['good'] += 1
            else:
                # print('NO COUNT')
                pass

        for key in list(result.keys()):
            print(f'========== {s} : {key} ==========')
            print(f'正常品に対する正答率 : {result[key]["good"]}/{result[key]["good_total"]}__{((result[key]["good"]/result[key]["good_total"])*100):.2f}%')
            print(f'欠陥品に対する正答率 : {result[key]["defect"]}/{result[key]["defect_total"]}__{((result[key]["defect"]/result[key]["defect_total"])*100):.2f}%')
            # 上記の3つのprintの内容を'/home/oshita/vlm/Link-Context-Learning/mllm/demo/result/mvtec/{s}/LCL_VI_log.txtに追記する
            with open(file_path, 'a') as file:
                file.write(f'========== {s} : {key} ==========\n')
                file.write(f'正常品に対する正答率 : {result[key]["good"]}/{result[key]["good_total"]}__{((result[key]["good"]/result[key]["good_total"])*100):.2f}%\n')
                file.write(f'欠陥品に対する正答率 : {result[key]["defect"]}/{result[key]["defect_total"]}__{((result[key]["defect"]/result[key]["defect_total"])*100):.2f}%\n')
