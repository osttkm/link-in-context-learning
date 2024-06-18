from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


shot_list = [1,2,3,4,5,6,7,8,9,10]
checkpoints = ['500','1000','1500','2000','2500', '2850']
ex1 = "/home/oshita/vlm/Link-Context-Learning/NON_MIX_LCL_2WAY_WEIGHT/last_image_token_result/"
ex2 = "/home/oshita/vlm/Link-Context-Learning/SHIFT_IMAGE_TOKEN_HEAD_NON_MIX_LCL_2WAY_WEIGHT/last_image_token_result/"
ex3 = "/home/oshita/vlm/Link-Context-Learning/SHIFT_IMAGE_TOKEN_TAIL_NON_MIX_LCL_2WAY_WEIGHT/last_image_token_result/"
# ex1 = "/home/oshita/vlm/Link-Context-Learning/NON_MIX_LCL_2WAY_WEIGHT/first_image_token_result/"
# ex2 = "/home/oshita/vlm/Link-Context-Learning/SHIFT_IMAGE_TOKEN_HEAD_NON_MIX_LCL_2WAY_WEIGHT/first_image_token_result/"
# ex3 = "/home/oshita/vlm/Link-Context-Learning/SHIFT_IMAGE_TOKEN_TAIL_NON_MIX_LCL_2WAY_WEIGHT/first_image_token_result/"
default = '/home/oshita/vlm/Link-Context-Learning/LCL_2WAY_WEIGHT/'

for checkpoint in checkpoints:
    result = {}
    for shot in shot_list:
        if not Path(ex1).parent.name in result:
            result[Path(ex1).parent.name] = []
        if not Path(ex2).parent.name in result:
            result[Path(ex2).parent.name] = []
        if not Path(ex3).parent.name in result:
            result[Path(ex3).parent.name] = []
        if not Path(default).name in result:
            result[Path(default).name] = []

        add_names = f"/result_shot{shot}/{checkpoint}"
        json_path1 = Path(ex1+add_names) / 'all_results.json'
        json_path2 = Path(ex2+add_names) / 'all_results.json'
        json_path3 = Path(ex3+add_names) / 'all_results.json'
        default_json_path = Path(f"{default}result_shot{shot}") / 'all_results.json'
        for json_path in [json_path1, json_path2,default_json_path, json_path3]:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                score = data['multitest_ImageNet1k_test100_accuracy']
                if json_path != default_json_path:
                    result[Path(json_path).parent.parent.parent.parent.name].append(score)
                else:
                    result[Path(json_path).parent.parent.name].append(score)
                # import pdb; pdb.set_trace()
            except:
                result[Path(json_path).parent.parent.parent.name].append(0.0)
            
    # import pdb; pdb.set_trace()
    # 折れ線グラフの描画    
    ex_names = [name for name in result.keys()]
    for name in ex_names:
        plt.plot(np.arange(1, len(result[name])+1),result[name], label=name)
    plt.xticks(np.arange(1, len(result[name])+1))
    plt.xlabel('Number of shots')
    plt.ylabel('LAST IMAGE TOKEN Classification Accuracy on Imagenet1k')
    # plt.ylabel('FIRST IMAGE TOKEN Classification Accuracy on Imagenet1k')
    plt.legend()
    plt.savefig(f'LAST_IMAGE_TOKEN_Default_checkpoint{checkpoint}.png')
    # plt.savefig(f'FIRST_IMAGE_TOKEN_Default_checkpoint{checkpoint}.png')
    plt.clf()
