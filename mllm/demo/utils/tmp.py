from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


# num_shot = 1
model_paths = ["/home/oshita/vlm/Link-Context-Learning/LCL_2WAY_WEIGHT/"]

result = {}

for path in model_paths:
    add_names = [f"result_shot{i}" for i in range(1,11,1)]
    if not Path(path).parent.name in result:
        result[Path(path).parent.name] = []
    for name in add_names:
        json_path = Path(path+name) / 'all_results.json'
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            score = data['multitest_ImageNet1k_test100_accuracy']
            result[Path(path).parent.name].append(score)
        except:
            result[Path(path).parent.name].append(0.0)

# 折れ線グラフの描画    
ex_names = [name for name in result.keys()]
for name in ex_names:
    plt.plot(np.arange(1, len(result[name])+1),result[name], label=name)
plt.xticks(np.arange(1, len(result[name])+1))
plt.xlabel('Number of shots')
plt.ylabel('Classification Accuracy on Imagenet1k')
plt.legend()
plt.savefig(f'Default_result.png')
