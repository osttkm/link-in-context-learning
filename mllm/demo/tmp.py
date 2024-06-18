import pandas as pd
mvtec = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']

for speacis in mvtec:
    path = "/home/oshita/vlm/Link-Context-Learning/mllm/demo/result/mvtec/{s}/{s}.csv"
    data = pd.read_csv(path.format(s=speacis))
    
    top_10 = data['F1 average'].nlargest(10)
    top_10_index = top_10.index.tolist()
    ex_name = [data['Unnamed: 0'][data['Unnamed: 0'].index==idx].item() for idx in top_10_index]
    
    for i in range(10):
        print(f'F1 score:{top_10.tolist()[i]}  {ex_name[i]}')
    import pdb;pdb.set_trace()
