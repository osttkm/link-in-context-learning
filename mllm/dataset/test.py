import random

def init_dict(num):
    # 辞書の初期化
    dict = {}
    dict['index'] = num
    return dict

def partial_shuffle(lst):
    # 3の倍数以外のインデックスの要素を取得
    non_multiples = [i for i in range(len(lst)) if i % 3 != 0]

    # これらの要素をシャッフル
    shuffled_values = [lst[i] for i in non_multiples]
    random.shuffle(shuffled_values)

    # シャッフルされた要素を元のリストに戻す
    for i, value in zip(non_multiples, shuffled_values):
        lst[i] = value
    return lst

# 使用例
ret_list = []
ret_list.append(init_dict(0))
ret_list.append(init_dict(1))
ret_list.append(init_dict(2))
ret_list.append(init_dict(3))
ret_list.append(init_dict(4))
ret_list.append(init_dict(5))
print(partial_shuffle(ret_list))