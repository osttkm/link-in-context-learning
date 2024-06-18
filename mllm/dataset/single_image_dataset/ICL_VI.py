
import os
import os.path as osp
import jsonlines
import random
import string
from pathlib import Path
from inspect import isfunction
from typing import Dict, Any, Sequence
from copy import deepcopy
import numpy as np
import math
import json
import cv2 as cv
from collections import defaultdict
import torch.distributed as dist
from .lcl import LCLDataset, LCLComputeMetrics, logger, LABEL_PLACEHOLDER
from ..root import (
    DATASETS,
    METRICS,
    EXPR_PLACEHOLDER
)

import torch.distributed as dist

def get_random_string():
    return ''.join(random.choices(string.ascii_uppercase, k=random.randint(1,10))).lower()

@DATASETS.register_module()
class CustomDataset_ICLVI_Train(LCLDataset):
    def __init__(self, lcl, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('==========TASK ICL_VI is initialized==========')
        self.sampled_defect_name=None
        self.lcl = lcl
        self.policy = "policy_2way_weight"

        custom_json = None
        # jsonファイルの読み込み
        with open(self.template_file, 'r') as f:
            print(f'load template file from :{self.template_file}')
            custom_json = json.load(f)
        
        self.context_question = custom_json['VI_instructions'][0]
        self.yes_answer = custom_json['answers_ng'][0]
        self.no_answer = custom_json['answers_ok'][0]
        if self.lcl:
            self.query_question = custom_json['query_VI_instructions'][0]

    
    def get_product_name(self, item):
        # # imagenetの変換
        # task_id = item['task_id']
        # if task_id == 3 and (self.task == "PG+VI" or self.task == "LCL_PG+VI"):
        #     self.product_name = item['class_name']
        # # 自作データの変換
        # else:
        self.product_name = item['name'].split('+')[0]
        self.product_name = self.product_name.replace('_', ' ')
        return self.product_name
        


    def get_samples(self, index, mode="cls_negative"):
        """
        neighbors...同じ製品で異なる欠陥を持つサンプル
        samples...同じ製品で同じ欠陥を持つサンプル
        
        mode(imagenetの話)
        cls_negative ー＞ neighborsをサンプル
        neighbors ー＞ samplesをサンプル
        Imagenetではsamplesが同一クラスでneighborsが異なるクラスの画像だった.neighborはsamplesとクエリ以外のクラスでかつ特徴空間で最も近いクラスから選択される
        """
        assert mode in ['cls_negative', 'neighbors']

        item = self.get_raw_item(index)
        samples = item['samples']
        neighbors = item['neighbors']
        # 欠陥名（自作データのみ）
        name = item['class_name']

        self.class_defect = [p.name for p in Path(samples[0]).parent.parent.iterdir() if p.is_dir()]
        self.class_defect = [i for i in self.class_defect if i != 'None']
        self.product_name = self.get_product_name(item)

        # """一枚目の画像をneighborsから選択"""
        if mode == "cls_negative":
            if name == 'None':
                # 欠陥品を選択．
                # itemのneighborは辞書型配列の0番目はNoneなのでそれ以外から選択
                random_idx = random.randint(1, len(neighbors)-1)
                sample = random.choice(neighbors[random_idx]['data'])
                label = neighbors[random_idx]['name'].lower()
            else:
                # 正常品を選択．
                # itemのneighborは辞書型配列の最初にNoneの辞書が入っている
                sample = random.choice(neighbors[0]['data'])
                label = "None".lower()
            self.sampled_defect_name = label

                
        # """二枚目の画像をsamplesから選択"""
        elif mode == "neighbors":
            """製品当てタスクではない処理"""
            if self.sampled_defect_name is not None: # VIのコンテキスト画像選択
                label = name.lower()
                sample = random.choice(samples)
            else: # VIのクエリ画像選択
                if name == 'None':
                    random_idx = random.randint(1, len(neighbors)-1)
                    sample = random.choice(neighbors[random_idx]['data'])
                    label = neighbors[random_idx]['name'].lower()
                else:
                    sample = random.choice(neighbors[0]['data'])
                    label = "None".lower()
                self.sampled_defect_name = None
        
            
        else:
            raise NotImplementedError
        image = self.get_image(sample)
        return image, label

    # get policy function according to name 
    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        return func(index, shot)
    """
    cls negative...Imagenetでは違うクラスの画像でかつ特徴量が似通っている画像だと思われる
    neighbors...samplesの中から持ってきている同じクラスのものだと思われる
    """
    def get_subfolder_strings(self, class_defect:list):
            subfolder_string = ''
            if len(class_defect) == 1:
                return class_defect[0]
            else:
                subfolder_string = ', '.join(class_defect[:-1])
                if len(class_defect) > 1:
                    subfolder_string += ' and ' + class_defect[-1]
                return subfolder_string

    def custom_convert_question(self, question):
            q = question.format(subfolder_string=self.get_subfolder_strings(self.class_defect),product=self.product_name)
            return q
    
    def custom_convert_answer(self, label, mode):
        assert mode in ['cls_negative', 'neighbors']

        # Create a defaultdict that returns an empty string for non-existing keys
        safe_dict = defaultdict(str)
        safe_dict.update({'product': self.product_name, 'subfolder_string': self.get_subfolder_strings(self.class_defect), 'defect': label})
        
        if label == 'none' or label == "None":
            idx = random.randint(0, len(self.no_answer)-1)
            answer = self.no_answer[idx].format_map(safe_dict)
        else:
            idx = random.randint(0, len(self.yes_answer)-1)
            answer = self.yes_answer[idx].format_map(safe_dict)

        answer = answer + ' [END EXAMPLE]'
        return answer

    def policy_2way_weight(self, index, shot):

        """ set context samples"""
        ret_list = []

        for mode in ['cls_negative', 'neighbors']:
            for _ in range(shot):
                image, label = self.get_samples(index, mode = mode)

                idx = random.randint(0, len(self.context_question)-1)
                mix_question = ' [BEGIN EXAMPLE] '+self.context_question[idx]
                mix_question = self.custom_convert_question(question=mix_question)
                
                answer = self.custom_convert_answer(label, mode = mode)
                ret = self.get_ret(image, question = mix_question, answer = answer, conv_mode = 'hypnotized_ans_v1.0')
                ret_list.append(ret)
            

        random.shuffle(ret_list)
        ret_list[0]['mode'] = 'causal_v1.0'

        """ set inference sample """
        mode = "neighbors"
        image, label = self.get_samples(index, mode = mode)
        infer_question = ''
        if self.lcl:
            idx = random.randint(0, len(self.query_question)-1)
            infer_question = self.query_question[idx]
        else:
            idx = random.randint(0, len(self.context_question)-1)
            infer_question = self.context_question[idx]
        mix_question = self.custom_convert_question(question=infer_question)
    
        """回答文章の生成"""
        answer = self.custom_convert_answer(label, mode = mode).replace(" [END EXAMPLE]", '')
        ret = self.get_ret(image, question = infer_question, answer = answer, conv_mode="final_v1.0")
        ret_list.append(ret)

        return ret_list

