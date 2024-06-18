
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
class CustomDatasetTrain_AC(LCLDataset):
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('==========TASK AC is initialized==========')
        self.sampled_defect_name=None

        self.policy = policy
        custom_json = None
        # jsonファイルの読み込み
        with open('/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/AC.json', 'r') as f:
            custom_json = json.load(f)
       
        self.context_question = custom_json['context_question_lines']
        self.yes_answer = custom_json['yes_responses_array']
        self.no_answer = custom_json['no_responses_array']

    

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


        # """一枚目の画像をneighborsから選択"""
        if mode == "cls_negative":
            # itemの持つ欠陥クラスとは異なる欠陥クラスをneighborから選択
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
                # print(f'===== first select {self.sampled_defect_name} ___ second select : {label}=====')
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

    def policy_2way_weight(self, index, shot):
        def custom_convert_answer(label, mode, idx):
            assert mode in ['cls_negative', 'neighbors']
            if label == 'none' or label == "None":
                answer = self.ac_no_answer[idx][0]
            else:
                answer = self.ac_yes_answer[idx][0].format(defect=label)
            answer = answer + ' [END EXAMPLE]'
            return answer

        """ set context samples"""
        ret_list = []

        for mode in ['cls_negative', 'neighbors']:
            for _ in range(shot):
                image, label = self.get_samples(index, mode = mode)
                idx = random.randint(0, len(self.ac_context_question)-1)
                mix_question = ' [BEGIN EXAMPLE] '+self.ac_context_question[idx][0]
                ret = self.get_ret(image, question = mix_question, answer = answer, conv_mode = 'hypnotized_ans_v1.0')
                ret_list.append(ret)

        random.shuffle(ret_list)
        ret_list[0]['mode'] = 'causal_v1.0'

        """ set inference sample """
        mode = "neighbors"
        image, label = self.get_samples(index, mode = mode)
        idx = random.randint(0, len(self.ac_context_question)-1)
        infer_question = self.ac_query_question[idx][0]
        

    
        """回答文章の生成"""
        answer = custom_convert_answer(label, mode = mode, idx = idx).replace(" [END EXAMPLE]", '')
        ret = self.get_ret(image, question = infer_question, answer = answer, conv_mode="final_v1.0")
        ret_list.append(ret)

        return ret_list

