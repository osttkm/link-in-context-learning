
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
class CustomDatasetTrain_LCL_PG(LCLDataset):
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('==========TASK LCL_PG is initialized==========')
        self.sampled_defect_name=None
        self.productA = None
        self.productA_idx = None
        self.productA_question = None
        self.productA_answer = None

        self.productB = None
        self.productB_idx = None
        self.productB_question = None
        self.productB_answer = None

        self.policy = policy
        self.cls_map = self.get_cls_map()
        custom_json = None
        # jsonファイルの読み込み
        with open('/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/LCL_PG.json', 'r') as f:
            custom_json = json.load(f)

        self.pg_context_question = custom_json['PG_instructions']
        self.pg_answer = custom_json['PG_answer']
        self.pg_no_answer = custom_json['PG_no_answer']

            
    def reset_productAB(self):
        self.productA = None
        self.productA_idx = None
        self.productA_question = None
        self.productA_answer = None

        self.productB = None
        self.productB_idx = None
        self.productB_question = None
        self.productB_answer = None

    
    def get_cls_map(self):
        # Map origin ImageNet1k class_id to Train900 index
        cls_map = {}
        for id, item in enumerate(self.data):
            cls_id = item["class_id"]
            if cls_id not in cls_map.keys():
                cls_map[cls_id] = id
            else:
                logger.warning("Class id conflict.")
        return cls_map
    
    def get_product_name(self, item):
        # # imagenetの変換
        # task_id = item['task_id']
        # if task_id == 3 and (self.task == "PG+VI" or self.task == "LCL_PG+VI"):
        #     self.product_name = item['class_name']
        # # 自作データの変換
        # else:
        #     self.product_name = item['name'].split('+')[0]
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
        self.pg_product_name = None

        # """一枚目の画像をneighborsから選択"""
        if mode == "cls_negative":
            sample = random.choice(samples)
            self.pg_product_name = self.product_name
            self.productA = self.product_name
            self.productA_idx = index
            label = "ERROR"
                
        # """二枚目の画像をsamplesから選択"""
        elif mode == "neighbors":

            label = "ERROR"
            """製品当てタスク処理"""
            if self.productA is None or self.productB is None:
                """コンテキストの二枚選択時"""
                index = np.arange(0, 1233)
                # Imagenetのように製品当てをする場合はjsonにデータを追加の上，以下のようにインデックスを追加する
                # index = np.append(index, np.arange(2466, 2695))
                
                index = random.choice(index)
                item = self.get_raw_item(index)
                samples = item['samples']
                self.product_name = self.get_product_name(item)
                sample = random.choice(samples)
                self.pg_product_name = self.product_name
                self.productB = self.product_name
                self.productB_idx = index
            elif self.productA is not None and self.productB is not None:
                    """
                    クエリ画像を選択時
                    50%で以下の1,2を実行
                    1 : コンテキストにない画像選択
                    2 : コンテキストのクラスを選択,ただしABの確率は25%ずつ.おそらくここは処理しないといけない
                    """
                    random_AB_number = random.randint(0,1)
                    random_AB_or_other_number = random.randint(0,1)
                    if random_AB_or_other_number == 0:
                        index = np.arange(0, 1233)
                        # index = np.arange(0, 1233)
                        # index = np.append(index, np.arange(2466, 2695))
                        index = index[index!=self.productA_idx]
                        index = index[index!=self.productB_idx]
                        index = random.choice(index)
                        item = self.get_raw_item(index)
                        samples = item['samples']
                        self.product_name = self.get_product_name(item)
                        sample = random.choice(samples)
                    else:
                        if random_AB_number == 0:
                            item = self.get_raw_item(self.productA_idx)
                            samples = item['samples']
                            self.product_name = self.get_product_name(item)
                            sample = random.choice(samples)
                        else:
                            item = self.get_raw_item(self.productB_idx)
                            samples = item['samples']
                            self.product_name = self.get_product_name(item)
                            sample = random.choice(samples)
        
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

        def custom_convert_answer(mode, idx):
            assert mode in ['cls_negative', 'neighbors']

            if self.productA == self.product_name and self.productB != self.product_name:
                answer = self.pg_answer[idx][0].format(product=self.productA)
            elif self.productB == self.product_name and self.productA != self.product_name:
                answer = self.pg_answer[idx][0].format(product=self.productB)
            elif self.productA == self.product_name and self.productB == self.product_name:
                answer = self.pg_answer[idx][0].format(product=self.productA)
            else:
                answer = self.pg_no_answer[idx][0].format(productA=self.productA,productB=self.productB)

            answer = answer + ' [END EXAMPLE]'
            return answer

        """ set context samples"""
        ret_list = []

        for mode in ['cls_negative', 'neighbors']:
            for _ in range(shot):
                image, label = self.get_samples(index, mode = mode)
                
                idx = random.randint(0, len(self.pg_context_question)-1)
                mix_question = ' [BEGIN EXAMPLE] '+self.pg_context_question[idx][0]
                answer = custom_convert_answer(mode = mode, idx = idx)
                ret = self.get_ret(image, question = mix_question, answer = answer, conv_mode = 'hypnotized_ans_v1.0')
                ret_list.append(ret)
            

        random.shuffle(ret_list)
        ret_list[0]['mode'] = 'causal_v1.0'

        """ set inference sample """
        mode = "neighbors"
        image, label = self.get_samples(index, mode = mode)
        idx = random.randint(0, len(self.pg_query_question)-1)
        infer_question = self.pg_query_question[idx][0]
    
        """回答文章の生成"""
        answer = custom_convert_answer(mode = mode, idx = idx).replace(" [END EXAMPLE]", '')
        # print(f'LCL_PG_TASK\n\ninfer_question:{mix_question}\ninfer_answer:{answer}\n')
        ret = self.get_ret(image, question = infer_question, answer = answer, conv_mode="final_v1.0")
        ret_list.append(ret)

        self.reset_productAB()
        return ret_list

