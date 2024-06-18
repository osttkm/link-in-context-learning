import torch.distributed as dist

import os
import os.path as osp
import jsonlines
import random
import string
from inspect import isfunction
from typing import Dict, Any, Sequence
from copy import deepcopy
import numpy as np
import math
import cv2 as cv
from .lcl import LCLDataset, LCLComputeMetrics, logger, LABEL_PLACEHOLDER
from ..root import (
    DATASETS,
    METRICS,
    EXPR_PLACEHOLDER
)

def get_random_string():
    return ''.join(random.choices(string.ascii_uppercase, k=random.randint(1,10))).lower()

@DATASETS.register_module()
class ImageNet1kDatasetTrain_2ClassICL(LCLDataset):
    def __init__(self, policy: str, only_image= False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neg_label = None
        self.neighbor_idx = None
        self.neighbor_label = None
        self.policy = policy
        self.only_image = only_image
        self.cls_map = self.get_cls_map()

        print(f'=========only_image:{only_image}==========')
    
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

    def get_samples(self, index, mode="cls_negative",flag=0):
        assert mode in ['cls_negative', 'neighbors']

        item = self.get_raw_item(index)
        samples = item['samples']
        neighbors = item['neighbors']

        if mode == "cls_negative":
            # current class image, random neighbor label
            if self.neg_label:
                label = self.neg_label
            else:
                metas = random.choice(neighbors)
                label = "like"
                self.neg_label = label
            sample = random.choice(samples)
        elif mode == "neighbors":
            if self.neighbor_idx:
                item_neighbor = self.get_raw_item(self.cls_map[self.neighbor_idx])
                samples = item_neighbor['samples']
                sample = random.choice(samples)
                label = "dislike"
            else:
                sample_weight = list(range(len(neighbors), 0, -1))  #これやとlenが5なら5,4,3,2,1みたいになる．優先順位をつけるのはデータの性質的な問題？？詳しくはtoolsを参照
                metas = random.choices(neighbors, weights=sample_weight)[0]
                self.neighbor_idx = metas[0]
                self.neighbor_label = metas[1]
                label = "dislike"
                sample = metas[2]
        
        else:
            raise NotImplementedError



                
        image = self.get_image(sample)
        return image, label

    # get policy function according to name 
    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        return func(index, shot)

    def policy_2way_weight(self, index, shot):
        random_string = None
        def _convert_answer(label, mode):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']
            answer = f'This image show "{LABEL_PLACEHOLDER}". [END EXAMPLE]'
            answer = answer.replace(LABEL_PLACEHOLDER, label)
            return answer

        # set context samples
        ret_list = []
        mix_question = '[BEGIN EXAMPLE] Which is like or dislike in this image <image>?'
        for mode in ['cls_negative', 'neighbors']:
            for _ in range(shot):
                image, label = self.get_samples(index, mode = mode)
                answer = _convert_answer(label, mode = mode)

                ret = self.get_ret(image, question = mix_question, answer = answer, conv_mode = 'hypnotized_ans_v1.0')
                ret_list.append(ret)
        random.shuffle(ret_list)

        ret_list[0]['mode'] = 'causal_v1.0'

        # set inference sample
        infer_question = self.get_template()
        mode = random.choice(["cls_negative", "neighbors"])
        image, label = self.get_samples(index, mode = mode,flag=1)
        answer = _convert_answer(label, mode = mode).replace(" [END EXAMPLE]", '')

        ret = self.get_ret(image, question = infer_question, answer = answer, conv_mode="final_v1.0")
        ret_list.append(ret)

        random_string = None
        self.neg_label = None
        self.neighbor_idx = None
        self.neighbor_label = None
        return ret_list



