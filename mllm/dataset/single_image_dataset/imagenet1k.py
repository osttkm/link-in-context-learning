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
class ImageNet1kDatasetTrain(LCLDataset):
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
                label = metas[1].lower()
                self.neg_label = label
            sample = random.choice(samples)
        elif mode == "neighbors":
            if self.neighbor_idx:
                item_neighbor = self.get_raw_item(self.cls_map[self.neighbor_idx])
                samples = item_neighbor['samples']
                sample = random.choice(samples)
                label = self.neighbor_label.lower()
            else:
                sample_weight = list(range(len(neighbors), 0, -1))  #これやとlenが5なら5,4,3,2,1みたいになる．優先順位をつけるのはデータの性質的な問題？？詳しくはtoolsを参照
                metas = random.choices(neighbors, weights=sample_weight)[0]
                self.neighbor_idx = metas[0]
                self.neighbor_label = metas[1]
                label = metas[1].lower()
                sample = metas[2]
        
        else:
            raise NotImplementedError

        # rank = dist.get_rank() 
        # if rank==0 :
        #     print(f'\nimage path: {sample}  label: {label}  mode: {mode}')

                
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
            # if self.only_image:
                # answer = f'{LABEL_PLACEHOLDER}'
            # else:
            answer = f'there is "{LABEL_PLACEHOLDER}" in the image. [END EXAMPLE]'
            if mode == "cls_negative":
                # set random string as answer
                if not random_string:
                    random_string = get_random_string()
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)
            elif mode == "neighbors":
                answer = answer.replace(LABEL_PLACEHOLDER, label)

            return answer

        # set context samples
        ret_list = []
        # mix_question = '[BEGIN EXAMPLE] What is in the image <image>?'
        mix_question = '<image> [BEGIN EXAMPLE] What is in the image?'
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



@DATASETS.register_module()
class ImageNetTest100Eval(LCLDataset):
    def __init__(self, policy, sample_per_class = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.sample_per_class = sample_per_class
        self.data_map = self._rearrange()

    def __len__(self):
        return len(self.data_map)

    def _rearrange(self):
        # Map dataloader index to self.data, according to class_idx and sample_idx
        data_map = []
        for cls_idx, item in enumerate(self.data):
            test_samples = item['test_samples']
            for sample_idx, sample in enumerate(test_samples):
                # sample_per_class = 0: all samples evaluation
                if sample_idx == self.sample_per_class and \
                    self.sample_per_class > 0:
                    break
                data_map.append([cls_idx, sample_idx])
        return data_map        

    def get_samples(self, index, shot):
        cls_idx, sample_idx = self.data_map[index]
        item = self.get_raw_item(cls_idx)
        class_id = item["class_id"]
        class_name = item["class_name"].lower()
        context_samples = item["context_samples"]
        test_samples = item['test_samples']

        test_img = self.get_image(test_samples[sample_idx])
        context_imgs = []
        for i in range(shot):
            context_imgs.append(self.get_image(context_samples[i]))
        return class_name, context_imgs, test_img


@DATASETS.register_module()
class ImageNetTest100Eval2Way(ImageNetTest100Eval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.data)%2 == 0

    def get_samples(self, index, shot):
        cls_idx, sample_idx = self.data_map[index]
        item = self.get_raw_item(cls_idx)
        class_id = item["class_id"]
        pos_cls_name = item["class_name"].lower()
        test_samples = item['test_samples']

        # construct positive and negtive pairs
        if cls_idx % 2 == 0:
            neg_cls_idx = cls_idx + 1
        else:
            neg_cls_idx = cls_idx - 1
        neg_item = self.get_raw_item(neg_cls_idx)
        pos_samples = item["context_samples"]
        neg_samples = neg_item["context_samples"]
        neg_cls_name = neg_item["class_name"].lower()
        
        pos_imgs, neg_imgs = [], []
        for i in range(shot):
            pos_imgs.append(self.get_image(pos_samples[i]))
            neg_imgs.append(self.get_image(neg_samples[i]))            
        # inference sample (positive class)
        infer_img = self.get_image(test_samples[sample_idx])

        sample_meta = dict(
            pos_cls_name = pos_cls_name,
            neg_cls_name = neg_cls_name,
            pos_imgs = pos_imgs,
            neg_imgs = neg_imgs,
            infer_img = infer_img
            )
        return sample_meta

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        
        sample_meta = self.get_samples(index, shot)
        QnA = func(sample_meta["pos_cls_name"],sample_meta["neg_cls_name"])
        ret_list = []

        # context sample: pos A image(text: there is A) + neg B image(text: there is B) + infer A image(label: there is A)
        for img in sample_meta["pos_imgs"]:
            ret_list.append(self.get_ret(img, question=QnA["pos_question"], answer=QnA["pos_answer"]))
        
        for img in sample_meta["neg_imgs"]:
            ret_list.append(self.get_ret(img, question=QnA["neg_question"], answer=QnA["neg_answer"]))
        random.shuffle(ret_list)

        for i in range(len(ret_list)):
            if i == 0:
                conv_mode = 'causal_v1.0'
            else:
                conv_mode = 'hypnotized_ans_v1.0'
            ret_list[i]['mode'] = conv_mode

        # inference
        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"], conv_mode="final_v1.0")) 
        return ret_list

    def policy_2way(self, cls_name_pos, cls_name_neg):
        pos_question = '[BEGIN EXAMPLE] What is in the image <image>?'
        neg_question = pos_question
        infer_question = f'Based on the previous examples, what is in the image <image>?'

        answer = f'there is "{LABEL_PLACEHOLDER}" in the image. [END EXAMPLE]'
        pos_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_pos)
        neg_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_neg)
        infer_answer = pos_answer.replace(" [END EXAMPLE]", "")

        return dict(
            pos_question = pos_question, 
            neg_question = neg_question,
            infer_question = infer_question,
            pos_answer = pos_answer,
            neg_answer = neg_answer, 
            infer_answer = infer_answer
        )

@METRICS.register_module()
class ImageNetTest100Metrics(LCLComputeMetrics):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(filename, *args, **kwargs)
        self.gt_pairs = self.gt_pairs_test100()
    
    def gt_pairs_test100(self):
        target_pairs = dict()
        cls_metas = []
        with jsonlines.open(self.filename) as reader:
            for metas in reader:
                cls_metas.append(metas)

        for cls_idx, _ in enumerate(cls_metas):
            if cls_idx % 2 == 0:
                neg_cls_idx = cls_idx + 1
            else:
                neg_cls_idx = cls_idx - 1
            pos_item = cls_metas[cls_idx]
            pos_cls_name = pos_item["class_name"].lower()
            neg_item = cls_metas[neg_cls_idx]
            neg_cls_name = neg_item["class_name"].lower()
            target_pairs[pos_cls_name] = neg_cls_name
        return target_pairs
    
    def get_neg_pair(self, index, target):
        return self.gt_pairs[target]