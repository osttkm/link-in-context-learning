import torch.distributed as dist

import os
import json
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

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def process_data(data):
    path_dict = {}
    for d in data:
        class_name = d['class_name']
        if class_name not in path_dict:
            path_dict[class_name] = {}
            path_dict[class_name]['class_id'] = d['class_id']
            path_dict[class_name]['class_name'] = d['class_name']
            path_dict[class_name]['image_path'] = []
            path_dict[class_name]['caption'] = []
        samples = d['samples']
        neighbors = d['neighbors'] 
        # import pdb;pdb.set_trace()
        for sample in samples:
            path_dict[class_name]['image_path'].append(sample)
        for neighbor in neighbors:
            neg_class_id = neighbor[0]
            neg_class_name = neighbor[1]
            if neg_class_name not in path_dict:
                path_dict[neg_class_name] = {}
                path_dict[neg_class_name]['class_id'] = neg_class_id
                path_dict[neg_class_name]['class_name'] = neg_class_name
                path_dict[neg_class_name]['image_path'] = []
                path_dict[neg_class_name]['caption'] = []
            path_dict[neg_class_name]['image_path'].append(neighbor[-1])
    for key in path_dict.keys():
        if len(path_dict[key]) != len(set(path_dict[key])):
            print('重複あり')
            path_dict[key] = list(set(path_dict[key]))
    return path_dict

@DATASETS.register_module()
class ImageNet1kDatasetTrain(LCLDataset):
    def __init__(self, policy: str, task_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neg_label = None
        self.neighbor_idx = None
        self.neighbor_label = None
        self.policy = policy
        self.task_type = task_type
        self.cls_map = self.get_cls_map()
        self.path2cap_data = read_jsonl("imagenet_path2caption.jsonl")
        self.path2cap_data = process_data(self.path2cap_data)

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

    def get_samples(self, index, mode="cls_negative"):
        assert mode in ['cls_negative', 'neighbors']

        item = self.get_raw_item(index)
        class_name = item['class_name'].lower()
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
                
        image = self.get_image(sample)
        caption = self.path2cap_data[class_name]["caption"][self.path2cap_data[class_name]["image_path"]==sample]
        return image, label, caption

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

            answer = f'{LABEL_PLACEHOLDER}'
            if mode == "cls_negative":
                # set random string as answer
                if not random_string:
                    random_string = get_random_string()
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)
            elif mode == "neighbors":
                answer = answer.replace(LABEL_PLACEHOLDER, label)

            return answer
        ret_list = []
        mix_question = '<image>'
        for mode in ['cls_negative', 'neighbors']:
            for _ in range(shot):
                image, label, caption = self.get_samples(index, mode = mode)
                # ret = self.get_ret(image, question = mix_question, answer = caption, conv_mode = 'hypnotized_ans_v1.0')
                if self.task_type=="captioning":
                    ret = self.get_ret(image, question = mix_question, answer = caption, conv_mode = 'Image_only_mode')
                elif self.task_type=="classification":
                    ret = self.get_ret(image, question = mix_question, answer = label, conv_mode = 'Image_only_mode')
                elif self.task_type=="symbol":
                    ret = self.get_ret(image, question = mix_question, answer = _convert_answer(label, mode), conv_mode = 'Image_only_mode')
                ret_list.append(ret)
        random.shuffle(ret_list)
        ret_list[0]['mode'] = 'Image_only_mode'

        # set inference sample
        # infer_question = self.get_template()
        mode = random.choice(["cls_negative", "neighbors"])
        image, label, caption = self.get_samples(index, mode = mode)
        # ret = self.get_ret(image, question = infer_question, answer = caption, conv_mode="Image_only_mode")
        if self.task_type=="captioning":
            ret = self.get_ret(image, question = mix_question, answer = caption, conv_mode = 'Image_only_mode')
        elif self.task_type=="classification":
            ret = self.get_ret(image, question = mix_question, answer = label, conv_mode = 'Image_only_mode')
        elif self.task_type=="symbol":
            ret = self.get_ret(image, question = mix_question, answer = _convert_answer(label, mode), conv_mode = 'Image_only_mode')
        ret_list.append(ret)
        # rank = dist.get_rank()
        # if rank==0:
        #     print(f'three images and answer:\n{ret_list[0]}\n{ret_list[1]}\n{ret_list[2]}\n')
        random_string = None
        self.neg_label = None
        self.neighbor_idx = None
        self.neighbor_label = None
        return ret_list

    def policy_2way_random_shot(self, index, shot):
        weight = [math.exp((i+1)/2) for i in range(shot)]
        shot_list = [i+1 for i in range(shot)]
        shot = random.choices(shot_list, weight)[0]

        ret_list = self.policy_2way_weight(index, shot)
        return ret_list

    def policy_2way_ref(self, index, shot):
        random_string = None
        random_name_list = []
        def _convert_answer(mode, final=False, order_list=None):
            nonlocal random_string
            nonlocal random_name_list
            assert mode in ['cls_negative', 'neighbors']
            if final:
                answer = f'{order_list}.' + '_' + random_string
            else:
                if not random_string:
                    random_string = get_random_string()
                
                name = get_random_string() +'_' + random_string
                random_name_list.append(name)
                answer = f'The reference name of this image is "{name}". [END EXAMPLE]'

            return answer
        
        ret_list = []
        mix_question = '[BEGIN EXAMPLE] Tell me something about this image <image>.'
        infer_question = 'Which images in the previous example are in the same category as this image <image>? (Provide the answer in order as list)'

        shot = random.randint(3, shot)
        ori_list = []
        for mode in ['cls_negative', 'neighbors']:
            for _ in range(shot):
                image, label, caption = self.get_samples(index, mode = mode)
                answer = _convert_answer(mode = mode)
                ret = self.get_ret(image, question = mix_question, answer = answer, conv_mode="hypnotized_ans_v1.0")
                ret_list.append(ret)
                ori_list.append(0 if mode == 'cls_negative' else 1)

        # combine and shuffle
        tmp = list(zip(ret_list, ori_list, random_name_list))
        random.shuffle(tmp)
        ret_list, ori_list, name_list = list(zip(*tmp))
        ret_list = list(ret_list)
        ori_list = list(ori_list)
        name_list = list(name_list)
        ret_list[0]['mode'] = 'causal_v1.0'

        name_list = np.array(name_list)
        p_idx = np.array(ori_list) == 0
        idx_p = list(name_list[p_idx])

        n_idx = np.array(ori_list) == 1
        idx_n = list(name_list[n_idx])

        # query sample
        mode = random.choice(["cls_negative", "neighbors"])
        order_list = str(idx_n) if mode == "cls_negative" else str(idx_p)

        image, label, caption = self.get_samples(index, mode = mode)
        answer = _convert_answer(mode = mode, final=True, order_list=order_list)
        ret = self.get_ret(image, question = infer_question, answer = answer, conv_mode="final_v1.0")
        ret_list.append(ret)
        random_string = None
        self.neg_label = None
        self.neighbor_idx = None
        self.neighbor_label = None
        return ret_list

    def policy_2way_ref_or_weight(self, index, shot):
        if random.randint(0, 1):
            ret_list = self.policy_2way_weight(index, shot)
            # # fit question to ref policy
            # mix_question = '[BEGIN EXAMPLE] Tell me something about this image <image>.'
            # for i in range(len(ret_list)):
            #     if i != len(ret_list) -1:
            #         ret_list[i]['conversations'][0]['value'] = mix_question
        else:
            ret_list = self.policy_2way_ref(index, shot)

        return ret_list

    def policy_jigsaw(self, index, shot):
        ret_list = []
        mix_question = '[BEGIN EXAMPLE] This image <image> is a puzzle piece.'
        infer_question = 'Using the puzzle pieces provided above to piece together this image <image>, and provide the coordinates of each piece on the image coordinate system in order.'

        tiles_list = [4,6,9,12,16]
        combination_dict = {4:[[2,2]],6:[[2,3],[3,2]],9:[[3,3]],12:[[3,4],[4,3]],16:[[4,4]]}
        N = random.choice(tiles_list)
        comb = combination_dict[N]
        comb = random.choice(comb)
        #shot = random.randint(1, shot)
        image, label, caption = self.get_samples(index, mode = 'cls_negative')
        #whole_image = deepcopy(image)

        tiles = [None]*N
        width, height = image.size
        canvas = np.zeros((height,width,3))
        split_w,split_h = comb
        w_element = int(width/split_w)
        h_element = int(height/split_h)
        order_list = []

        meta_order_list = []
        cnt = 0
        for i in range(split_w):
            for j in range(split_h):
                tmp = image.crop([i*w_element,j*h_element,(i+1)*w_element,(j+1)*h_element])
                tiles[cnt] = tmp
                cnt += 1
                order_list.append([i,j])
                meta_order_list.append([i,j])

        tmp = list(zip(tiles,order_list))
        random.shuffle(tmp)
        tile,order_list=list(zip(*tmp))

        sub_cnt = 0
        for j in range(split_h):
            for i in range(split_w):
                cv_img = cv.cvtColor(np.array(tile[sub_cnt]),cv.COLOR_RGB2BGR)
                canvas[j*h_element:(j+1)*h_element,i*w_element:(i+1)*w_element] = cv_img
                sub_cnt += 1

        tmp = list(zip(tile,meta_order_list))
        random.shuffle(tmp)
        tile,meta_order_list=list(zip(*tmp))
        
        for i in range(len(tile)):
            answer = '[END EXAMPLE]'
            question = mix_question
            image = tile[i]
            if i == 0:
                ret = self.get_ret(image, question = question, answer = answer, conv_mode="causal_v1.0")    
            else:
                ret = self.get_ret(image, question = question, answer = answer, conv_mode="hypnotized_ans_v1.0")
            ret_list.append(ret)
        
        answer = 'The order is: '+str(list(meta_order_list))+'.'
        ret = self.get_ret(image, question = infer_question, answer = answer, conv_mode="final_v1.0")
        ret_list.append(ret)

        self.neg_label = None
        self.neighbor_idx = None
        self.neighbor_label = None
        return ret_list
