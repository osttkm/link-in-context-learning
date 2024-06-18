import copy
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
from functools import partial
from typing import Dict, Any, Optional
from mllm.conversation import get_conv_template
from mllm.dataset.root import TRANSFORMS, FUNCTIONS
from mllm.dataset.single_image_convsation import SingleImageConvDatasetMixin

def prepare_demo_dataset(
        json_path, 
        model_args,
        preprocessor: Dict[str, Any],
):
    conv_args = model_args.conv_args
    tokenize_kwargs = conv_args.get('tokenize_kwargs', {})
    conv_template_ = conv_args.get('conv_template', 'vicuna_v1.1')
    if isinstance(conv_template_, list):
        conv_template = {item: partial(get_conv_template, name=item) for item in conv_template_}
    else:
        conv_template = partial(get_conv_template, name=conv_template_)
    transforms = conv_args.get('transforms', None)
    if transforms is not None:
        transforms = TRANSFORMS.build(transforms)
    # process func
    process_func = {}
    for k, v in model_args.process_func_args.items():
        process_func[k] = FUNCTIONS.build(cfg=v)

    ds = SingleImageInteractive(
        json_path=json_path,
        preprocessor=preprocessor,
        process_func=process_func,
        tokenize_kwargs=tokenize_kwargs,
        conv_template=conv_template,
        training_args=None,
        transforms=transforms,
        use_icl=True,
        mode='test',
    )
    return ds

class SingleImageInteractive(SingleImageConvDatasetMixin):

    def __init__(self, json_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_meta = None
        
        with open(json_path) as f:
            json_data = json.load(f)
        # self.ac_query_question_lines = np.array(json_data['query_question_lines'])
        # self.ac_context_question_lines = np.array(json_data['context_question_lines'])
        # self.ac_no_responses_array = np.array(json_data['no_responses_array'])
        # self.ac_yes_responses_array = np.array(json_data['yes_responses_array'])
        if "LCL" in json_path:
            self.vi_query_question_lines = np.array(json_data['query_VI_instructions'])
        else:
            self.vi_query_question_lines = np.array(json_data['VI_instructions'])
        self.vi_context_question_lines = np.array(json_data['VI_instructions'])
        self.vi_no_responses_array = np.array(json_data['answers_ok'])
        self.vi_yes_responses_array = np.array(json_data['answers_ng'])

    def get_defect_name(self,name):
        _class,defect = name.split('+')[0],name.split('+')[1]
        name = None
        if _class == "bottle":
            if defect == "broken_large" or defect == "broken_small":
                name = "broken"
            elif defect == "contamination":
                name = "contamination"

        elif _class == "cable":
            if defect == "bent_wire":
                name="bent"
            elif defect == "cable_swap":
                name = "swap"
            elif defect == "cut_inner_insulation" or defect == "cut_outer_insulation":
                name = "crack"
            elif defect == "missing_wire" or defect == "missing_cable":
                name = "missing"
            elif defect == "poke_insulation":
                name = "hole"

        elif _class == "capsule":
            if defect == "crack":
                name = "crack"
            elif defect == "faulty_imprint":
                name = "misprint"
            elif defect == "squeeze":
                name = "hole"
            elif defect == 'poke':
                name = "poke"
            elif defect == 'scratch':
                name = "misshapen"

        elif _class == "carpet":
            if defect == "color":
                name = "stain"
            elif defect == "hole":
                name = "hole"
            elif defect == 'metal_contamination' or defect == "thread":
                name = "contamination"
            elif defect == "cut":
                name = "cut"

        elif _class == "grid":
            if defect == 'bent':
                name = "bent"
            elif defect == 'broken':
                name = "broken"
            elif defect == 'metal_contamination' or defect == 'thread' or defect == 'glue':
                name = "contamination"

        elif _class == "hazelnut":
            if defect == 'crack':
                name = "crack"
            elif defect == 'hole':
                name = "hole"
            elif defect == 'cut':
                name = "scratch"
            elif defect == 'print':
                name = "misprint"

        elif _class == "leather":
            if defect == 'color' or defect == 'glue':
                name = "stain"
            elif defect == 'cut':
                name = "scratch"
            elif defect == 'fold':
                name = "wrinkle"
            elif defect == 'poke':
                name = "hole"

        elif _class == "metal_nut":
            if defect == 'bent':
                name = "bent"
            elif defect == 'flip':
                name = "flip"
            elif defect == 'scratch':
                name = "scratch"
            elif defect == 'color':
                name = "stain"

        elif _class == "pill":
            if defect == 'color':
                name = "stain"
            elif defect == 'crack':
                name = "crack"
            elif defect == 'faulty_imprint':
                name = "misprint"
            elif defect == 'scratch':
                name = "scratch"
            elif defect == 'pill_type':
                name = "stain"
            elif defect == 'contamination':
                name = "contamination"

        elif _class == "screw":
            if defect == 'manipulated_front':
                name = "strip"
            elif defect == 'scratch_head' or defect == 'scratch_neck' or defect == 'thread_side' or defect == 'thread_top':
                name = "chip"

        elif _class == "tile":
            if defect == 'crack':
                name = "crack"
            elif defect == 'glue_strip':
                name = "contamination"
            elif defect == 'gray_stroke' or  defect == 'oil' or defect == "rough":
                name = "stain"

        elif _class == 'toothbrush':
            if defect == 'defective':
                name = "broken"

        elif _class == "transistor":
            if defect == 'bent_lead':
                name = "bent"
            elif defect == 'cut_lead':
                name = "cut"
            elif defect == 'damaged_case':
                name = "broken"
            elif defect == "misplaced":
                name = "misalignment"

        elif _class == "wood":
            if defect == 'color' or defect == 'liquid':
                name = "stain"
            elif defect == 'hole':
                name = "hole"
            elif defect == "scratch":
                name = "scratch"

        elif _class == "zipper":
            if defect == 'broken_teeth': 
                name = "broken"
            elif defect == 'fabric_border':
                name = "tear"
            elif defect == 'fabric_interior' or defect == 'rough' :
                name = "frayed"
            elif defect == 'squeezed_teeth'or defect == 'split_teeth':
                name = "misshapen"
        return name
    
    def get_subfolder_strings(self, class_defects:list):
        # class_defectに"combined"が含まれる場合は、"combined"を削除する
        if "combined" in class_defects:
            class_defects.remove("combined")
        converted_defects = [self.get_defect_name(f"{self.speacis}+{defect}") for defect in class_defects]
        # converted_defectsをユニークな配列に変換する
        converted_defects = list(set(converted_defects))
        
        subfolder_string = ''
        if len(converted_defects) == 1:
            return converted_defects[0]
        else:
            subfolder_string = ', '.join(converted_defects[:-1])
            if len(converted_defects) > 1:
                subfolder_string += ' and ' + converted_defects[-1]
            return subfolder_string

    def update_data(self, data,defect_name=None,mode=1,eval_mode="AC",speacis=None):
        self.data_meta = data
        self.defect_name = defect_name
        self.mode = mode
        self.eval_mode = eval_mode
        if self.eval_mode=='VI':
            self.speacis = speacis
            path = Path(f'/dataset/mvtec/{speacis}/test')
            self.class_defect = [p.name for p in path.iterdir() if p.is_dir()]
            self.class_defect.remove('good')
        # elif self.eval_mode=='AC+VI':
        #     self.speacis = speacis
        #     path = Path(f'/dataset/mvtec/{speacis}/test')
        #     self.class_defect = [p.name for p in path.iterdir() if p.is_dir()]
        #     self.class_defect.remove('good')



    def check_data(self):
        print(self.data_meta)

    def calc_index(self,q,mode):
        # answerから質問文を特定する関数
        idx = None
        if mode == "pos":
            if self.eval_mode=="AC":
                if len(np.where(self.ac_yes_responses_array == q)[0]) != 1:
                    i = random.randint(0,len(np.where(self.ac_yes_responses_array == q)[0])-1)
                    idx = np.where(self.ac_yes_responses_array == q)[0][i]
                elif len(np.where(self.ac_yes_responses_array == q)[0]) == 1:
                    idx = np.where(self.ac_yes_responses_array == q)[0][0]
            elif self.eval_mode=="VI":
                if len(np.where(self.vi_yes_responses_array == q)[0]) != 1:
                    i = random.randint(0,len(np.where(self.vi_yes_responses_array == q)[0])-1)
                    idx = np.where(self.vi_yes_responses_array == q)[0][i]
                elif len(np.where(self.vi_yes_responses_array == q)[0]) == 1:
                    idx = np.where(self.vi_yes_responses_array == q)[0][0]

            
        elif mode=="neg":
            if self.eval_mode=="AC":
                if len(np.where(self.ac_no_responses_array == q)[0]) != 1:
                    i = random.randint(0,len(np.where(self.ac_no_responses_array == q)[0])-1)
                    idx = np.where(self.ac_no_responses_array == q)[0][i]
                elif len(np.where(self.ac_no_responses_array == q)[0]) == 1:
                        idx = np.where(self.ac_no_responses_array == q)[0][0]
            elif self.eval_mode=="VI":
                if len(np.where(self.vi_no_responses_array == q)[0]) != 1:
                    i = random.randint(0,len(np.where(self.vi_no_responses_array == q)[0])-1)
                    idx = np.where(self.vi_no_responses_array == q)[0][i]
                elif len(np.where(self.vi_no_responses_array == q)[0]) == 1:
                        idx = np.where(self.vi_no_responses_array == q)[0][0]
            
        if idx == None:
            raise Exception("There is no index.")

        return idx
        # self.context_question_lines


    def clear_data(self):
        self.data_meta = None
    
    def get_ret(self, image, question, answer, conv_mode=None):
        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f"{answer}",
                },
            ]
        }
        if conv_mode is not None:
            ret['mode'] = conv_mode
        return ret

    def get_raw_icl_item(self, index, shot):
        assert self.data_meta is not None
        result = []
        
        if self.data_meta['mode'] == 'lcl':
            if self.eval_mode == "AC":
                for img in self.data_meta['pos_img']:
                    if self.mode == 1:
                        # 異常データに対する回答
                        # コンテキストの質問は回答から逆算
                        answer = self.data_meta['pos_a'][0] + ' [END EXAMPLE]'
                        idx = self.calc_index(self.data_meta['pos_a'],"pos") 
                        question = '[BEGIN EXAMPLE] '+self.ac_context_question_lines[idx][0]
                        answer = answer.format(defect=self.defect_name)
                        # print(f'異常pos1  Q:{question}   A:{answer}')
                    elif self.mode==2:
                        # 正常データに対する回答
                        answer = self.data_meta['pos_a'][0] + ' [END EXAMPLE]'
                        idx = self.calc_index(self.data_meta['pos_a'],"neg") 
                        question = '[BEGIN EXAMPLE] '+self.ac_context_question_lines[idx][0]
                        # print(f'正常pos2  Q:{question}   A:{answer}')
                    else:
                        raise NotImplementedError
                    result.append(self.get_ret(image=img, question=question, answer=answer))
                for img in self.data_meta['neg_img']:
                    if self.mode == 1:
                        # 正常データに対する回答
                        answer = self.data_meta['neg_a'][0] + ' [END EXAMPLE]'
                        idx = self.calc_index(self.data_meta['neg_a'],"neg") 
                        question = '[BEGIN EXAMPLE] '+self.ac_context_question_lines[idx][0]
                        # print(f'正常neg1  Q:{question}   A:{answer}')
                    elif self.mode==2:
                        # 異常データに対する回答
                        answer = self.data_meta['neg_a'][0] + ' [END EXAMPLE]'
                        idx = self.calc_index(self.data_meta['neg_a'],"pos") 
                        question = '[BEGIN EXAMPLE] '+self.ac_context_question_lines[idx][0]
                        answer = answer.format(defect = self.defect_name)
                        # print(f'異常neg2  Q:{question}   A:{answer}')
                    else:
                        raise NotImplementedError
                    result.append(self.get_ret(image=img, question=question, answer=answer))

            elif self.eval_mode == 'VI':
                for img in self.data_meta['pos_img']:
                    if self.mode == 1:
                        # 異常データに対する回答
                        # コンテキストの質問は回答から逆算
                        answer = self.data_meta['pos_a'][0] + ' [END EXAMPLE]'
                        idx = self.calc_index(self.data_meta['pos_a'],"pos") 
                        question = '[BEGIN EXAMPLE] '+self.vi_context_question_lines[idx][0]
                        question = question.format(product=self.speacis, subfolder_string = self.get_subfolder_strings(self.class_defect))
                        answer = answer.format(product=self.speacis, defect=self.defect_name)
                        # print(f'異常pos1  Q:{question}   \nA:{answer}')
                    elif self.mode==2:
                        # 正常データに対する回答
                        answer = self.data_meta['pos_a'][0] + ' [END EXAMPLE]'
                        idx = self.calc_index(self.data_meta['pos_a'],"neg") 
                        question = '[BEGIN EXAMPLE] '+self.vi_context_question_lines[idx][0]
                        question = question.format(product=self.speacis, subfolder_string = self.get_subfolder_strings(self.class_defect))
                        answer = answer.format(product=self.speacis, subfolder_string = self.get_subfolder_strings(self.class_defect))
                        # print(f'正常pos2  Q:{question}   \nA:{answer}')
                    else:
                        raise NotImplementedError
                    result.append(self.get_ret(image=img, question=question, answer=answer))
                for img in self.data_meta['neg_img']:
                    if self.mode == 1:
                        # 正常データに対する回答
                        answer = self.data_meta['neg_a'][0] + ' [END EXAMPLE]'
                        idx = self.calc_index(self.data_meta['neg_a'],"neg") 
                        question = '[BEGIN EXAMPLE] '+self.vi_context_question_lines[idx][0]
                        question = question.format(product=self.speacis, subfolder_string = self.get_subfolder_strings(self.class_defect))
                        answer = answer.format(product=self.speacis, subfolder_string = self.get_subfolder_strings(self.class_defect))
                        # print(f'正常neg1  Q:{question}   \nA:{answer}')
                    elif self.mode==2:
                        # 異常データに対する回答
                        answer = self.data_meta['neg_a'][0] + ' [END EXAMPLE]'
                        idx = self.calc_index(self.data_meta['neg_a'],"pos") 
                        question = '[BEGIN EXAMPLE] '+self.vi_context_question_lines[idx][0]
                        question = question.format(product=self.speacis, subfolder_string = self.get_subfolder_strings(self.class_defect))
                        answer = answer.format(product=self.speacis, defect=self.defect_name)
                        # print(f'異常neg2  Q:{question}   \nA:{answer}')
                    else:
                        raise NotImplementedError
                    result.append(self.get_ret(image=img, question=question, answer=answer))

            # remove system infomation in the middle of prompt
            for i in range(len(result)):
                if i == 0:
                    support_mode = 'causal_v1.0'
                else:
                    support_mode = 'hypnotized_ans_v1.0'
                result[i]['mode'] = support_mode

            infer_mode = 'final_v1.0'
        elif self.data_meta['mode'] == 'vqa':
            infer_mode = 'vicuna_v1.1'
        else:
            raise NotImplementedError

        infer_img = self.data_meta['infer_img'][0]
        infer_question = self.data_meta['infer_q'][0]
        if self.eval_mode == 'AC+VI' or self.eval_mode == 'VI':
            infer_question = infer_question.format(product = self.speacis, subfolder_string = self.get_subfolder_strings(self.class_defect))
        
        # print(f'infer_question:{infer_question}')
        result.append(self.get_ret(image=infer_img, question=infer_question, answer='', conv_mode=infer_mode))
        return result

    def __getitem__(self, index, debug_mode=False) -> Dict[str, Any]:
        item = super().__getitem__(index, debug_mode)
        update_keys = ['image', 'input_ids', 'attention_mask', 'labels']

        ret = dict()
        for k, v in item.items():
            if k == 'image':
                k = 'images'
            ret[k] = v.unsqueeze(0).cuda()
        return ret


    def __len__(self):
        return 1
