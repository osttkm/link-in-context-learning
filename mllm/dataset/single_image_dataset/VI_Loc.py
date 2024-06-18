import json
import random
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from ..root import DATASETS, BOXES_PLACEHOLDER, IMAGE_PLACEHOLDER
from ..utils import MInstrDataset
from ..utils.flickr30k_entities_utils import (
    PHRASE_ED_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER,
)

from torch.distributed import get_rank


@DATASETS.register_module()
class CustomDataset_VILoc_Train(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))
        self.custom_json = None
        # テンプレの読み込み
        with open(self.template_file, 'r') as f:
            print(f'load template file from :{self.template_file}')
            self.custom_json = json.load(f)
        self.defect, self.position=None,None
        self.defectA, self.defectB, self.defectC = None, None, None
        self.positionA, self.positionB, self.positionC = None, None, None


        # 欠陥名マッピング用の辞書作成
        path = '/home/oshita/vlm/Link-Context-Learning/output_full_rearranged.json'
        self.mapping_data = {}
        _data = None
        with open(path, 'r') as f:
            _data = json.load(f)
        for key,item in _data.items():
            key = key.replace('\\','/')
            if item['category'] not in self.mapping_data:
                    self.mapping_data[item['category']] = []
            if item['bboxes']==[]:
                pass
            elif isinstance(item['bboxes'][0]['bbox'], list):
                if isinstance(item['bboxes'][0]['mode'][0], list):
                    for i in item['bboxes']['mode']:
                        self.mapping_data[item['category']].append(i)
                else:
                    self.mapping_data[item['category']].append(item['bboxes'][0]['mode'])
            else:
                raise Exception(f'Unknown pattern of data from {path} is loaded.')
            
        self.all_defects = []
        for key in self.mapping_data.keys():
            self.mapping_data[key] = list(set(self.mapping_data[key]))
            for d in self.mapping_data[key]:
                self.all_defects.append(d)
        self.all_defects = list(set(self.all_defects))



    def __len__(self):
        return len(self.data)
    
    def get_concatenate_string(self, item):
        subfolder_string = ''
        if len(item) == 1:
            return item[0]
        else:
            subfolder_string = ', '.join(item[:-1])
            if len(item) > 1:
                subfolder_string += ' and ' + item[-1]
            return subfolder_string
    
    def get_product_name(self, product):
        return product.replace('_', ' ')

    def get_defect(self):
        defect = self.mapping_data[self.product]
        if len(defect) == 1:
            return defect
        elif len(defect) == 0:
            # 欠陥が存在しない場合はランダムに欠陥名を返す．ただしほかに適当なデータに存在しない欠陥だと
            # 暗記してしまうと思われるので，全データに存在する欠陥名をランダムに返す．
            # 特定の製品＝欠陥なし　ってなると怖いけどそこはそもそも学習データの欠陥
            return [random.choice(self.all_defects)]
        else:
            # 1~len(defect)までの個数内で欠陥名を返す
            # ただし合計が1となり，指数関数的に減少する確率で返り値の要素数が決まる
            probabilities = np.exp(-np.arange(len(defect)))
            probabilities /= probabilities.sum()  # 合計が1になるように正規化

            # 欠陥名を返す個数を決定
            num_defects = np.random.choice(len(defect), p=probabilities)+1
            # 欠陥名をランダムに選択
            return np.random.choice(defect, size=num_defects, replace=False)


    def convert_no_defect_answer(self, answer):
        product = self.get_product_name(self.product)
        defect = self.get_concatenate_string(self.defect)
        q = answer.format(product=product, defect=defect)
        return q
    def convert_answer_type1(self, answer):
        product = self.get_product_name(self.product)
        position = self.get_concatenate_string(self.position)
        q = answer.format(defect=self.defect, position=position, product=product)
        return q
    def convert_answer_2type_11_defect_answer(self, answer):
        product = self.get_product_name(self.product)
        positionA = self.get_concatenate_string(self.positionA)
        positionB = self.get_concatenate_string(self.positionB)

        defect_list = [self.defectA, self.defectB]
        position_list = [positionA, positionB]
        combined = list(zip(defect_list, position_list))
        random.shuffle(combined)
        defect_list, position_list = zip(*combined)
        q = answer.format(defect1=defect_list[0], position1=position_list[0], defect2=defect_list[1], position2=position_list[1], product=product)
        return q
    
    def convert_answer_2type_12_defect_answer(self, answer):
        product = self.get_product_name(self.product)
        positionA = self.get_concatenate_string(self.positionA)
        positionB = self.get_concatenate_string(self.positionB)

        if len(self.defectA)==1 and len(self.defectB)>1:
            q = answer.format(defect1=self.defectA, position1=positionA, defects2=self.defectB, position2=positionB, product=product)
            return q
        elif len(self.defectB)==1 and len(self.defectA)>1:
            q = answer.format(defect1=self.defectB, position1=positionB, defects2=self.defectA, position2=positionA, product=product)
            return q
        else:
            raise ValueError

    def convert_answer_2type_22_defect_answer(self, answer):
        product = self.get_product_name(self.product)
        positionA = self.get_concatenate_string(self.positionA)
        positionB = self.get_concatenate_string(self.positionB)

        defect_list = [self.defectA, self.defectB]
        position_list = [positionA, positionB]
        combined = list(zip(defect_list, position_list))
        random.shuffle(combined)
        defect_list, position_list = zip(*combined)
        q = answer.format(defects1=defect_list[0], position1=position_list[0], defects2=defect_list[1], position2=position_list[1], product=product)
        return q
    
    def convert_answer_3type_111_defect_answer(self, answer):
        product = self.get_product_name(self.product)
        positionA = self.get_concatenate_string(self.positionA)
        positionB = self.get_concatenate_string(self.positionB)
        positionC = self.get_concatenate_string(self.positionC)
        
        defect_list = [self.defectA, self.defectB, self.defectC]
        position_list = [positionA, positionB, positionC]
        combined = list(zip(defect_list, position_list))
        random.shuffle(combined)
        defect_list, position_list = zip(*combined)
        q = answer.format(defect1=defect_list[0], position1=position_list[0], defect2=defect_list[1], position2=position_list[1], defect3=defect_list[2], position3=position_list[2], product=product)
        return q
    
    def convert_answer_3type_112_defect_answer(self, answer):
        product = self.get_product_name(self.product)
        positionA = self.get_concatenate_string(self.positionA)
        positionB = self.get_concatenate_string(self.positionB)
        positionC = self.get_concatenate_string(self.positionC)

        if len(self.defectA) == 2:
            defect_list = [self.defectB, self.defectC]
            position_list = [positionB, positionC]
            combined = list(zip(defect_list, position_list))
            random.shuffle(combined)
            defect_list, position_list = zip(*combined)
            q = answer.format(defect1= defect_list[0], position1=position_list[0], defects2=self.defectA, position2=positionA, defect3=defect_list[1], position3=position_list[1], product=product)
            return q
        elif len(self.defectB) == 2:
            defect_list = [self.defectA, self.defectC]
            position_list = [positionA, positionC]
            combined = list(zip(defect_list, position_list))
            random.shuffle(combined)
            defect_list, position_list = zip(*combined)
            q = answer.format(defect1= defect_list[0], position1=position_list[0], defects2=self.defectB, position2=positionB, defect3=defect_list[1], position3=position_list[1], product=product)
            return q
        elif len(self.defectC) == 2:
            defect_list = [self.defectA, self.defectB]
            position_list = [positionA, positionB]
            combined = list(zip(defect_list, position_list))
            random.shuffle(combined)
            defect_list, position_list = zip(*combined)
            q = answer.format(defect1= defect_list[0], position1=position_list[0], defects2=self.defectC, position2=positionC, defect3=defect_list[1], position3=position_list[1], product=product)
            return q
        else:
            raise ValueError

    def convert_answer_3type_122_defect_answer(self, answer):
        product = self.get_product_name(self.product)
        positionA = self.get_concatenate_string(self.positionA)
        positionB = self.get_concatenate_string(self.positionB)
        positionC = self.get_concatenate_string(self.positionC)
        
        defect_lists = [self.defectA, self.defectB, self.defectC]
        position_lists = [positionA, positionB, positionC]
        defects_length_one,defects_length_two_or_more = [],[]
        position_length_one,position_length_two_or_more = [],[]
        for defect_list,position_list in zip(defect_lists,position_lists):
            if len(defect_list) == 1:
                defects_length_one.append(defect_list)
                position_length_one.append(position_list)
            elif len(defect_list) >= 2:
                defects_length_two_or_more.append(defect_list)
                position_length_two_or_more.append(position_list)
        rand_int = random.randint(0,1)
        if rand_int == 0:
            q = answer.format(defects1=defects_length_two_or_more[0][0], position1=position_length_two_or_more[0][0], defects2=defects_length_one[0][0], position2=position_length_one[0][0], defect3=defects_length_two_or_more[1][0], position3=position_length_two_or_more[1][0], product=product)
            return q
        elif rand_int==1:
            q = answer.format(defects1=defects_length_two_or_more[1][0], position1=position_length_two_or_more[1][0], defects2=defects_length_one[0][0], position2=position_length_one[0][0], defect3=defects_length_two_or_more[0][0], position3=position_length_two_or_more[0][0], product=product)
            return q

    def convert_question(self, question):
        product = self.get_product_name(self.product)
        defect = self.get_concatenate_string(self.set_defect)
        q = question.format(product=product, defect=defect)
        return q


    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = f"{item['image_id']}"
        image = self.get_image(img_path)

        self.product = item['class_name']
        self.defect = item['defect_name']
        self.set_defect = list(set(item['defect_name']))
        self.location = np.array(item['location'])
        box_seq = item['box_seq']

        answer = None
        if item['box_seq'] == []:
            answer = random.choice(self.custom_json['no_defect_answer'])
            self.defect = self.get_defect()
            answer = self.convert_no_defect_answer(answer)
        elif len(item['box_seq']) == 1:
            answer = None
            if len(item['defect_name']) == 1:
                answer = random.choice(self.custom_json['1type_1_defect_answer'])
            elif len(item['defect_name']) > 1:
                answer = random.choice(self.custom_json['1type_2_defect_answer'])
            self.defect = self.defect[box_seq[0][0]]
            self.position = list(set(self.location))
            answer = self.convert_answer_type1(answer)
        elif len(item['box_seq']) == 2:
            answer = None

            self.defectA = self.defect[box_seq[0][0]]
            self.defectB = self.defect[box_seq[1][0]]
            self.positionA = list(set(self.location[box_seq[0]]))
            self.positionB = list(set(self.location[box_seq[1]]))

            if len(self.defectA)==1 and len(self.defectB)==1:
                answer = random.choice(self.custom_json['2type_11_defect_answer'])
                answer = self.convert_answer_2type_11_defect_answer(answer)
            elif len(self.defectA)>1 or len(self.defectB)>1:
                answer = random.choice(self.custom_json['2type_12_defect_answer'])
                answer = self.convert_answer_2type_12_defect_answer(answer)
            elif len(self.defectA)>1 and len(self.defectB)>1:
                answer = random.choice(self.custom_json['2type_22_defect_answer'])
                answer = self.convert_answer_2type_22_defect_answer(answer)

        elif len(item['box_seq']) == 3:
            answer = random.choice(self.custom_json['answer_3'])
            self.defectA = self.defect[box_seq[0][0]]
            self.defectB = self.defect[box_seq[1][0]]
            self.defectC = self.defect[box_seq[2][0]]
            self.positionA = list(set(self.location[box_seq[0]]))
            self.positionB = list(set(self.location[box_seq[1]]))
            self.positionC = list(set(self.location[box_seq[2]]))
            condition1 = len(self.defectA)==1
            condition2 = len(self.defectB)==1
            condition3 = len(self.defectC)==1
            if condition1+condition2+condition3 == 3:
                answer = random.choice(self.custom_json['3type_111_defect_answer'])
                answer = self.convert_answer_3type_111_defect_answer(answer)
            elif condition1+condition2+condition3 == 2:
                answer = random.choice(self.custom_json['3type_112_defect_answer'])
                answer = self.convert_answer_3type_112_defect_answer(answer)
            elif condition1+condition2+condition3 == 1:
                answer = random.choice(self.custom_json['3type_122_defect_answer'])
                answer = self.convert_answer_3type_122_defect_answer(answer)
            else:
                raise ValueError

        else:
            print(item['box_seq'])
            print('Erorr: boxes length is not 1, 2, 3')
            raise ValueError
           
        # elif len(item['boxes'][0]) == 4:
        #     caption = self.custom_json['answer_4']
     
        answer = answer.replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
        question = random.choice(self.custom_json['question'])
        question = self.convert_question(question)
        rank = get_rank()
        if rank == 0:
            print(f'question: {question}')
            print(f'answer: {answer}')
        

        ret = {
            'image': image,
            'target': {'boxes': item['boxes']},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                    'boxes_seq': item['box_seq'],
                }
            ]
        }
        return ret
