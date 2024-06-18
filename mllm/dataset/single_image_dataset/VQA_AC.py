from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
)
from ..utils import MInstrDataset
import random
import json


@DATASETS.register_module()
class CustomDataset_VQAAC_Train(MInstrDataset):
    def __init__(self, *args, has_annotation=True, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.has_annotation = has_annotation
        with open(self.template_file, 'r') as f:
            print(self.template_file)
            self.custom_json = json.load(f)
        print('==========TASK VQA AC is initialized==========')
        with open('/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/VQA.json', 'r') as f:
            self.template = json.load(f)

    def convert_question(self, question,product_name):
        question = question.format(product=product_name)
        return question
    
    def ok_convert_answer(self, answer,product_name):
        answer = answer.format(product=product_name)
        return answer

    def ng_convert_answer(self, answer,product_name,defect_name):
        answer = answer.format(product=product_name, defect=defect_name)
        return answer
    
    def get_product_name(self, item):
        product_name = item['product_name'].split('+')[0]
        product_name = product_name.replace('_', ' ')
        return product_name
    
    def get_defect_name(self, item):
        defect_name = item['mode']
        return defect_name

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        image = self.get_image(image_path=img_path)
        
        self.product_name = self.get_product_name(item)
        defect_name = self.get_defect_name(item)

        question = random.choice(self.custom_json['question'])
        question = self.convert_question(question,self.product_name)
        final_question = random.choice(self.template).replace(QUESTION_PLACEHOLDER, question)

        if defect_name == 'None' or defect_name == 'none':
            answer = self.ok_convert_answer(random.choice(self.custom_json['ok_answer']),self.product_name)
        else:
            answer = self.ng_convert_answer(random.choice(self.custom_json['ng_answer']),self.product_name, defect_name)
        

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                },
            ]
        }
        return ret
