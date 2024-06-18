from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
)
from ..utils import MInstrDataset
import random
import json


@DATASETS.register_module()
class CustomDataset_VQAPG_Train(MInstrDataset):
    def __init__(self, *args, has_annotation=True, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.has_annotation = has_annotation
        print('==========TASK VQA PG is initialized==========')
        with open(self.template_file, 'r') as f:
            self.custom_json = json.load(f)
        with open('/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/VQA.json', 'r') as f:
            self.template = json.load(f)
    
    def convert_answer(self, answer,product_name):
        answer = answer.format(product=product_name)
        return answer
    
    def get_product_name(self, item):
        product_name = item['product_name'].split('+')[0]
        product_name = product_name.replace('_', ' ')
        return product_name


    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        image = self.get_image(image_path=img_path)
        product_name = self.get_product_name(item)


        question = random.choice(self.custom_json['question'])
        answer = self.convert_answer(random.choice(self.custom_json['answer']),product_name)
        final_question = random.choice(self.template).replace(QUESTION_PLACEHOLDER, question)

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
