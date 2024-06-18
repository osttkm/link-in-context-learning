import jsonlines
import random
import logging
import sys
from ..utils import MInstrDataset
from .. import BaseComputeMetrics
from typing import Dict, Any, Sequence
from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    METRICS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@DATASETS.register_module()
class ImageNet1kClassifyDatasetTrain(MInstrDataset):
    def __init__(self, *args, has_annotation=True, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        samples = item['samples']
        sample = random.choice(samples)
        
        image = self.get_image(sample)
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, "What is in the image?")
        label = item['class_name'].lower()
        answer = f"There is {label} in the image."


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
    

@DATASETS.register_module()
class ImageNetTest_Classify_Eval(MInstrDataset):
    def __init__(self, *args, has_annotation=True, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        samples = item['samples']
        sample = random.choice(samples)
        
        image = self.get_image(sample)
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, "What is in the image?")
        label = item['class_name'].lower()
        answer = f"There is {label} in the image."


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
    

@METRICS.register_module()
class ImageNet_Classify_ComputeMetrics(BaseComputeMetrics):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename

    def extract_target(self, string: str):
        try:
            found = string.split("gpt:")[-1].split("</s>")[0]
            found = found.replace("The answer is", "")
            found = found.replace('There is', '').replace('in the image', '')
            found = found.replace("\"", "").replace("\'", "").replace(".", "").strip().lower()
            return found
        except (IndexError, AttributeError):
            return None

    def extract_pred(self, string: str):
        try:
            found = string.replace("The answer is", "")
            found = found.replace('There is', '').replace('in the image', '')
            found = found.replace("\"", "").replace("\'", "").replace(".", "").strip().lower()
            return found
        except (IndexError, AttributeError):
            return None

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        correct = 0
        failed = 0
        target_failed = 0
        for idx, (pred, target) in enumerate(zip(preds, targets)):
            extract_pred = self.extract_pred(pred)
            extract_target = self.extract_target(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans from target. maybe the response string is truncated: {target}.")
                continue
            if extract_pred is None:
                failed += 1
            
            pos_target = extract_target

            if pos_target in pred:
                correct += 1
        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
        }