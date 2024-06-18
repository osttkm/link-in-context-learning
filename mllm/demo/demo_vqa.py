import os
import sys
import json
import random
import logging
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np
import gradio as gr
from gradio.themes.utils.sizes import Size
from PIL import Image
from mmengine import Config
import transformers
from transformers import BitsAndBytesConfig

SLURM_ENV = {k: v for k, v in os.environ.items() if 'SLURM' in k}
sys.path.append(str(Path(__file__).parent.parent.parent))

from mllm.models.builder.build_llava import load_pretrained_llava
from mllm.dataset.process_function import PlainBoxFormatter
from demo_dataset import prepare_demo_dataset

# log_level = logging.DEBUG
# transformers.logging.set_verbosity(log_level)
# transformers.logging.enable_default_handler()
# transformers.logging.enable_explicit_format()

TEMP_FILE_DIR = Path(__file__).parent / 'temp'
TEMP_FILE_DIR.mkdir(parents=True, exist_ok=True)
#########################################
# mllm model init
#########################################

#region paser
parser = argparse.ArgumentParser("LCL Web Demo")
parser.add_argument('--base_model', default='llama', choices=['llama'])
parser.add_argument('--model_path', default=r'/home/oshita/vlm/Link-Context-Learning/model_result/LCL_VI')
parser.add_argument('--json_path', default=r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_VQA_template/VQA_AC_10.json')
parser.add_argument('--server_name', default=SLURM_ENV.get('SLURM_JOB_NODELIST', None))
parser.add_argument('--server_port', type=int, default=20489)
parser.add_argument('--remove_model', action='store_true')
parser.add_argument('--load_in_8bit', action='store_true')
parser.add_argument('--load_in_4bit', action='store_true')
parser.add_argument('--q_type', type=int, default=0)

args = parser.parse_args()
args.cluster_mode = bool(SLURM_ENV)
if args.load_in_4bit and args.load_in_8bit:
    warnings.warn("use `load_in_4bit` and `load_in_8bit` at the same time. ignore `load_in_8bit`")
    args.load_in_8bit = False
print(args)

model_name_or_path = args.model_path
if args.cluster_mode:
    vision_tower_path = r'/home/oshita/vlm/Link-Context-Learning/clip_vit_large_patch14.pt'  # offline
else:
    vision_tower_path = r'/home/oshita/vlm/Link-Context-Learning/clip_vit_large_patch14.pt'
#endregion

#region configs
model_args = dict(
    type='llava',
    # TODO: process version; current version use default version
    version='v1',

    # checkpoint config
    cache_dir=None,
    model_name_or_path=model_name_or_path,
    vision_tower=vision_tower_path,
    pretrain_mm_mlp_adapter=None,
    # model config
    mm_vision_select_layer=-2,
    model_max_length=2048,
    
    # finetune config
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,
    freeze_mm_projector=False,

    # data process config
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=256,
    mm_use_im_start_end=True,

    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='LLavaConvProcessV1'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='LlavaTextProcessV2'),
        image=dict(type='LlavaImageProcessorV1'),
    ),

    conv_args=dict(
        conv_template=['causal_v1.0', 'hypnotized_ans_v1.0', 'final_v1.0', 'vicuna_v1.1'],
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=2048),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)
model_args = Config(model_args)

training_args = Config(dict(
    bf16=False,
    fp16=True,
    device='cuda',
    fsdp=None,
))

if args.load_in_4bit:
    quantization_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    )
elif args.load_in_8bit:
    quantization_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
        )
    )
else:
    quantization_kwargs = dict()

#endregion

#region Load model and dataset
if not args.remove_model:
    model, preprocessor = load_pretrained_llava(model_args, training_args, **quantization_kwargs)
    preprocessor['target'] = {'boxes': PlainBoxFormatter()}
    tokenizer = preprocessor['text']

    if not getattr(model, 'is_quantized', False):
        model.to(dtype=torch.float16, device=torch.device('cuda'))
    if not getattr(model.model.vision_tower[0], 'is_quantized', False):
        model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device('cuda'))

    dataset_demo = prepare_demo_dataset(model_args=model_args, preprocessor=preprocessor)

    print(f"LLM device: {model.device}, is_quantized: {getattr(model, 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
    print(f"vision device: {model.model.vision_tower[0].device}, is_quantized: {getattr(model.model.vision_tower[0], 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
else:
    print(f'Skip model process.')
#endregion

def get_example_path(img_name):
    path = os.path.join(os.path.dirname(__file__), f'examples/{img_name}')
    return path

def lcl2shot_examples_fn(infer_imgbox, infer_q, pos_imgbox1, pos_a, neg_imgbox1, neg_a):
    return None, None

def convert_img(img):
    if img is None:
        return
    import pdb;pdb.set_trace()
    img = Image.fromarray(img)
    return img

def state_update(state, key, value):
    if value is None:
        return
    # format inputs
    if isinstance(value, str):
        special_tokens = [' <question>',' <image>', '<im_start>', '<im_end>', '[BEGIN EXAMPLE]', '[END EXAMPLE]', '[FINAL QUESTION]']
        for token in special_tokens:
            value = value.replace(token, '')
    state[key].append(value)

def predict(data_meta,class_name,idx,img_path):
    if len(data_meta['infer_q']) == 0:
        raise Exception('Please input question.')
    
    dataset_demo.update_data(data_meta)
    model_inputs = dataset_demo[0]
    model_dtype = next(model.parameters()).dtype
    model_inputs['images'] = model_inputs['images'].to(model_dtype)
    # print(f"model_inputs: {model_inputs}")

    gen_kwargs = dict(
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
    )

    input_ids = model_inputs['input_ids']
    with torch.inference_mode():
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            outputs = model.generate(**model_inputs, **gen_kwargs, return_dict_in_generate=True, output_scores=True)
            output_ids = outputs.sequences

            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            generated_tokens = outputs.sequences[:, input_ids.shape[-1]:]
            import numpy as np
            for tok, score, full_score in zip(generated_tokens[0], transition_scores[0], outputs.scores):
                full_score = full_score[0]
                topk_softmax_score, topk_index = full_score.softmax(dim=-1).topk(5)
                topk_origin_score = full_score[topk_index]
                topk_tokens = tokenizer.convert_ids_to_tokens(topk_index)
                topk_strs = [f"[{idx:5d} | {token:8s} | {oscore:.3f} | {sscore:.2%}]" for idx, token, oscore, sscore in zip(topk_index, topk_tokens, topk_origin_score, topk_softmax_score)]
                # print(",".join(topk_strs))

            # print(tokenizer.batch_decode(generated_tokens))

    input_token_len = input_ids.shape[-1]
    response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # print(f'===== {img_path} class:{class_name} idx:{idx} =====')
    # print(f"question: {tokenizer.batch_decode(model_inputs['labels'], skip_special_tokens=True)[0].replace(' <im_patch>', '')}")
    # with open(f'./{Path(args.model_path).name}_imagenet_test_q{args.q_type}.txt', 'a') as f:
    #     f.write(f"{class_name}__{img_path}\n{response}\n\n")

    print(f"response: {response}")
    with open(f'./{Path(args.model_path).name}_product_knowledge.txt', 'a') as f:
        f.write(f"{class_name}__{img_path}\n{response}\n\n")

def init_vqa_state():
    return {
        'mode' : 'vqa',
        'infer_img': [],
        'infer_q': []
    }


random.seed(42)


def get_random_image_path(dir_path):
    # ディレクトリ内のフォルダリストを取得
    folders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
    # フォルダリストからランダムに1つ選択
    selected_folder = random.choice(folders)
    class_name = Path(selected_folder).name
    selected_folder = selected_folder + "/None"
    # 選択したフォルダ内の画像リストを取得
    images = [f.path for f in os.scandir(selected_folder) if f.is_file() and f.name.endswith(('.png', '.jpg', '.jpeg'))]
    # 画像リストからランダムに1つ選択
    selected_image = random.choice(images)
    return selected_image,class_name
# ディレクトリパス

def get_product_name(product_name):
        product_name = product_name.split('+')[0]
        product_name = product_name.replace('_', ' ')
        return product_name

import pdb;pdb.set_trace()

jsonl_path = "/home/oshita/vlm/Link-Context-Learning/docs/VQA_AC.jsonl"
data = {}
with open(jsonl_path, 'r') as f:
    for line in f:
        _data = json.loads(line)
        img_path = _data['img_path']
        if not img_path in data:
            data[img_path] = {"mode":_data['mode'],
                              "product_name":_data['product_name'],
                              }
with open(args.json_path, 'r') as f:
    questions = json.load(f)
questions = questions['question']

for key in data.keys():
    state = init_vqa_state()

    defect_name = data[key]['mode']
    product_name = get_product_name(data[key]['product_name'])
    infer_imgbox = Image.open(key).convert('RGB')

    q = random.choice(questions).format(product=product_name)

    state_update(state, 'infer_img', infer_imgbox)
    state_update(state, 'infer_q', q)

    predict(state,class_name,0,img_path)



