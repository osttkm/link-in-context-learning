import os
import re
import sys
import random
import logging
import argparse
import warnings
import json
from pathlib import Path

import torch
import torch.distributed as dist
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
from custom_dataset import prepare_demo_dataset

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
parser.add_argument('--eval_mode', default='VI', type=str)
parser.add_argument('--model_path', default=r'/home/oshita/vlm/Link-Context-Learning/model_result/PG+LCL_VI')
parser.add_argument('--json_path', default=r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/LCL_VI.json')
parser.add_argument('--server_name', default=SLURM_ENV.get('SLURM_JOB_NODELIST', None))
parser.add_argument('--server_port', type=int, default=20488)
parser.add_argument('--remove_model', action='store_true')
parser.add_argument('--load_in_8bit', action='store_true')
parser.add_argument('--load_in_4bit', action='store_true')
args = parser.parse_args()
args.cluster_mode = bool(SLURM_ENV)
if args.load_in_4bit and args.load_in_8bit:
    warnings.warn("use `load_in_4bit` and `load_in_8bit` at the same time. ignore `load_in_8bit`")
    args.load_in_8bit = False
# print(args)

# seedの固定
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


model_name_or_path = args.model_path

if args.cluster_mode:
    vision_tower_path = r'/home/oshita/vlm/Link-Context-Learning/clip_vit_large_patch14.pt'  # offline
else:
    vision_tower_path = r'/home/oshita/vlm/Link-Context-Learning/clip_vit_large_patch14.pt'

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



# Model setup
if not args.remove_model:
    # print(model_args)
    # print(training_args)
    # print(**quantization_kwargs)
    model, preprocessor = load_pretrained_llava(model_args, training_args, **quantization_kwargs)
    preprocessor['target'] = {'boxes': PlainBoxFormatter()}
    tokenizer = preprocessor['text']

    if not getattr(model, 'is_quantized', False):
        model.to(dtype=torch.float16, device=torch.device('cuda'))
    if not getattr(model.model.vision_tower[0], 'is_quantized', False):
        model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device('cuda'))

    dataset_demo = prepare_demo_dataset(model_args=model_args, preprocessor=preprocessor, json_path=args.json_path)

    # print(f"LLM device: {model.device}, is_quantized: {getattr(model, 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
    # print(f"vision device: {model.model.vision_tower[0].device}, is_quantized: {getattr(model.model.vision_tower[0], 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
else:
    print(f'Skip model process.')


def convert_img(img):
    if img is None:
        return
    # img = Image.fromarray(img)
    img = Image.open(img)
    return img

def init_lcl_state():
            return {
                'mode' : 'lcl',
                'pos_img': [],
                'neg_img': [],
                'infer_img': [],
                'pos_a': [],
                'neg_a': [],
                'infer_q': []
            }

def state_update(state, key, value):
    if value is None:
        return
    # format inputs
    if key != 'infer_q':
        if isinstance(value, str):
            special_tokens = ['<image>', '<im_start>', '<im_end>', '[BEGIN EXAMPLE]', '[END EXAMPLE]', '[FINAL QUESTION]']
            for token in special_tokens:
                value = value.replace(token, '')
    elif key=='infer_q':
        pass
    state[key].append(value)

def set_lcl_state(state, pos_imgbox1, pos_a, neg_imgbox1, neg_a, infer_imgbox, infer_q):
            # set state
            state = init_lcl_state()
            pos_imgbox1 = convert_img(pos_imgbox1)
            # pos_imgbox2 = convert_img(pos_imgbox2)
            neg_imgbox1 = convert_img(neg_imgbox1)
            # neg_imgbox2 = convert_img(neg_imgbox2)
            infer_imgbox = convert_img(infer_imgbox)

            state_update(state, 'pos_img', pos_imgbox1)
            # state_update(state, 'pos_img', pos_imgbox2)
            state_update(state, 'neg_img', neg_imgbox1)
            # state_update(state, 'neg_img', neg_imgbox2)

            if len(state['pos_img']) + len(state['neg_img']) == 0:
                grWarning(f"The LCL mode requires providing support samples, otherwise the results may not meet expectations. \
                        Alternatively, you can try inferring a single image directly in VQA mode.")

            state_update(state, 'pos_a', pos_a)
            state_update(state, 'neg_a', neg_a)

            state_update(state, 'infer_img', infer_imgbox)
            state_update(state, 'infer_q', infer_q)
            return state

class utils():
    def __init__(self):
        super().__init__()
        self.path = args.json_path
        with open(self.path) as f:
            json_data = json.load(f)

        # self.ac_context_question_lines = json_data['context_question_lines']
        # self.ac_no_responses_array = json_data['no_responses_array']
        # self.ac_yes_responses_array = json_data['yes_responses_array']
        # self.ac_query_question_lines = json_data['query_question_lines']

        if "LCL" in self.path:
            self.vi_query_question_lines = np.array(json_data['query_VI_instructions'])
        else:
            self.vi_query_question_lines = np.array(json_data['VI_instructions'])
        self.vi_context_question_lines = np.array(json_data['VI_instructions'])
        self.vi_no_responses_array = np.array(json_data['answers_ok'])
        self.vi_yes_responses_array = np.array(json_data['answers_ng'])

    # def get_AC_QA(self):
    #     idx = random.randint(0,29)
    #     # u need adding if process later.
    #     context_question = self.ac_context_question_lines[idx][0]
    #     yes_answer = self.ac_yes_responses_array[idx][0]
    #     no_answer = self.ac_no_responses_array[idx][0]
    #     query_quastion = self.ac_query_question_lines[idx][0]
    #     return context_question, yes_answer, no_answer, query_quastion
    def get_VI_QA(self):
        idx = random.randint(0,12)
        # u need adding if process later.
        context_question = self.vi_context_question_lines[idx][0]
        yes_answer = self.vi_yes_responses_array[idx][0]
        no_answer = self.vi_no_responses_array[idx][0]
        query_quastion = self.vi_query_question_lines[idx][0]
        return context_question, yes_answer, no_answer, query_quastion

def get_defect_name(name):
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


def predict(data_meta,txt_path,query_path,defect_name,mode=1,eval_mode="AC",speacis=None):
    dataset_demo.update_data(data_meta,defect_name,mode,eval_mode,speacis)
    model_inputs = dataset_demo[0]

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


    input_token_len = input_ids.shape[-1]
    response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    print(f"response: {response}")
    # txtに書き込み
    with open(txt_path, mode='a') as f:
        f.write(f'query:{query_path} {response}\n')

qa = utils()
speasies = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
for num in range(0,4,1):
    for speacis in speasies:
        data = {}
        os.makedirs(f'/home/oshita/vlm/Link-Context-Learning/mllm/demo/result/mvtec/{speacis}',exist_ok=True)
        if re.match(r'checkpoint*', args.model_path.split('/')[-1]):
            dir_name = args.model_path.split('/')[-2]
        else:
            dir_name = args.model_path.split('/')[-1]
        txt_path = f'/home/oshita/vlm/Link-Context-Learning/mllm/demo/result/mvtec/{speacis}/{dir_name}_{num}_context_inference.txt'
        with open(txt_path, mode='w') as f:
            f.close()

        dataset_path = Path(f'/dataset/mvtec/{speacis}/test')
        ano_class_dir = [p for p in dataset_path.iterdir() if p.is_dir()]
        for _ano_class in ano_class_dir:
            # _ano_classにあるpngファイルをすべて取得
            png_files = [p for p in _ano_class.iterdir() if p.suffix == '.png']
            data[_ano_class.name] = np.array(png_files)
        names = list(data.keys())
        for name in names:
            if name == 'good':
                pass
            else:
                if args.eval_mode == "AC":
                    for p in data[name]:
                        p = str(p)
                        state = init_lcl_state()
                        pos_imgbox1 = convert_img(str(data[name][num]))
                        neg_imgbox1 = convert_img(str(data['good'][num]))
                        infer_imgbox = convert_img(p)

                        qa_list = qa.get_AC_QA()
                        state_update(state, 'pos_img', pos_imgbox1) #異常   
                        state_update(state, 'neg_img', neg_imgbox1) #正常
                        state_update(state, 'pos_a', qa_list[1])    # yes
                        state_update(state, 'neg_a', qa_list[2])    # no
                        state_update(state, 'infer_img', infer_imgbox)
                        # print(f'demo_local:{qa_list[3]}')
                        state_update(state, 'infer_q', qa_list[3])

                        defect_pair = speacis+'+'+name
                        predict(state,txt_path,p,get_defect_name(defect_pair),mode=1,eval_mode=args.eval_mode)

                    for p in data['good']:
                        state = init_lcl_state()
                        pos_imgbox1 = convert_img(str(data['good'][num]))
                        neg_imgbox1 = convert_img(str(data[name][num]))
                        infer_imgbox = convert_img(p)
                        qa_list = qa.get_AC_QA()
                        assert Path(p).parent.name == 'good'

                        state_update(state, 'pos_img', pos_imgbox1) #正常
                        state_update(state, 'neg_img', neg_imgbox1) #異常
                        state_update(state, 'pos_a', qa_list[2])    # no
                        state_update(state, 'neg_a', qa_list[1])    # yes
                        state_update(state, 'infer_img', infer_imgbox)
                        state_update(state, 'infer_q', qa_list[3])
                        # print(f'demo_local:{qa_list[3]}')

                        defect_pair = speacis+'+'+name
                        predict(state,txt_path,p,get_defect_name(defect_pair),mode=2,eval_mode=args.eval_mode)

                elif args.eval_mode == "VI":
                    for p in data[name]:
                        p = str(p)
                        state = init_lcl_state()
                        pos_imgbox1 = Image.new('RGB', (224, 224), (255, 255, 255))
                        neg_imgbox1 = Image.new('RGB', (224, 224), (255, 255, 255))
                        infer_imgbox = convert_img(p)

                        qa_list = qa.get_VI_QA()
                        state_update(state, 'pos_img', pos_imgbox1) #異常   
                        state_update(state, 'neg_img', neg_imgbox1) #正常
                        state_update(state, 'pos_a', qa_list[1])    # yes
                        state_update(state, 'neg_a', qa_list[2])    # no
                        state_update(state, 'infer_img', infer_imgbox)
                        # print(f'demo_local:{qa_list[3]}')
                        state_update(state, 'infer_q', qa_list[3])

                        defect_pair = speacis+'+'+name
                        predict(state,txt_path,p,get_defect_name(defect_pair),mode=1,eval_mode=args.eval_mode,speacis=speacis)
                    
                    for p in data['good']:
                        state = init_lcl_state()
                        pos_imgbox1 = Image.new('RGB', (224, 224), (255, 255, 255))
                        neg_imgbox1 = Image.new('RGB', (224, 224), (255, 255, 255))
                        infer_imgbox = convert_img(p)
                        qa_list = qa.get_VI_QA()
                        assert Path(p).parent.name == 'good'

                        state_update(state, 'pos_img', pos_imgbox1) #正常
                        state_update(state, 'neg_img', neg_imgbox1) #異常
                        state_update(state, 'pos_a', qa_list[2])    # no
                        state_update(state, 'neg_a', qa_list[1])    # yes
                        state_update(state, 'infer_img', infer_imgbox)
                        state_update(state, 'infer_q', qa_list[3])
                        # print(f'demo_local:{qa_list[3]}')

                        defect_pair = speacis+'+'+name
                        predict(state,txt_path,p,get_defect_name(defect_pair),mode=2,eval_mode=args.eval_mode,speacis=speacis)
                elif args.eval_mode == "AC+VI":
                    for p in data[name]:
                        p = str(p)
                        state = init_lcl_state()
                        pos_imgbox1 = Image.new('RGB', (224, 224), (255, 255, 255))
                        neg_imgbox1 = Image.new('RGB', (224, 224), (255, 255, 255))
                        infer_imgbox = convert_img(p)

                        qa_list = qa.get_VI_QA()
                        state_update(state, 'pos_img', pos_imgbox1) #異常   
                        state_update(state, 'neg_img', neg_imgbox1) #正常
                        state_update(state, 'pos_a', qa_list[1])    # yes
                        state_update(state, 'neg_a', qa_list[2])    # no
                        state_update(state, 'infer_img', infer_imgbox)
                        state_update(state, 'infer_q', qa_list[3])

                        defect_pair = speacis+'+'+name
                        predict(state,txt_path,p,get_defect_name(defect_pair),mode=1,eval_mode=args.eval_mode,speacis=speacis)
                    
                    for p in data['good']:
                        state = init_lcl_state()
                        pos_imgbox1 = Image.new('RGB', (224, 224), (255, 255, 255))
                        neg_imgbox1 = Image.new('RGB', (224, 224), (255, 255, 255))
                        infer_imgbox = convert_img(p)
                        qa_list = qa.get_VI_QA()
                        assert Path(p).parent.name == 'good'

                        state_update(state, 'pos_img', pos_imgbox1) #正常
                        state_update(state, 'neg_img', neg_imgbox1) #異常
                        state_update(state, 'pos_a', qa_list[2])    # no
                        state_update(state, 'neg_a', qa_list[1])    # yes
                        state_update(state, 'infer_img', infer_imgbox)
                        state_update(state, 'infer_q', qa_list[3])

                        defect_pair = speacis+'+'+name
                        predict(state,txt_path,p,get_defect_name(defect_pair),mode=2,eval_mode=args.eval_mode,speacis=speacis)




