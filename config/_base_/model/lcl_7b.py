model_args = dict(
    type='oshita_llava',
    # TODO: process version; current version use default version
    version='v1',

    # checkpoint config
    cache_dir=None,
    model_name_or_path=r'/home/oshita/vlm/Link-Context-Learning/LCL_2WAY_WEIGHT/',
    vision_tower=r'openai/clip-vit-large-patch14',
    pretrain_mm_mlp_adapter=None,
    # model config
    mm_vision_select_layer=-2,
    model_max_length=30000,
    
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
        conv_template=['hypnotized_v1.0','hypnotized_v1.1','hypnotized_ans_v1.0','vicuna_v1.1','causal_v1.0','final_v1.0'],
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=2048),
        # conv_template='vicuna_v1.1',
        # transforms=dict(type='Expand2square'),
        # tokenize_kwargs=dict(truncation_size=2048),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)
