_base_ = [
    'DEFAULT_TRAIN_GQA_VARIANT.py',
    'DEFAULT_TRAIN_CLEVR_VARIANT.py',
    'DEFAULT_TRAIN_POINT_VARIANT.py',
    'DEFAULT_TRAIN_GPTGEN_VARIANT.py',
    'DEFAULT_TRAIN_VCR_VARIANT.py',
    'DEFAULT_TRAIN_VQAv2_VARIANT.py',
    'DEFAULT_TRAIN_VQAEX_VARIANT.py',
    'DEFAULT_TRAIN_V3DET_VARIANT.py',
    'DEFAULT_TRAIN_IMAGENET.py',
    'DEFAULT_TRAIN_VQA_PG.py',
    'DEFAULT_TRAIN_VQA_AC.py',
    'DEFAULT_TRAIN_ICL_VI.py',
    'DEFAULT_TRAIN_VI_Loc.py',
    'DEFAULT_TRAIN_IMAGENET_CLASSIFY_VARIANT.py',
    'DEFAULT_TRAIN_IMAGENET_2CLASS_ICL_VARIANT.py',
]

DEFAULT_TRAIN_DATASET = dict(
    flickr=dict(
        type='FlickrDataset',
        filename=r'/home/oshita/vlm/shikra/data/CWB_flickr30k_train.jsonl',
        image_folder=r'/dataset/flickr30k/flickr30k-images',
        template_file=r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/flickr30k.json',
    ),
    rec=dict(
        type='RECDataset',
        filename=r'/home/oshita/vlm/shikra/data/REC_ref3_train.jsonl',
        template_file=r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/REC.json',
    ),
    recvg=dict(
        type='RECDataset',
        filename=r'/home/oshita/vlm/shikra/data/GC_genome196_train_revised.jsonl',
        template_file=r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/REC.json',
    ),
    reg=dict(
        type='REGDataset',
        filename=r'/home/oshita/vlm/shikra/data/REC_ref3_train.jsonl',
        template_file=r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/REG.json',
    ),
    gc=dict(
        type='GCDataset',
        filename=r'/home/oshita/vlm/shikra/data/GC_genome196_train_revised.jsonl',
        template_file=r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/GC.json',
    ),
    caption=dict(
        type='CaptionDataset',
        filename=r'/home/oshita/vlm/shikra/data/CAP_coco2014_train.jsonl',
        template_file=r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/image_cap.json',
        image_folder = r'/dataset/mscoco2014/train2014',
    ),
    llavacc3m=dict(
        type='InstructDataset',
        filename=r"/home/oshita/vlm/shikra/data/llava_cc3m.jsonl",
        image_folder=r'/dataset/cc3m/train/',   
    ),
    llavalcs=dict(
        type='InstructDataset',
        filename=r"/home/oshita/vlm/shikra/data/blip_laion_cc_sbu_558k.jsonl",
        image_folder=r'/dataset/cc3m/train/',   
    ),
    instruct=dict(
        type='InstructDataset',
        filename=r'/home/oshita/vlm/shikra/data/revised_llava_instruct_150k.jsonl',
        image_folder=r'/dataset/cc3m/train/',
        add_coco_prefix=False,
    ),
    v3det=dict(
        type='V3DetDataset',
        filename=r'/mnt/lustre/share_data/zhangzhao2/VG/v3det/v3det_2023_v1_train_neig_expired.json',
        image_folder=r'/home/oshita/vlm/Link-Context-Learning',
        template_file=r"{{fileDirname}}/template/ICL.json",
    ),
    vqav2=dict(
        type='VQAv2Dataset',
        filename=r'/home/oshita/vlm/Link-Context-Learning/docs/v2_OpenEnded_mscoco_train2014_questions.jsonl',
        image_folder=r'/dataset/mscoco2014',
        template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/VQA.json",
    ),
    imagenet_2way_weight = dict(
    **_base_.DEFAULT_TRAIN_IMAGENET1K_VARIANT.imagenet1k_train,
    policy="policy_2way_weight"
    ),
    icl_vi_10 = dict(
    **_base_.CUSTOM_VI_DATA_TRAIN,
    template_file="/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_ICL_template/VI_10.json",
    lcl=False
    ),
    icl_vi_20 = dict(
    **_base_.CUSTOM_VI_DATA_TRAIN,
    template_file="/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_ICL_template/VI_20.json",
    lcl=False
    ),
    icl_vi_30 = dict(
    **_base_.CUSTOM_VI_DATA_TRAIN,
    template_file="/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_ICL_template/VI_30.json",
    lcl=False
    ),
    icl_lcl_vi_10 = dict(
    **_base_.CUSTOM_VI_DATA_TRAIN,
    template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_ICL_template/VI_10.json",
    lcl=True
    ),
    icl_lcl_vi_20 = dict(
    **_base_.CUSTOM_VI_DATA_TRAIN,
    template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_ICL_template/VI_20.json",
    # template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/old_one/LCL_VI_20.json",
    lcl=True
    ),
    icl_lcl_vi_30 = dict(
    **_base_.CUSTOM_VI_DATA_TRAIN,
    template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_ICL_template/VI_30.json",
    lcl=True
    ),
    vqa_ac_10=dict(
        **_base_.VQA_AC_TRAIN_COMMON_CFG,
        template_file = r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_VQA_template/VQA_AC_10.json',
    ),
    vqa_ac_20=dict(
        **_base_.VQA_AC_TRAIN_COMMON_CFG,
        template_file = r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_VQA_template/VQA_AC_20.json',
    ),
    vqa_ac_30=dict(
        **_base_.VQA_AC_TRAIN_COMMON_CFG,
        template_file = r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_VQA_template/VQA_AC_30.json',
    ),
    vqa_pg=dict(
        **_base_.VQA_PG_TRAIN_COMMON_CFG,
    ),
    vi_loc_10=dict(
        **_base_.VI_Loc_TRAIN_COMMON_CFG,
        template_file = r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_VQA_template/VI_Loc_10.json',
    ),
    vi_loc_20=dict(
        **_base_.VI_Loc_TRAIN_COMMON_CFG,
        template_file = r'/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_VQA_template/VI_Loc_20.json',
    ),
    imagenet_classify=dict(
        **_base_.IMAGENET1K_TRAIN_CLASSIFY,
    ),
    imagenet_2class_icl=dict(
        **_base_.IMAGENET1K_TRAIN_2CLASS_ICL,
    ),

    **_base_.DEFAULT_TRAIN_GQA_VARIANT,
    **_base_.DEFAULT_TRAIN_CLEVR_VARIANT,
    **_base_.DEFAULT_TRAIN_POINT_VARIANT,
    **_base_.DEFAULT_TRAIN_GPTGEN_VARIANT,
    **_base_.DEFAULT_TRAIN_VCR_VARIANT,
    **_base_.DEFAULT_TRAIN_VQAEX_VARIANT,
    **_base_.DEFAULT_TRAIN_V3DET_VARIANT,
)
