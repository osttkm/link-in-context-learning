CUSTOM_VI_DATA_TRAIN = dict(
    type='CustomDataset_ICLVI_Train',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/VI.jsonl',
    image_folder = r'/dataset/yyama_dataset/VI_images/',
    # template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/old_one/LCL_VI_20.json"
)

DEFAULT_TRAIN_VI_CUSTOMDATA_VARIANT = dict(
    customdata_train=dict(**CUSTOM_VI_DATA_TRAIN),
)