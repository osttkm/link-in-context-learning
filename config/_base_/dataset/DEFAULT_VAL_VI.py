CUSTOM_VI_DATA_TRAIN = dict(
    type='CustomDatasetTrain_Val_VI',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/Val_VI.jsonl',
    image_folder = r'/dataset/yyama_dataset/VI_images/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

DEFAULT_VAL_VI_CUSTOMDATA_VARIANT = dict(
    customdata_train=dict(**CUSTOM_VI_DATA_TRAIN),
)