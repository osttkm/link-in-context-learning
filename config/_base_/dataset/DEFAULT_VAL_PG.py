CUSTOM_VAL_DATA_TRAIN = dict(
    type='CustomDatasetTrain_Val_VI',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/Val_PG.jsonl',
    image_folder = r'/dataset/yyama_dataset/VI_images/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

DEFAULT_VAL_PG_CUSTOMDATA_VARIANT = dict(
    customdata_train=dict(**CUSTOM_VAL_DATA_TRAIN),
)