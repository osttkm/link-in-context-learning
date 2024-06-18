CUSTOM_AC_DATA_TRAIN = dict(
    type='CustomDatasetTrain_AC',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/AC.jsonl',
    image_folder = r'/dataset/yyama_dataset/VI_images/',
    template_file=r"{{fileDirname}}/template/ICL.json",
)

DEFAULT_TRAIN_AC_CUSTOMDATA_VARIANT = dict(
    customdata_train=dict(**CUSTOM_AC_DATA_TRAIN),
)