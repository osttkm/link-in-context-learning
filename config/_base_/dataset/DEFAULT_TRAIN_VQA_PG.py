VQA_PG_TRAIN_COMMON_CFG = dict(
    type='CustomDataset_VQAPG_Train',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/VQA_PG.jsonl',
    image_folder=r'zz1424:s3://publicdataset_49/VQAv2/unzip/',
    # template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/VQA.json",
)

DEFAULT_TRAIN_VQA_PG_VARIANT = dict(
    customdata_train=dict(**VQA_PG_TRAIN_COMMON_CFG),
)


