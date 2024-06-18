VQA_AC_TRAIN_COMMON_CFG = dict(
    type='CustomDataset_VQAAC_Train',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/VQA_AC.jsonl',
    # image_folder=r'/home/oshita/vlm/Link-Context-Learning/',
)

DEFAULT_TRAIN_VQA_AC_VARIANT = dict(
    customdata_train=dict(**VQA_AC_TRAIN_COMMON_CFG),
)