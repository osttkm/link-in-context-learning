VI_Loc_TRAIN_COMMON_CFG = dict(
    type='CustomDataset_VILoc_Train',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/VI_Loc.jsonl',
    image_folder=r'/home/oshita/vlm/Link-Context-Learning/',
)

DEFAULT_TRAIN_VI_LOC_VARIANT = dict(
    customdata_train=dict(**VI_Loc_TRAIN_COMMON_CFG),
)