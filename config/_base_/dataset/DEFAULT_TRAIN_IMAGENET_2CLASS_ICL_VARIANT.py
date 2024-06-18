IMAGENET1K_TRAIN_2CLASS_ICL = dict(
    type='ImageNet1kDatasetTrain_2ClassICL',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/train900_pairs.jsonl',
    image_folder=r'/dataset/imagenet_2012/',
    template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/ICL.json",
    policy="policy_2way_weight",
)

DEFAULT_TRAIN_IMAGENET1K_2CLASS_ICL_VARIANT = dict(
    imagenet1k_train=dict(**IMAGENET1K_TRAIN_2CLASS_ICL),
)