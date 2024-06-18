IMAGENET1K_TRAIN = dict(
    type='ImageNet1kDatasetTrain',
    filename=r'/home/oshita/vlm/Link-Context-Learning/LCL_2WAY_WEIGHT/train900_pairs.jsonl',
    image_folder=r'/dataset/imagenet_2012/',
    template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/ICL.json",
    only_image=False,
)

DEFAULT_TRAIN_IMAGENET1K_VARIANT = dict(
    imagenet1k_train=dict(**IMAGENET1K_TRAIN),
)