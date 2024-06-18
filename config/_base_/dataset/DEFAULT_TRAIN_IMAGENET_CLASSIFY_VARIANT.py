IMAGENET1K_TRAIN_CLASSIFY = dict(
    type='ImageNet1kClassifyDatasetTrain',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/Imagenet_classify_train.jsonl',
    image_folder=r'/dataset/imagenet_2012/',
    template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/VQA.json",
)

DEFAULT_TRAIN_IMAGENET1K_CLASSIFY_VARIANT = dict(
    imagenet1k_train=dict(**IMAGENET1K_TRAIN_CLASSIFY),
)