IMAGENET_TEST100 = dict(
    type='ImageNetTest100Eval',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/test100_pairs.jsonl',
    image_folder=r'/dataset/imagenet_2012',
    template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/templateICL.json",
)
IMAGENET_TEST100_2WAY = dict(
    type='ImageNetTest100Eval2Way',
    filename=r'/home/oshita/vlm/Link-Context-Learning/docs/test100_pairs.jsonl',
    image_folder=r'/dataset/imagenet_2012',
    template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/ICL.json",
)