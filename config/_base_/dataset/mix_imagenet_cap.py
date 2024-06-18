_base_ = ['DEFAULT_TRAIN_DATASET.py']


data_args = dict(
    #
    train=dict(
        type='InterleaveDateset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.imagenet_2way_weight}},
            {{_base_.DEFAULT_TRAIN_DATASET.caption}},
        ],
        
        # probabilities=[0.3] + [1./21.*0.7]*21,
        # probabilities=[0.3] + [1./3.*0.7]*3,
        probabilities=[0.75,0.25],
        seed=None,
        stopping_strategy='first_exhausted',
    ),
    validation=None,
    test=None,
    
    # mix training
    use_mix=True,
    icl_dataset_list=[0],

    # compute_metric
    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
    shot=8,
)
