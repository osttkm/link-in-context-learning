_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    train=dict(
        type='InterleaveDateset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.vi_loc}},   
            {{_base_.DEFAULT_TRAIN_DATASET.custom_shuffle_20_lcl_vi_data}}, 
        ],
        probabilities=[1/2.*1.0]*2,
        seed=None,
        stopping_strategy='first_exhausted',
    ),
    validation=None,
    test=None,
    
    # mix training
    use_mix=True,
    icl_dataset_list=[1],

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
)
