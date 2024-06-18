_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    train=dict(
        type='InterleaveDateset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.vqa_ac_20}},   
            {{_base_.DEFAULT_TRAIN_DATASET.icl_lcl_vi_20}}, 
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
