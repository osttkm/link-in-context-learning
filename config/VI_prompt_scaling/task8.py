_base_ = ['_base_/dataset/DEFAULT_TRAIN_FIX_20_VI.py', '_base_/model/lcl_7b.py', '_base_/train/llava_fsdp.py']

data_args = dict(
    train=dict(
    **_base_.CUSTOM_FIX_20_VI_DATA_TRAIN,
    policy="policy_2way_weight",
    ),
    validation=None,
    test=None,

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

training_args = dict(
    num_train_epochs=50,
    save_strategy='steps',
    save_steps=50,
    output_dir = './shikra2VI_FIX_20',
    save_total_limit=5,
)
