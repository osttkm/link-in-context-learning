_base_ = ['_base_/dataset/DEFAULT_TRAIN_IMAGENET.py', '_base_/model/lcl_7b.py', '_base_/train/test_llava_fsdp.py']

data_args = dict(
    train=dict(
    **_base_.IMAGENET1K_TRAIN,
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
    shot=8,
)

training_args = dict(
    num_train_epochs=50,
    save_strategy='steps', # steps, epoch, no
    save_steps=500,
    output_dir = './DEFAULT_LCL_2WAY_WEIGHT',
    save_total_limit=15,
)
