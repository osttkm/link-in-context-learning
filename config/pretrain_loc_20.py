_base_ = ['_base_/dataset/DEFAULT_TRAIN_DATASET.py', '_base_/model/lcl_7b.py', '_base_/train/llava_fsdp.py']

data_args = dict(
    train=dict(
    **_base_.DEFAULT_TRAIN_DATASET.vi_loc_20,
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
    num_train_epochs=10,
    save_strategy='steps',
    save_steps=10000,
    output_dir = './shikra2pretrain_loc_20',
    save_total_limit=5,
)
