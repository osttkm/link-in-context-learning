_base_ = ['_base_/dataset/DEFAULT_TRAIN_VQA_AC.py', '_base_/model/lcl_7b.py', '_base_/train/llava_fsdp.py']

data_args = dict(
    train=dict(
    **_base_.VQA_AC_TRAIN_COMMON_CFG,
    template_file=r"/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/4_18_remake_VQA_template/VQA_AC_20.json",
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
    num_train_epochs=3,
    save_strategy='epoch',
    # save_steps=10000,
    output_dir = './shikra2pretrain_ac_20',
    save_total_limit=3,
)
