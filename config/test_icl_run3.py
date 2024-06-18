_base_ = ['_base_/dataset/mix_imagenet_vqa.py', '_base_/model/lcl_7b.py', '_base_/train/test_llava_fsdp.py']

training_args = dict(
    num_train_epochs=50,
    save_strategy='steps', # steps, epoch, no
    save_steps=500,
    output_dir = './Mixed_5050_LCL_2WAY_WEIGHT',
    save_total_limit=15,
)
