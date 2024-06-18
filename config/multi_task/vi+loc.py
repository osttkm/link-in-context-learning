_base_ = ['_base_/dataset/vi+loc.py', '_base_/model/multi_train_lcl_7b.py', '_base_/train/multi_train_llava_fsdp.py']
    
training_args = dict(
    num_train_epochs=50,
    save_strategy='steps',
    save_steps=100,
    output_dir = './shuffle20_lcl_vi+loc',
    save_total_limit=5,
)
