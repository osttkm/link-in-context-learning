_base_ = ['_base_/dataset/vi+ac+loc.py', '_base_/model/multi_train_lcl_7b.py', '_base_/train/multi_train_llava_fsdp.py']
    
training_args = dict(
    num_train_epochs=50,
    save_strategy='steps',
    save_steps=150,
    output_dir = './shuffle20_lcl_vi+ac+loc',
    save_total_limit=5,
)
