_base_ = ['_base_/dataset/vi+ac+pg.py', '_base_/model/multi_train_lcl_7b.py', '_base_/train/multi_train_llava_fsdp.py']
    
training_args = dict(
    num_train_epochs=50,
    save_strategy='steps',
    save_steps=140,
    output_dir = './shikra2shuffle20_lcl_vi+ac+pg',
    save_total_limit=5,
)
