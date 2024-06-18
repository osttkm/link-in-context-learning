# from pathlib import Path

# # shikraで始まるディレクトリを探す
# dirs = [dir for dir in Path('.').iterdir() if dir.is_dir() and dir.name.startswith('shikra')]

# # エポック数の選択肢
# epoch_nums = [10, 20, 30, 40, 50]
# # epoch_nums = [1,2,3,4,5,6,7,8]

# for dir in dirs:
#     sub_dirs = [sub_dir for sub_dir in dir.iterdir() if sub_dir.is_dir()]
#     # checkpoint-○○の名前を持つディレクトリのみを対象とする
#     checkpoint_dirs = [sub_dir for sub_dir in sub_dirs if 'checkpoint' in sub_dir.name or 'epoch' in sub_dir.name]
#     # 既にepoch-▽▽の名前を持つディレクトリは除外
#     # target_dirs = [sub_dir for sub_dir in checkpoint_dirs if not sub_dir.name.startswith('epoch')]
#     target_dirs = checkpoint_dirs

#     if target_dirs:
#         # イテレーション数を取得し、大小関係に基づいてソート
#         iteration_nums = sorted([int(sub_dir.name.split('-')[-1]) for sub_dir in target_dirs])
#         # イテレーション数とエポック数をマッピング
#         iteration_to_epoch = {num: epoch for num, epoch in zip(iteration_nums, epoch_nums)}

#         for sub_dir in target_dirs:
#             iteration_num = int(sub_dir.name.split('-')[-1])
#             # 新しいディレクトリ名を決定
#             new_name = f"checkpoint-epoch-{iteration_to_epoch[iteration_num]}"
#             # ディレクトリ名を変更
#             sub_dir.rename(sub_dir.parent / new_name)


# # 利用可能なGPUを表示
# import torch

# num_gpus = torch.cuda.device_count()
# if num_gpus > 0:
#     print(f"Number of GPUs available: {num_gpus}")
#     for i in range(num_gpus):
#         device = torch.device(f'cuda:{i}')
#         try:
#             torch.ones((1,), device=device)
#             print(f"GPU {i}: {torch.cuda.get_device_name(i)} is available.")
#         except RuntimeError as e:
#             print(f"GPU {i}: {torch.cuda.get_device_name(i)} is not available. {e}")
# else:
#     print("No GPU available.")


from pathlib import Path

base_path = Path('/home/oshita/vlm/Link-Context-Learning/SHIFT_IMAGE_TOKEN_NON_MIX_LCL_2WAY_WEIGHT')
dirs = [dir for dir in base_path.iterdir() if dir.is_dir() and 'shot' in dir.name]
import pdb;pdb.set_trace()
for dir in dirs:
    sub_dirs = [sub_dir for sub_dir in dir.iterdir() if sub_dir.is_dir()]
    for sub_dir in sub_dirs:
        if not (sub_dir / 'all_results.json').exists():
            print(f"""{sub_dir} doesn't have all_results.json""")

base_path = Path('/home/oshita/vlm/Link-Context-Learning/NON_MIX_LCL_2WAY_WEIGHT')
dirs = [dir for dir in base_path.iterdir() if dir.is_dir() and 'shot' in dir.name]
for dir in dirs:
    sub_dirs = [sub_dir for sub_dir in dir.iterdir() if sub_dir.is_dir()]
    for sub_dir in sub_dirs:
        if not (sub_dir / 'all_results.json').exists():
            print(f"""{sub_dir} doesn't have all_results.json""")
    

