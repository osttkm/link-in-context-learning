import json
import numpy as np
import os
import math

import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

class CustomModelLossCallback(TrainerCallback):
    def __init__(self, output_dir):
        print(f'callback is initialized.')
        self.iter_loss = []
        self.yesno_loss = []
        self.output_dir = output_dir
        self.device = torch.device('cuda')
    

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # nanは省いているがここは議論の余地あり，またロスの算出方法も雑な気がする...
        if self.iter_loss != []:
            self.yesno_loss.append(self.iter_loss)
            self.iter_loss = []
        else:
            rank = dist.get_rank()
            if rank == 0:
                print('iter_loss is empty')

    def on_step_end(self, args, state, control, **kwargs):
        # rank = dist.get_rank()
        # if rank == 0:
        #     print('step end')
        #     print(kwargs['model'].get_yesno_loss())
        local_loss = kwargs['model'].get_yesno_loss()
        # 損失がNaNであれば0に、そうでなければ1をカウントするテンソルを作成
        local_loss_tensor = torch.tensor([0.0 if math.isnan(local_loss) else local_loss.item()],device=self.device)
        local_valid_count = torch.tensor([0.0 if math.isnan(local_loss) else 1.0],device=self.device)

        # 全GPUからの損失と有効カウントを集約
        dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_valid_count, op=dist.ReduceOp.SUM)

        # 有効な損失値の平均を計算（有効カウントが0でない場合）
        if local_valid_count.item() > 0:
            aggregated_loss = local_loss_tensor.item() / local_valid_count.item()
        else:
            # ここで代替値を設定
            aggregated_loss = -1  # 例: すべてNaNの場合に-1を使用

        self.iter_loss.append(aggregated_loss)
        # rank = dist.get_rank()
        # if rank == 0:
        #     print(aggregated_loss)
 

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        train_steps, train_losses = [], []
        self.yesno_loss = self.fill_nan_with_last_value(self.yesno_loss)
        
        for i in range(len(self.yesno_loss) - 1):
            train_steps.append(i+1)
            train_losses.append(self.yesno_loss[i])
        plt.figure()
        plt.plot(train_steps, train_losses)
        plt.title("training loss of {}".format(self.output_dir))
        plt.xlabel("step")
        plt.ylabel("training loss")
        plt.savefig(os.path.join(self.output_dir, 'yesno_loss.png'), format="png", transparent=True, dpi=300)
        print("Figure saved: {}".format(os.path.join(self.output_dir, 'yesno_loss.png')))
        #yesno_lossの保存
        with open(os.path.join(self.output_dir, 'yesno_loss.json'), "w") as f:
            json.dump(self.yesno_loss, f)
    
    def fill_nan_with_last_value(self, yesno_loss):
        # Convert list to numpy array for easier manipulation
        yesno_loss = np.array(yesno_loss)
        # もし第一配列がnanだったら、それを-1にする
        if math.isnan(yesno_loss[0]):
            yesno_loss[0] = -1

        # Get indices where nan values are present
        nan_indices = np.where(np.isnan(yesno_loss))

        # For each nan index, replace it with the last non-nan value
        for i in nan_indices:
            if i > 0:  # Avoid index error for first element
                yesno_loss[i] = -1

        return yesno_loss.tolist() 