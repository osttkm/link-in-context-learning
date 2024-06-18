from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import torch.distributed as dist

class CustomModelSaveCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None  # 初期化時にはTrainerインスタンスは設定しない
        self.save_interval = 0
    def set_trainer(self, trainer_instance):
        self.trainer = trainer_instance
    def set_save_interval(self, interval):
        self.save_interval = interval.item()
        print(f'set save interval : {self.save_interval}!!')

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        rank = dist.get_rank()
        print(f"NODE{rank}: {state.max_steps//args.save_total_limit}")
        # print(f"NODE{rank}: {state.global_step % (state.max_steps//args.save_total_limit) == 0}")
        # rank0_onlyを使用している際の挙動．rank0のみが保存するので重み集約は不要であり，バリアで全GPUの処理が終了してから重みを保存している
        # rank0_only=Falseの際はGPT曰く，別々のフォルダに重みが保存されるらしく，一つのファイルに保存したかったら集約が必要らしい
        # しかし，挙動的にはたぶん嘘．ようわからん．でも安パイはrank0_only=True，OOMするならCPUにオフロード．タイムアウトするならNCCLのタイムアウトを長めに設定するのがよさげ

        # ここで辺に変数を使用するとプロセスによって異なる値が入っており，プロセス間でハングする可能性があるので注意
        if (state.global_step % self.save_interval)==0:
            print(f'RANK:{rank}__{state.max_steps / self.trainer.args.save_total_limit}')
            print('start saving!!')
            # dist.barrier() バリアがあると動作しない．DDP専用？？
            output_dir = f"{args.output_dir}/checkpoint-{int(state.global_step)}"
            self.trainer.save_model(output_dir)
            print(f"Saving model to {args.output_dir} at step {state.global_step}")

    # def on_epoch_end(self, args, state, control, **kwargs):
    #     print('one epoch ends!!')
    #     if (state.epoch % self.save_interval == 0 or state.epoch == state.num_train_epochs) and self.trainer is not None:
    #         dist.barrier()
    #         output_dir = f"{args.output_dir}/checkpoint-epoch-{int(state.epoch)}"
    #         print(f"Saving model to {output_dir} at epoch {int(state.epoch)}")
    #         self.trainer.save_model(output_dir)
    #         # self.trainer._save(output_dir)
