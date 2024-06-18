import os
import sys
import math
import yaml
import logging
import pathlib
import typing
import warnings
from collections import Counter


SLURM_ENV = {k: v for k, v in os.environ.items() if 'SLURM' in k}
# print(f"SLURM_ENV: {SLURM_ENV}")
project_path = pathlib.Path(__file__).parent.parent.parent
# print(f"add project path: `{project_path}` to path to enable import form mllm")
sys.path.append(str(project_path))

import torch
import torch.cuda
import torch.distributed as dist
import accelerate

from mllm.config import prepare_args
from mllm.models import load_pretrained
from mllm.utils import print_trainable_params
from mllm.engine import prepare_trainer_collator
from mllm.dataset import prepare_data, smart_prepare_target_processor
from custom_callback import CustomModelSaveCallback
from custom_loss_callback import CustomModelLossCallback

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)



# # 日付と時刻を取得,文字列に変換
# import datetime
# now = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M')
# with open(f'/home/oshita/vlm/Link-Context-Learning/training_log/{now}.log', 'w') as f:
#     f.write(f'==========log start at {now}==========\n')
#     f.close()
# handler = logging.FileHandler(f'/home/oshita/vlm/Link-Context-Learning/training_log/{now}.log')
# handler.setLevel(logging.ERROR)


def main():
    cfg, training_args = prepare_args()
    model, preprocessor = load_pretrained(cfg.model_args, training_args)
    
    # Some ugly codes to inject target_processor into preprocessor. maybe effect model. (e.g. add special token; resize embedding)
    model, preprocessor = smart_prepare_target_processor(model, preprocessor, cfg.model_args, training_args)

    # Prepare data_collator
    collator_kwargs = cfg.data_args.collator_kwargs
    trainer_cls, data_collator_dict = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
    dataset, compute_metrics = prepare_data(cfg.data_args, cfg.model_args, training_args, preprocessor)


    # Initialize Trainer
    # custom_save_callback = CustomModelSaveCallback()
    # custom_loss_callback = CustomModelLossCallback(training_args.output_dir)
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=preprocessor['text'],
        train_dataset=dataset['train'] if training_args.do_train else None,
        eval_dataset=dataset['validation'] if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        **data_collator_dict,
    )

    # Training
    accelerator = accelerate.Accelerator()
    # """FSDPで全プロセスに共有変数を定義したいときは以下のようにする"""
    # # 全プロセスでsave_stepsを初期化，tensorとして占有
    """
    FSDPにおいてプロセス間で学習ステップがどのように扱か可視化
    ->プロセスによってそう学習ステップ数が違う．おそらくメインプロセス学習数か多数決の学習数に合わせて学習が行われている
    """
    # save_steps = torch.tensor(0, dtype=torch.int32, device=accelerator.device)
    # if accelerator.is_local_main_process:
    #     dataset_length = len(dataset['train'])
    #     total_iterations = math.ceil(dataset_length / (trainer.args.per_device_train_batch_size * accelerator.num_processes)) * trainer.args.num_train_epochs
    #     calculated_save_steps = int(total_iterations / trainer.args.save_total_limit)
    #     save_steps.fill_(calculated_save_steps)
    # dist.broadcast(save_steps, src=0)


    # custom_save_callback.set_trainer(trainer)
    # custom_save_callback.set_save_interval(save_steps)
    # trainer.add_callback(custom_save_callback)
    # trainer.add_callback(custom_loss_callback)
    
    config = accelerator.state
    os.makedirs(training_args.output_dir, exist_ok=True)
            
    if training_args.do_train:
        try:
            if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
                train_result = trainer.train(resume_from_checkpoint=True)
            else:
                train_result = trainer.train()
            trainer.log_metrics("train", train_result.metrics)  # noqa
            trainer.save_metrics("train", train_result.metrics)  # noqa
            trainer.save_model()
        except RuntimeError as e:
            print(f"got RuntimeError: {e.args}")
            try:
                print(f"#### device {training_args.local_rank} summary ####\n{torch.cuda.memory_summary(training_args.local_rank)}")
            except Exception as inner_e:
                print(f"get Exception when show cuda summary: {inner_e.args}")
            raise e
        finally:
            trainer.save_state()  # noqa
            trainer.plot_loss()

    # save cfg to output_dir
    try:
        output_dir = training_args.output_dir
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg.dump(os.path.join(output_dir, "cfg.py"))
    except Exception as e:
        warnings.warn(f'try to save cfg to output_dir, but get exception {e.args}')

    # Keyword arguments for `model.generate`
    gen_kwargs = dict(cfg.data_args.gen_kwargs)
    gen_kwargs.setdefault('use_cache', True)
    # important for use model.generate in batch mode. some model config with wrong special_token_id
    # (e.g. llava generationConfig set pad_token_id to -1)
    if hasattr(cfg.model_args, 'gen_kwargs_set_pad_token_id') and cfg.model_args.gen_kwargs_set_pad_token_id:
        gen_kwargs['pad_token_id'] = preprocessor['text'].pad_token_id
    if hasattr(cfg.model_args, 'gen_kwargs_set_bos_token_id') and cfg.model_args.gen_kwargs_set_bos_token_id:
        gen_kwargs['bos_token_id'] = preprocessor['text'].bos_token_id
    if hasattr(cfg.model_args, 'gen_kwargs_set_eos_token_id') and cfg.model_args.gen_kwargs_set_eos_token_id:
        gen_kwargs['eos_token_id'] = preprocessor['text'].eos_token_id

    # Evaluation
    if training_args.do_eval:
        if hasattr(trainer, '_test_collator') and hasattr(trainer, '_eval_collator') \
                and trainer._test_collator != trainer._eval_collator:  # noqa
            warnings.warn('[WARNING!!!] use different collator for eval and test. but do_eval and '
                          'do_predict both use trainer.predict (i.e. only test_collator is used.)')
        eval_results = trainer.predict(dataset['validation'], metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", eval_results.metrics)  # noqa
        trainer.save_metrics("eval", eval_results.metrics)  # noqa
        trainer.save_prediction(eval_results, file_key_prefix='eval')

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset['test'], metric_key_prefix="test", **gen_kwargs)
        trainer.log_metrics("test", predict_results.metrics)  # noqa
        trainer.save_metrics("test", predict_results.metrics)  # noqa
        trainer.save_prediction(predict_results, file_key_prefix='test')

    # Multi Predict
    if training_args.do_multi_predict:
        old_compute_metrics = trainer.compute_metrics
        multitest = dataset['multitest']
        multitest = typing.cast(dict, multitest)
        for _idx, (k, item) in enumerate(multitest.items()):
            print(f'processing multitest set {_idx}/{len(multitest)}: {k}')
            _ds = item['dataset']
            _compute_metrics = item['compute_metric']
            _prefix = f"multitest_{k}"

            trainer.compute_metrics = _compute_metrics
            # _pred_results, probs = trainer.predict(_ds, metric_key_prefix=_prefix, **gen_kwargs)
            _pred_results = trainer.predict(_ds, metric_key_prefix=_prefix, **gen_kwargs)
            trainer.log_metrics(_prefix, _pred_results.metrics)  # noqa
            trainer.save_metrics(_prefix, _pred_results.metrics)  # noqa
            trainer.save_prediction(_pred_results, file_key_prefix=_prefix, probs=None)
        trainer.compute_metrics = old_compute_metrics


# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
