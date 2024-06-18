# 日付を獲得
now=$(date +"%Y_%m_%d_%H_%M")
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG_FILE=/home/oshita/vlm/Link-Context-Learning/training_log/${now}_nccl.log

# accelerate launch --num_processes 8 \
#         --main_process_port 23786 \
#         mllm/pipeline/finetune.py \
#         config/test_icl_run.py \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/model_result/LCL_pretrained_ac_30/checkpoint-epoch-1 \
#         --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch1_ac_30
# accelerate launch --num_processes 8 \
#         --main_process_port 23786 \
#         mllm/pipeline/finetune.py \
#         config/test_icl_run.py \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/model_result/LCL_pretrained_ac_30/checkpoint-epoch-2 \
#         --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch2_ac_30
accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/test_icl_run.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/model_result/LCL_pretrained_ac_30/checkpoint-epoch-3 \
        --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch3_ac_30
# accelerate launch --num_processes 8 \
#         --main_process_port 23786 \
#         mllm/pipeline/finetune.py \
#         config/test_icl_run.py \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/model_result/LCL_pretrained_ac_20/checkpoint-epoch-1 \
#         --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch1_ac_20
# accelerate launch --num_processes 8 \
#         --main_process_port 23786 \
#         mllm/pipeline/finetune.py \
#         config/test_icl_run.py \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/model_result/LCL_pretrained_ac_20/checkpoint-epoch-2 \
#         --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch2_ac_20
# accelerate launch --num_processes 8 \
#         --main_process_port 23786 \
#         mllm/pipeline/finetune.py \
#         config/test_icl_run.py \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/model_result/LCL_pretrained_ac_20/checkpoint-epoch-3 \
#         --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch3_ac_20

# accelerate launch --num_processes 8 \
#         --main_process_port 23786 \
#         mllm/pipeline/finetune.py \
#         config/test_icl_run.py \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/model_result/shikra2pretrain_ac_20/checkpoint-epoch-3 \
#         --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/shikra2LCL_VI+AC_20_pretrained_epoch3_ac_20

accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/test_icl_run.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/model_result/shikra2pretrain_ac_30/checkpoint-epoch-1 \
        --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/shikra2LCL_VI+AC_20_pretrained_epoch1_ac_20
accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/test_icl_run.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/model_result/shikra2pretrain_ac_30/checkpoint-epoch-2 \
        --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/shikra2LCL_VI+AC_20_pretrained_epoch2_ac_20
# accelerate launch --num_processes 8 \
#         --main_process_port 23786 \
#         mllm/pipeline/finetune.py \
#         config/test_icl_run.py \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/model_result/shikra2pretrain_ac_30/checkpoint-epoch-3 \
#         --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/shikra2LCL_VI+AC_20_pretrained_epoch3_ac_20

sh pretrain_pg.sh
