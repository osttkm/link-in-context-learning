# # 日付を獲得
# now=$(date +"%Y_%m_%d_%H_%M")
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_CUMEM_ENABLE=0
# export NCCL_DEBUG_FILE=/home/oshita/vlm/Link-Context-Learning/training_log/${now}_nccl.log


accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/test_single_run.py \
        --cfg-options data_args.use_icl=False \
        --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/LCL_2WAY_WEIGHT \
        --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/imagenet1k_classify \



# accelerate launch --num_processes 8 \
#         --main_process_port 23786 \
#         mllm/pipeline/finetune.py \
#         config/imagenet_2class_icl.py \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/imagenet1k_classify \
#         --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/cls_icl \


# accelerate launch --num_processes 8 \
#         --main_process_port 23786 \
#         mllm/pipeline/finetune.py \
#         config/imagenet_2class_icl_mix.py \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/imagenet1k_classify \
#         --cfg-options training_args.output_dir=/home/oshita/vlm/Link-Context-Learning/cls_cls+icl \