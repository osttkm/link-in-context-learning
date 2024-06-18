# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_CUMEM_ENABLE=0
# export NCCL_DEBUG_FILE=/home/oshita/vlm/Link-Context-Learning/training_log/${now}_nccl.log
accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/test_icl_run.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/home/oshita/vlm/Link-Context-Learning/LCL_2WAY_WEIGHT 
