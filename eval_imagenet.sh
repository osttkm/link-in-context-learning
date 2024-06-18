# shots=(1 2 3 4 5 6 7 8 9 10)
# checkpoints=("500" "1000" "1500" "2000" "2500" "2850")
# paths=("/home/oshita/vlm/Link-Context-Learning/SHIFT_IMAGE_TOKEN_TAIL_NON_MIX_LCL_2WAY_WEIGHT" "/home/oshita/vlm/Link-Context-Learning/SHIFT_IMAGE_TOKEN_HEAD_NON_MIX_LCL_2WAY_WEIGHT" "/home/oshita/vlm/Link-Context-Learning/NON_MIX_LCL_2WAY_WEIGHT")
# for path in "${paths[@]}"; do
#     for shot in "${shots[@]}"; do
#         for checkpoint in "${checkpoints[@]}"; do
#             if [ "$checkpoint" = "2850" ]; then
#                 if [ $shot -ge 1 ] && [ $shot -le 3 ]; then
#                     accelerate launch --num_processes 8 \
#                         --main_process_port 23786 \
#                         mllm/pipeline/finetune.py \
#                         config/lcl_eval_test100.py \
#                         --cfg-options data_args.use_icl=True \
#                         --cfg-options model_args.model_name_or_path=$path \
#                         --cfg-options data_args.shot=$shot \
#                         --cfg-options training_args.per_device_eval_batch_size=6 \
#                         --cfg-options training_args.output_dir=$path/first_image_token_result/result_shot$shot/2850
#                 elif [ $shot -ge 4 ] && [ $shot -le 8 ]; then
#                     accelerate launch --num_processes 8 \
#                         --main_process_port 23786 \
#                         mllm/pipeline/finetune.py \
#                         config/lcl_eval_test100.py \
#                         --cfg-options data_args.use_icl=True \
#                         --cfg-options model_args.model_name_or_path=$path \
#                         --cfg-options data_args.shot=$shot \
#                         --cfg-options training_args.per_device_eval_batch_size=4 \
#                         --cfg-options training_args.output_dir=$path/first_image_token_result/result_shot$shot/2850
#                 elif [ $shot -ge 9 ] && [ $shot -le 10 ]; then
#                     accelerate launch --num_processes 8 \
#                         --main_process_port 23786 \
#                         mllm/pipeline/finetune.py \
#                         config/lcl_eval_test100.py \
#                         --cfg-options data_args.use_icl=True \
#                         --cfg-options model_args.model_name_or_path=$path \
#                         --cfg-options data_args.shot=$shot \
#                         --cfg-options training_args.per_device_eval_batch_size=2 \
#                         --cfg-options training_args.output_dir=$path/first_image_token_result/result_shot$shot/2850
#                 fi
#             else
#                 if [ $shot -ge 1 ] && [ $shot -le 3 ]; then
#                     accelerate launch --num_processes 8 \
#                         --main_process_port 23786 \
#                         mllm/pipeline/finetune.py \
#                         config/lcl_eval_test100.py \
#                         --cfg-options data_args.use_icl=True \
#                         --cfg-options model_args.model_name_or_path=$path/checkpoint-$checkpoint \
#                         --cfg-options data_args.shot=$shot \
#                         --cfg-options training_args.per_device_eval_batch_size=6 \
#                         --cfg-options training_args.output_dir=$path/first_image_token_result/result_shot$shot/$checkpoint
#                 elif [ $shot -ge 4 ] && [ $shot -le 8 ]; then
#                     accelerate launch --num_processes 8 \
#                         --main_process_port 23786 \
#                         mllm/pipeline/finetune.py \
#                         config/lcl_eval_test100.py \
#                         --cfg-options data_args.use_icl=True \
#                         --cfg-options model_args.model_name_or_path=$path/checkpoint-$checkpoint \
#                         --cfg-options data_args.shot=$shot \
#                         --cfg-options training_args.per_device_eval_batch_size=4 \
#                         --cfg-options training_args.output_dir=$path/first_image_token_result/result_shot$shot/$checkpoint
#                 elif [ $shot -ge 9 ] && [ $shot -le 10 ]; then
#                     accelerate launch --num_processes 8 \
#                         --main_process_port 23786 \
#                         mllm/pipeline/finetune.py \
#                         config/lcl_eval_test100.py \
#                         --cfg-options data_args.use_icl=True \
#                         --cfg-options model_args.model_name_or_path=$path/checkpoint-$checkpoint \
#                         --cfg-options data_args.shot=$shot \
#                         --cfg-options training_args.per_device_eval_batch_size=2 \
#                         --cfg-options training_args.output_dir=$path/first_image_token_result/result_shot$shot/$checkpoint
#                 fi
#             fi
#         done
#     done
# done

# path=/home/oshita/vlm/Link-Context-Learning/LCL_2WAY_WEIGHT
# accelerate launch --num_processes 1 \
#     --main_process_port 23786 \
#     mllm/pipeline/finetune.py \
#     config/lcl_eval_test100.py \
#     --cfg-options data_args.use_icl=True \
#     --cfg-options model_args.model_name_or_path=$path \
#     --cfg-options data_args.shot=1 \
#     --cfg-options training_args.per_device_eval_batch_size=1 \
#     --cfg-options training_args.output_dir=$path/eval_test

path=/home/oshita/vlm/Link-Context-Learning/imagenet1k_classify
accelerate launch --num_processes 8 \
    --main_process_port 23786 \
    mllm/pipeline/finetune.py \
    config/eval_imagenet_classify.py \
    --cfg-options data_args.use_icl=False \
    --cfg-options model_args.model_name_or_path=$path \
    --cfg-options training_args.per_device_eval_batch_size=32 \
    --cfg-options training_args.output_dir=$path/eval_test