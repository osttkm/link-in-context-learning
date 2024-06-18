#!/bin/bash
# num_gpus=8
# num_jobs=0
# pids=()
model_paths=(
  "/home/oshita/vlm/Link-Context-Learning/SHUFFLE_LCL_VI_20_Adapt_LossWeight"
  )

# json_paths=(
#   "/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/old_one/LCL_VI_20.json" "/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/old_one/LCL_VI_20.json" 
#   "/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/old_one/LCL_VI_20.json" "/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/old_one/LCL_VI_20.json" 
#   "/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/old_one/LCL_VI_20.json" "/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/old_one/LCL_VI_20.json"
#   "/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/old_one/LCL_VI_20.json" "/home/oshita/vlm/Link-Context-Learning/config/_base_/dataset/template/old_one/LCL_VI_20.json"  
#   )


# for index in "${!model_paths[@]}"; do
#   model_path="${model_paths[$index]}"
#   json_path="${json_paths[$index]}"

#   gpu_id=$((num_jobs % num_gpus))
#   # echo "GPU ID: $gpu_id"
#   # pidにはプロセス番号が格納される
#   CUDA_VISIBLE_DEVICES=$gpu_id python demo_icl_local.py --model_path "$model_path" --json_path "$json_path" & pids+=($!)

#   ((num_jobs++))
#   if ((num_jobs >= num_gpus)); then
#       for pid in "${pids[@]}"; do
#           wait $pid
#       done
#       num_jobs=0
#       pids=()
#   fi
# done


for model_path in "${model_paths[@]}"; do
  python utils/analyze_txt.py --txt_path "$model_path"
done

for model_path in "${model_paths[@]}"; do
  python utils/culc_f1score.py --txt_path "$model_path"
done

# python utils/get_dataset_f1score.py --txt_path "${model_paths[@]}"
# python utils/get_dataset_f1score.py
