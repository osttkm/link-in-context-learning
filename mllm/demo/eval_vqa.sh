#!/bin/bash
num_gpus=4
num_jobs=0
pids=()
model_paths=(
  "/home/oshita/vlm/Link-Context-Learning/model_result/LCL_VI_20_pretrained_epoch1_ac_30" "/home/oshita/vlm/Link-Context-Learning/model_result/LCL_VI_20_pretrained_epoch2_ac_30" "/home/oshita/vlm/Link-Context-Learning/model_result/LCL_VI_20_pretrained_epoch3_ac_30"
  "/home/oshita/vlm/Link-Context-Learning/model_result/shikra2LCL_VI_20_pretrained_epoch1_ac_30" "/home/oshita/vlm/Link-Context-Learning/model_result/shikra2LCL_VI_20_pretrained_epoch2_ac_30" /home/oshita/vlm/Link-Context-Learning/model_result/shikra2LCL_VI_20_pretrained_epoch3_ac_30
  "/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch1_ac_20" "/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch1_ac_30"
  "/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch2_ac_20" "/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch2_ac_30"
  "/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch3_ac_20" "/home/oshita/vlm/Link-Context-Learning/LCL_VI+AC_20_pretrained_epoch3_ac_30"
  "/home/oshita/vlm/Link-Context-Learning/shikra2LCL_VI+AC_20_pretrained_epoch1_ac_20" "/home/oshita/vlm/Link-Context-Learning/shikra2LCL_VI+AC_20_pretrained_epoch2_ac_20" "/home/oshita/vlm/Link-Context-Learning/shikra2LCL_VI+AC_20_pretrained_epoch3_ac_20"
  )

for index in "${!model_paths[@]}"; do
  model_path="${model_paths[$index]}"
  json_path="${json_paths[$index]}"

  gpu_id=$((num_jobs % num_gpus))
  # echo "GPU ID: $gpu_id"
  # pidにはプロセス番号が格納される
  CUDA_VISIBLE_DEVICES=$gpu_id python demo_vqa.py --model_path "$model_path" --json_path "$json_path" & pids+=($!)

  ((num_jobs++))
  if ((num_jobs >= num_gpus)); then
      for pid in "${pids[@]}"; do
          wait $pid
      done
      num_jobs=0
      pids=()
  fi
done

