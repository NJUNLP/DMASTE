#!/usr/bin/env bash
# sources=("electronics" "beauty" "fashion" "home")

targets="book toy grocery pet"
source=$1
model_name=$2
device=$3
    # CUDA_VISIBLE_DEVICES=$device \
    launch -- python3 train.py \
    --data_dir ../ia-multi-dataset/ \
    --log_dir ./log/$source/ \
    --source ${source} \
    --device 0 \
    --mode train \
    --model_dir ./save_models/${source}/ \
    --model_name ${model_name} 
    for target in $targets
    do 
    # CUDA_VISIBLE_DEVICES=$device \
    launch -- python3 train.py \
    --data_dir ../ia-multi-dataset \
    --log_dir ./log/cross-domain/multi-base/$source/${target} \
    --source ${source} \
    --target ${target} \
    --device 0 \
    --mode test \
    --model_dir ./save_models/${source}/ \
    --model_name ${model_name} 
done