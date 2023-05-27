#!/usr/bin/env bash
source=$1
target=$2
ad_steps=$3
model_name=$4
device=$5

    CUDA_VISIBLE_DEVICES=$device \
    python3 DANN_main.py \
    --data_dir ../eq-dataset \
    --source ${source} \
    --target ${target} \
    --model_dir save_models/ad_steps${ad_steps}/${source}/${target} \
    --mode train \
    --log_dir log/cross-domain/dann/ad_steps${ad_steps}/${source}/${target} \
    --model_name ${model_name} \
    --batch_size 1 \
    --n_epochs 30 \
    --seed ${model_name} \
    --ad_steps ${ad_steps} 

    CUDA_VISIBLE_DEVICES=$device 
    python3 DANN_main.py \
    --data_dir ../eq-dataset \
    --source ${source} \
    --target ${target} \
    --model_dir save_models/ad_steps${ad_steps}/${source}/${target} \
    --mode test \
    --log_dir log/cross-domain/dann/ad_steps${ad_steps}/${source}/${target} \
    --model_name ${model_name} \
    --batch_size 1 \
    --n_epochs 20 \
    --seed ${model_name} \
    --ad_steps ${ad_steps} 

    rm save_models/ad_steps${ad_steps}/${source}/${target}/${model_name}.pt 
