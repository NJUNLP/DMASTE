#!/usr/bin/env bash
source=$1
target=$2
ad_steps=$3
model_name=$4
device=$5

    CUDA_VISIBLE_DEVICES=$device python3 DANN_main.py --log_dir log/cross-domain/dann/ad_steps${ad_steps}/${source}/${target} \
    --tmp_log tmp_log/${source}/${target} \
    --source ${source} \
    --target ${target} \
    --mode train \
    --batch_size 2 \
    --model_dir save_models/ad_steps${ad_steps}/${source}/${target} \
    --model_name ${model_name} \
    --ad_steps ${ad_steps}

    CUDA_VISIBLE_DEVICES=$device python3 DANN_main.py --log_dir log/cross-domain/dann/ad_steps${ad_steps}/${source}/${target} \
    --tmp_log tmp_log/${source}/${target} \
    --source ${source} \
    --target ${target} \
    --mode test \
    --batch_size 2 \
    --model_dir save_models/ad_steps${ad_steps}/${source}/${target} \
    --model_name ${model_name} \
    --ad_steps ${ad_steps}

    rm save_models/ad_steps${ad_steps}/${source}/${target}/${model_name}.pt 
