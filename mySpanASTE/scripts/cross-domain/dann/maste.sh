#!/usr/bin/env bash
# source=$1
# target=$2
sources="electronics fashion home beauty"
targets="toy pet grocery book"
source=$1
ad_steps=$2
model_name=0
device=0

# for source in $sources
# do 
    for target in $targets 
    do 
    # CUDA_VISIBLE_DEVICES=$device \
    python3 DANN_main.py \
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

    # CUDA_VISIBLE_DEVICES=$device 
    python3 DANN_main.py \
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

    # rm save_models/ad_steps${ad_steps}/${source}/${target}/${model_name}.pt 
done 
# done