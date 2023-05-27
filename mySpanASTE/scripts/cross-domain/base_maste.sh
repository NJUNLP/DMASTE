#!/usr/bin/env bash
sources=("electronics" "beauty" "fashion" "home" "all")
targets="electronics beauty fashion home book toy grocery pet"
# source_idx=$1
model_name=0
source=${sources[source_idx]}

# for model_name in {0..4}
# do
for source in $targets
do
    python3 main.py \
    --source ${source} \
    --target ${source} \
    --model_dir save_models/${source} \
    --mode train \
    --log_dir log/cross-domain/base/${source} \
    --model_name ${model_name} \
    --batch_size 1 \
    --n_epochs 20 \
    --seed ${model_name} 
    for target in $targets 
    do
    # CUDA_VISIBLE_DEVICES=$device 
    python3 main.py \
    --source ${source} \
    --target ${target} \
    --model_dir save_models/${source} \
    --mode test \
    --log_dir log/cross-domain/base/${source}/${target} \
    --model_name ${model_name} \
    --batch_size 1 \
    --n_epochs 20 \
    --seed ${model_name} 
    done
done
