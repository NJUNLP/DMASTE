#!/usr/bin/env bash
# sources=("electronics" "beauty" "fashion" "home")
source='all'
targets="electronics beauty fashion home book toy grocery pet"
# source_idx=$1
# model_name=$2
# source=${sources[source_idx]}

for model_name in {0,}
do
    python train.py \
    --data_dir ../ia-dataset/ \
    --log_dir ./log/$source/ \
    --source ${source} \
    --device 0 \
    --mode train \
    --model_dir ./save_models/${source}/ \
    --model_name ${model_name} 
    for target in $targets
    do 
    python train.py \
    --data_dir ../ia-dataset \
    --log_dir ./log/cross-domain/base/$source/${target} \
    --source ${source} \
    --target ${target} \
    --device 0 \
    --mode test \
    --model_dir ./save_models/${source}/ \
    --model_name ${model_name} 
done
done