#!/usr/bin/env bash
sources=("14res" "14lap" "15res" "16res")
targets="14res 15res 16res 14lap"
source_idx=$1
source=${sources[source_idx]}
for model_name in {0..4}
do
    python3 train.py \
    --data_dir ../ia-dataset/ \
    --log_dir ./log/$source/ \
    --source ${source} \
    --target ${source} \
    --device 0 \
    --mode train \
    --model_dir ./save_models/${source}/ \
    --model_name ${model_name} 
    
    for target in ${targets}
    do
    python3 train.py \
    --data_dir ../ia-dataset \
    --log_dir ./log/base/$source/${target} \
    --source ${source} \
    --target ${target} \
    --device 0 \
    --mode test \
    --model_dir ./save_models/${source}/ \
    --model_name ${model_name} 
    done
done