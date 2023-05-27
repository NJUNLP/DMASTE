#!/usr/bin/env bash
# sources=("electronics" "beauty" "fashion" "home" "14res" "15res" "16res" "14lap")
sources="beauty fashion home 14res 15res 16res 14lap"
targets="book pet toy grocery"
source_idx=$1
# model_name=$2
# source=${sources[source_idx]}
for source in $sources 
do 
for model_name in {0..4}
do
    # python train.py \
    # --data_dir ../ia-dataset/ \
    # --log_dir ./log_v1/in-domain/$source/ \
    # --source ${source} \
    # --target ${source} \
    # --device 0 \
    # --mode train \
    # --model_dir ./save_models/${source}/ \
    # --model_name ${model_name} 
    for target in $targets
do
    python train.py \
    --data_dir ../ia-dataset \
    --log_dir ./log_v1/in-domain/$source/$target \
    --source ${source} \
    --target ${target} \
    --device 0 \
    --mode test \
    --model_dir ./save_models/${source}/ \
    --model_name ${model_name} 
done
done
done
