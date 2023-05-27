#!/usr/bin/env bash
sources=("14res" "14lap" "15res" "16res")
targets='14res 14lap 15res 16res'
source_idx=$1
# for source_idx in {0,1}
# do
source=${sources[source_idx]}
# target=${targets[source_idx]}
for model_name in {0..4}
do
    python main.py \
    --log_dir ./log/$source/ \
    --tmp_log tmp_log/${source} \
    --source ${source} \
    --target ${source} \
    --mode train \
    --batch_size 2 \
    --model_dir ./save_models/${source}/ \
    --model_name ${model_name} 
    for target in ${targets}
    do
    python main.py --log_dir log/cross-domain/base/${source}/${target} \
    --tmp_log tmp_log/${source} \
    --source ${source} \
    --target ${target} \
    --mode test \
    --batch_size 2 \
    --model_dir save_models/${source} \
    --model_name ${model_name} 
    done
done
# done