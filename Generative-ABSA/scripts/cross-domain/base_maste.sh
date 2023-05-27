#!/usr/bin/env bash
# sources=("electronics" "beauty" "fashion" "home")
# targets="electronics beauty fashion home book toy grocery pet"
# # source_idx=$1
# source=$1
# target=$2
source='all'
targets="electronics beauty fashion home book toy grocery pet"

# model_name=$3
# source=${sources[source_idx]}

for model_name in {0..4}
do
    python main.py --task aste \
              --dataset ../ia-dataset/ \
              --source ${source} \
              --seed ${model_name} \
              --model_name ${model_name} \
              --log_dir log/cross-domain/base/${source}/ \
              --model_dir save_models/in-domain/${source} \
              --tmp_dir tmp/in-domain/${source} \
              --model_name_or_path t5-base \
              --paradigm extraction \
              --n_gpu 0 \
              --do_train \
              --train_batch_size 16 \
              --gradient_accumulation_steps 2 \
              --eval_batch_size 16 \
              --learning_rate 3e-4 \
              --num_train_epochs 20
    for target in ${targets}
    do
        python main.py --task aste \
              --dataset ../ia-dataset/ \
              --source ${source} \
              --target ${target} \
              --seed ${model_name} \
              --model_name ${model_name} \
              --log_dir log/cross-domain/base/${source}/${target} \
              --model_dir save_models/in-domain/${source} \
              --tmp_dir tmp/in-domain/${source} \
              --model_name_or_path t5-base \
              --paradigm extraction \
              --n_gpu 0 \
              --do_eval \
              --train_batch_size 16 \
              --gradient_accumulation_steps 2 \
              --eval_batch_size 16 \
              --learning_rate 3e-4 \
              --num_train_epochs 20
    done
done
# done 