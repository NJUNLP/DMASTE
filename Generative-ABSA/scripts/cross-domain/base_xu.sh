#!/usr/bin/env bash
sources=("14res" "14lap")
targets=("14lap" "14res")
source_idx=$1
source=${sources[source_idx]}
target=${targets[source_idx]}
for model_name in {0..4}
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