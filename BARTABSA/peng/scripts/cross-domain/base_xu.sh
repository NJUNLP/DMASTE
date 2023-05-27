sources=("14res" "14lap" "15res" "16res")
targets='14res 14lap 15res 16res'
source_idx=$1
for model_name in {0..4}
do
    source=${sources[source_idx]}
    # target=${targets[source_idx]}
    python3 train.py --log_dir ./log/in-domain/$source/ \
    --mode train \
    --source ${source} \
    --batch_size 32 \
    --save_model_dir ./save_models/${source}/ \
    --model_name ${model_name} 

    for target in $targets
    do
      python3 train.py --log_dir ./log/cross-domain/base/$source/${target} \
        --mode test \
        --source ${source} \
        --target ${target} \
        --batch_size 32 \
        --save_model_dir ./save_models/${source}/ \
        --model_name ${model_name} 
    done
done
