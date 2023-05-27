sources=("res14" "lap14" "res15" "res16")
targets='res14 lap14 res15 res16'
source_idx=$1
for model_name in {0..4}
do
    source=${sources[source_idx]}
    
    mkdir -p log/in-domain/$source
    mkdir -p save_models/${source}/${model_name}
    
    python3 main.py \
    --mode train \
    --task triplet \
    --prefix ../../data/ASTE_DATA_V2/ \
    --source ${source} \
    --model_dir ./save_models/${source}/${model_name}/  > ./log/in-domain/$source/${model_name}.txt

    for target in $targets
    do
    mkdir -p log/in-domain/$source/${target}
      python3 main.py \
        --mode test \
        --task triplet \
        --prefix ../../data/ASTE_DATA_V2/ \
        --source ${source} \
        --target ${target} \
        --model_dir ./save_models/${source}/${model_name}/  > ./log/in-domain/$source/${target}/${model_name}.txt
    done
done
