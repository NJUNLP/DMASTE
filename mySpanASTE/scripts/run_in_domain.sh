sources=("electronics" "beauty" "fashion" "home" "all")
source_idx=$1
source=${sources[source_idx]}
for model_name in {0..4}
do
    # python3 main.py \
    # --source ${source} \
    # --target ${source} \
    # --model_dir save_models/${source} \
    # --mode train \
    # --log_dir log/in-domain/${source} \
    # --model_name ${model_name} \
    # --batch_size 1 \
    # --n_epochs 20 \
    # --seed ${model_name} 

    # CUDA_VISIBLE_DEVICES=$device 
    python3 main.py \
    --source ${source} \
    --target ${source} \
    --model_dir save_models/${source} \
    --mode test \
    --log_dir log/in-domain/${source} \
    --model_name ${model_name} \
    --batch_size 1 \
    --n_epochs 20 \
    --seed ${model_name} 

done
