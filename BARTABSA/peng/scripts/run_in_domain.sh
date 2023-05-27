sources=("electronics" "beauty" "fashion" "home" "14res" "15res" "16res" "14lap")
source_idx=$1
for model_name in {0..4}
do
    source=${sources[source_idx]}
    # python3 train.py --log_dir ./log/in-domain/$source/ \
    # --mode train \
    # --source ${source} \
    # --batch_size 32 \
    # --save_model_dir ./save_models/${source}/ \
    # --model_name ${model_name} 


      python3 train.py --log_dir ./log/in-domain/$source \
        --mode test \
        --source ${source} \
        --target ${source} \
        --batch_size 32 \
        --save_model_dir ./save_models/${source}/ \
        --model_name ${model_name} 
done
