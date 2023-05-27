sources=("electronics" "beauty" "fashion" "home" "14res" "15res" "16res" "14lap" "all")
source_idx=$1
model_name=$2
source=${sources[source_idx]}
# for model_name in {4,3,2,1}
# do
    python main.py --log_dir log/in-domain/${source} \
    --tmp_log tmp_log/${source} \
    --source ${source} \
    --target ${source} \
    --mode train \
    --batch_size 2 \
    --model_dir save_models/${source} \
    --model_name ${model_name} 

    python main.py --log_dir log/in-domain/${source} \
    --tmp_log tmp_log/${source} \
    --source ${source} \
    --target ${source} \
    --mode test \
    --batch_size 2 \
    --model_dir save_models/${source} \
    --model_name ${model_name} 
# done
