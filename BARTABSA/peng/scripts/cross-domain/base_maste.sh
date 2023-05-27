# sources="electronics beauty fashion home"
# targets="electronics beauty fashion home book toy grocery pet"
# source_idx=$1
source='all'
targets="electronics beauty fashion home book toy grocery pet"

# for source in $sources
# do 
    for model_name in {0..4}
    do
        python3 train.py --log_dir ./log/cross-domain/$source/ \
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
# done
