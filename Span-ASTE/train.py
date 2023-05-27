import sys 
sys.path.append('aste')
from wrapper import SpanModel 
import argparse
import os 

def main():
    parser = argparse.ArgumentParser(description='Bidirectional MRC-based sentiment triplet extraction')
    parser.add_argument('--data_dir', type=str, default="../dataset/")
    parser.add_argument('--log_dir', type=str, default="./log/")
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--model_dir', type=str, default="./model/")
    parser.add_argument('--model_name', type=str, default="1")

    args = parser.parse_args()
    if args.mode == 'train':
        os.makedirs(os.path.join(args.model_dir, args.model_name), exist_ok=True)
        source = os.path.join(args.data_dir, args.source)
        model = SpanModel(save_dir=os.path.join(args.model_dir, args.model_name), random_seed=int(args.model_name))#, path_config_base=f"training_config/config{args.device}.jsonnet")
        model.fit(f'{source}/train.txt', f'{source}/dev.txt', random_seed=int(args.model_name))
    else:
        os.makedirs(f'{args.log_dir}/{args.model_name}', exist_ok=True)
        model = SpanModel(save_dir=os.path.join(args.model_dir, args.model_name), random_seed=int(args.model_name))#, path_config_base=f"training_config/config{args.device}.jsonnet")
        log_dir = args.log_dir
        pred_file = f'{log_dir}/{args.model_name}/pred.txt'
        target = os.path.join(args.data_dir, args.target)
        model.predict(f'{target}/test.txt', pred_file, device=args.device)
        results = model.score(f'{target}/test.txt', pred_file)
        with open(f'{log_dir}/{args.model_name}/metric.txt', 'w') as f:
            f.write(str(results) + '\n')
main()