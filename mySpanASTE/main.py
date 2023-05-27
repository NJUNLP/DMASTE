import os 
import random 
import argparse

import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm 
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter 
import random, os
import numpy as np

from utils.collate import collate_fn
from utils.data_utils import ABSADataset, ABSAProcessor, convert_pad_tensor_to_list, convert_predictions_to_triples
from models.span_aste import SpanModel
from utils.metric import Metric


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='../dataset', 
                    help="the dataset for train")
parser.add_argument("--unlabeled_data_dir", type=str, default='../amazon')
parser.add_argument("--source", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--model_dir", type=str, default="save_models",
                    help="the model.pkl save path")
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument("--batch_size", type=int, default=8, help="number of batch_size")
parser.add_argument("--encoder_lr", type=float, default=5e-5, help="learning rate of adam")
parser.add_argument('--cls_lr', type=float, default=1e-3)
parser.add_argument("--mode", type=str, choices=['train', 'test'])
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument('--reduction', type=str, default='sum', choices=['mean', 'sum'])
parser.add_argument('--seed', type=int)

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(args)



def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf``
    (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
if args.seed is not None:
    print('set seed', args.seed)
    set_seed(args.seed)

def get_dataset(dataset, mode, tokenizer):
    data_dir = os.path.join(args.data_dir, dataset) 
    processor = ABSAProcessor(tokenizer)
    examples = processor.get_examples(data_dir, mode) 
    features = processor.convert_examples_to_features(examples) 
    dataset = ABSADataset(features)
    return examples, dataset 


def evaluate(dataloader, model, examples):
    model.eval()
    all_predictions = []
    metric = Metric()
    for batch_i, batch in enumerate(dataloader):

        input_dict = dict()
        for k in ['input_ids', 'attention_mask', 'spans', 'span_labels', 'span_mask', 'relation_labels', 'seq_length']:
            input_dict[k] = batch[k].to(device)
        output = model(**input_dict)
        batch_example = examples[batch_i * args.batch_size: (batch_i + 1) * args.batch_size]
        all_predictions.extend(metric.compute(batch_example, output, batch))
    model.train()
    return metric.get_metric(), all_predictions

def test(test_dataloader, model, test_examples, mode):
    metric, predictions = evaluate(test_dataloader, model, test_examples)

    print('test metric', metric)
    os.makedirs(os.path.join(args.log_dir, args.model_name), exist_ok=True)
    metric_file = os.path.join(args.log_dir, args.model_name, 'metric.txt')
    with open(metric_file, 'w') as f:
        f.write(str(metric) + '\n')
    predict_file = os.path.join(args.log_dir, args.model_name, 'pred.txt')
    with open(predict_file, 'w') as f:
        for p in predictions:
            f.write(str(p) + '\n')        

def main():
    metric_file = os.path.join(args.log_dir, args.model_name, 'metric.txt')
    if os.path.exists(metric_file):
        print('------------------------------ file exists, return ---------------------------')
        return
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<null-aspect>']})
    tb = SummaryWriter('tb_' + args.log_dir)
    if args.mode == 'train':
        os.makedirs(args.model_dir, exist_ok=True)
        _, train_dataset = get_dataset(args.source, 'train.txt', tokenizer)
        dev_examples, dev_dataset = get_dataset(args.source, 'dev.txt', tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
        print('num train data', len(train_dataset), 'num dev data', len(dev_dataset))
        bert = BertModel.from_pretrained('bert-base-uncased')
        bert.resize_token_embeddings(len(tokenizer))
        model = SpanModel(bert).to(device)
        optimizer = AdamW([{'params': model.encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': 1e-2}, 
                          {'params': list(set(model.parameters()) - set(model.encoder.parameters())), 'lr': args.cls_lr, 'weight_decay': 0}])
        scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=int(args.n_epochs * len(train_dataloader) * 0.1), 
                    num_training_steps=args.n_epochs * len(train_dataloader))
        total_steps = args.n_epochs * len(train_dataloader)
        best_metric = None
        num_steps = 0
        with tqdm(total=len(train_dataloader)) as pbar:        
            for epoch in range(args.n_epochs):
                model.train()
                pbar.reset()
                for batch in train_dataloader:
                    pass
                    num_steps += 1
                    input_dict = dict()
                    for k in ['input_ids', 'attention_mask', 'spans', 'span_labels', 'span_mask', 'relation_labels', 'seq_length']:
                        input_dict[k] = batch[k].to(device)
                    output = model(**input_dict)
                    loss = output['loss']
                    if num_steps % int(total_steps / 300) == 0:
                        tb.add_scalar('loss', loss.item(), global_step=num_steps)
                        tb.add_scalar('ner loss', output['ner_loss'].item(), global_step=num_steps)
                        tb.add_scalar('relation loss', output['relation_loss'].item(), global_step=num_steps)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    pbar.update(1)
                    pbar.set_postfix(epoch=f'{epoch + 1}/{args.n_epochs}', loss=loss.item(), best_f1=f"{round(best_metric['triplet']['f1'] * 100, 2)}" if best_metric is not None else 'none')
                metric, _ = evaluate(dev_dataloader, model, dev_examples)
                for name in metric:
                    for k in metric[name]:
                        tb.add_scalar(f'{name}_{k}', metric[name][k], global_step=num_steps)
                if best_metric is None or best_metric['triplet']['f1'] < metric['triplet']['f1']:
                    best_metric = metric
                    torch.save(model, os.path.join(args.model_dir, args.model_name + '.pt'))
            tb.add_hparams(hparam_dict=vars(args), metric_dict=best_metric['triplet'])
        # torch.save(model, os.path.join(args.model_dir, args.model_name + '.pt'))

    else:
        model = torch.load(os.path.join(args.model_dir, args.model_name + '.pt'))
        test_examples, test_dataset = get_dataset(args.target, 'test.txt', tokenizer)
        print('num test data', len(test_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        test(test_dataloader, model, test_examples, 'test')
        
        # dev_examples, dev_dataset = get_dataset(args.target, 'dev.txt', tokenizer)
        # print('num dev data', len(dev_dataset))
        # dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        # test(dev_dataloader, model, dev_examples, 'dev')
    os.makedirs(args.log_dir, exist_ok=True)
    param_file = os.path.join(args.log_dir, args.model_name + '_params.txt')
    with open(param_file, 'w') as f:
        f.write(str(args) + '\n')

if __name__ == '__main__':
    main()
    
    