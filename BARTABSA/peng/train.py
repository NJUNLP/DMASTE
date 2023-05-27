import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
warnings.filterwarnings('ignore')
from data.pipe import BartBPEABSAPipe
from peng.model.bart_absa import BartSeq2SeqModel

from fastNLP import Trainer, Tester
from peng.model.metrics import Seq2SeqSpanMetric
from peng.model.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback
from fastNLP import FitlogCallback
from fastNLP.core.sampler import SortedSampler
from peng.model.generator import SequenceGeneratorModel
from peng.convert_to_triplets import convert
import fitlog

# fitlog.debug()
os.makedirs('logs', exist_ok=True)
fitlog.set_log_dir('logs')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str)
parser.add_argument('--target', type=str)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--save_model_dir', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--save_model', type=int, default=0)
parser.add_argument('--mode', type=str, choices=['train', 'test'])
parser.add_argument('--log_dir', type=str)

args= parser.parse_args()

lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams
source = args.source 
target = args.target
opinion_first = args.opinion_first
length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
use_encoder_mlp = args.use_encoder_mlp
save_model = args.save_model
fitlog.add_hyper(args)

#######hyper
#######hyper




# @cache_results(cache_fn, _refresh=False)
def get_data(dataset_name):
    demo=False
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{opinion_first}.pt"
    @cache_results(cache_fn, _refresh=False)
    def func():
        pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
        data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo)
        return data_bundle, pipe.tokenizer, pipe.mapping2id
    return func()
    

source_data_bundle, tokenizer, mapping2id = get_data(source)

max_len = 10
max_len_a = 1.5

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0  #
eos_token_id = 1  #
label_ids = list(mapping2id.values())
model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                     copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)
vocab_size = len(tokenizer)
print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                               restricter=None)

import torch
if torch.cuda.is_available():
    # device = list([i for i in range(torch.cuda.device_count())])
    device = 'cuda'
else:
    device = 'cpu'

parameters = []
params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
parameters.append(params)

params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr':lr, 'weight_decay':0}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

optimizer = optim.AdamW(parameters)


callbacks = []
callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
callbacks.append(FitlogCallback())

sampler = None
# sampler = ConstTokenNumSampler('src_seq_len', max_token=1000)
sampler = BucketSampler(seq_len_field_name='src_seq_len')
metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first)


model_path = None
if save_model:
    model_path = 'save_models/'

if args.mode == 'train':
    trainer = Trainer(train_data=source_data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                  loss=Seq2SeqLoss(),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=2, n_epochs=n_epochs, print_every=1,
                  dev_data=source_data_bundle.get_dataset('dev'), metrics=metric, metric_key='triple_f',
                  validate_every=-1, save_path=model_path, use_tqdm=True, device=device,
                  callbacks=callbacks, check_code_level=0, test_use_tqdm=False,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size)

    trainer.train(load_best_model=True)
    os.makedirs(args.save_model_dir, exist_ok=True)
    torch.save(model, f'{args.save_model_dir}/{args.model_name}.pt')
elif args.mode == 'test':
    target_data_bundle, _, _ = get_data(target)
    model = torch.load(f'{args.save_model_dir}/{args.model_name}.pt')
    tester = Tester(data=target_data_bundle.get_dataset('test'), model=model, metrics=metric, batch_size=args.batch_size,
                num_workers=2, device=device, use_tqdm=True, callbacks=callbacks)
    res = tester.test()
    fitlog.add_best_metric(value=res, name='test')
    os.makedirs(os.path.join(args.log_dir, args.model_name), exist_ok=True)
    log_file = f'{args.log_dir}/{args.model_name}/metric.txt'
    with open(log_file, 'w') as f:
        import json 
        f.write(json.dumps(res) + '\n')
    pred = metric.get_pred()
    examples = []
    with open(f'../../ia-dataset/{target}/test.txt') as f:
        for line in f:
            sent, triplets = line.split('####')
            triplets = eval(triplets)
            examples.append([sent, triplets])
    pred = convert(tokenizer, examples, pred)
    with open(f'{args.log_dir}/{args.model_name}/pred.txt', 'w') as f:
        for ts in pred:
            f.write(str(ts) + '\n')



