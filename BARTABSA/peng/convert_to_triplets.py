from transformers import AutoTokenizer
import json 
import numpy as np 

def init_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    unique_no_split_tokens = tokenizer.unique_no_split_tokens
    tokenizer.unique_no_split_tokens = unique_no_split_tokens + ['[ia]']
    tokenizer.add_tokens(['[ia]'])
    mapping = {  # so that the label word can be initialized in a better embedding.
        'POS': '<<positive>>',
        'NEG': '<<negative>>',
        'NEU': '<<neutral>>'
    }

    cur_num_tokens = tokenizer.vocab_size
    cur_num_token = cur_num_tokens

    tokens_to_add = sorted(list(mapping.values()), key=lambda x:len(x), reverse=True)
    unique_no_split_tokens = tokenizer.unique_no_split_tokens
    sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
    for tok in sorted_add_tokens:
        assert tokenizer.convert_tokens_to_ids([tok])[0]==tokenizer.unk_token_id
    tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
    tokenizer.add_tokens(sorted_add_tokens)
     
    mapping2id = {}
    mapping2targetid = []

    for key, value in mapping.items():
        mapping2targetid.append(key)
    return tokenizer

def convert_span_to_idx(tokenizer, sent, spans):
    mapping2targetid = []
    mapping = {  # so that the label word can be initialized in a better embedding.
        'POS': '<<positive>>',
        'NEG': '<<negative>>',
        'NEU': '<<neutral>>'
    }
    for key, value in mapping.items():
        mapping2targetid.append(key)
    triplets = []
    raw_words = sent.split()
    word_bpes = [[tokenizer.bos_token_id]]
    for word in raw_words:
        bpes = tokenizer.tokenize(word, add_prefix_space=True)
        bpes = tokenizer.convert_tokens_to_ids(bpes)
        word_bpes.append(bpes)
    word_bpes.append([tokenizer.eos_token_id])

    lens = list(map(len, word_bpes))
    cum_lens = np.cumsum(list(lens)).tolist()
    def subword2word(subword_idx, cum_lens):
        for i in range(len(cum_lens)):
            if i < len(cum_lens) and cum_lens[i] <= subword_idx < cum_lens[i + 1]:
                return i 
        return len(cum_lens) - 1 
    for span in spans:
        target_shift = 5 # pos, neg, neu, <sos>, <eos>
        new_spans = [subword2word(i - target_shift, cum_lens) for i in span[:4]]
        a, o = new_spans[:2], new_spans[2:]
        if not 0 <= span[-1] -2 < len(mapping2targetid):
            print('invalid span', span)
            continue
        s = mapping2targetid[span[-1] - 2]
        a = [a[0]] if a[0] == a[-1] else a 
        o = [o[0]] if o[0] == o[-1] else o 
        triplets.append((a, o, s))
    return triplets

def convert(tokenizer, examples, all_pred_spans):
    tp = 0 
    pred_num = 0 
    gold_num = 0
    all_pred_triplets = []
    for example, pred_spans in zip(examples, all_pred_spans):
        func = lambda triplets: set([' '.join([str(a), str(o), s]) for a, o, s in triplets])
        sent, golden_spans = example 
        golden_triplets = func(golden_spans)
        # print('golden triplets', golden_triplets)
        # print('pred spans', pred_spans)
        origin_pred_triplets = convert_span_to_idx(tokenizer, sent, pred_spans)
        pred_triplets = func(origin_pred_triplets)
        all_pred_triplets.append(origin_pred_triplets)
    #     print('pred triplets', pred_triplets)
    #     print()
    #     tp += len(pred_triplets & golden_triplets)
    #     pred_num += len(pred_triplets)
    #     gold_num += len(golden_triplets)
    # precision = tp / pred_num if pred_num != 0 else 0 
    # recall = tp / gold_num if gold_num != 0 else 0 
    # f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0 
    # print('precision', precision, 'recall', recall, 'f1', f1)
    return all_pred_triplets
        

def main():
    all_pred_spans = []
    examples = []
    tokenizer = init_tokenizer()
    with open('../../ia-dataset/fashion/test.txt') as f:
        for line in f:
            sent, triplets = line.split('####')
            triplets = eval(triplets)
            examples.append([sent, triplets])
    with open('tmp.txt') as f:
        for line in f:
            all_pred_spans.append(eval(line))
    convert(tokenizer, examples, all_pred_spans)
if __name__ == '__main__':
    main()