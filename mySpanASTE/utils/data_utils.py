import os 
from enum import IntEnum 
from pydantic import BaseModel
from typing import List
from torch.utils.data import Dataset 
import torch 

class SpanLabel(IntEnum):
    INVALID = 0
    ASPECT = 1
    OPINION = 2


class RelationLabel(IntEnum):
    INVALID = 0
    POS = 1
    NEG = 2
    NEU = 3

class ABSADataset(Dataset):
    def __init__(self, features):
        self.features = features 
    
    def __getitem__(self, index):
        return self.features[index]
    
    def __len__(self):
        return len(self.features)

class SentimentTriple(BaseModel):
    aspects: List
    opinions: List 
    triples: List

    @classmethod
    def from_sentiment_triple(cls, triples, token_range):
        """read from sentiment triple"""
        sentiment_map = {'POS': RelationLabel.POS, 'NEG': RelationLabel.NEG, 'NEU': RelationLabel.NEU}
        aspects, opinions, new_triples = [], [], []
        for a, o, s in triples:
            new_a, new_o = None, None 
            if a[1] < len(token_range):
                if -1 in a:
                    new_a = (1, 1)
                else:
                    new_a = (token_range[a[0]][0], token_range[a[1]][1]) 
                aspects.append(new_a)
            if o[1] < len(token_range):
                assert -1 not in o 
                new_o = (token_range[o[0]][0], token_range[o[1]][1])
                opinions.append(new_o)
            if new_a is not None and new_o is not None:
                new_triples.append((new_a, new_o, sentiment_map[s]))
        return cls(
            aspects=aspects,
            opinions=opinions,
            triples=new_triples,
        )



class ABSAFeature:
    def __init__(self, input_ids, spans, span_labels, triples, token_range, seq_length) -> None:
        self.input_ids = input_ids
        self.spans = spans
        self.span_labels = span_labels 
        # self.relation_labels = relation_labels
        self.seq_length = seq_length
        self.token_range = token_range 
        self.triples = triples 

class ABSAProcessor:
    def __init__(self, tokenizer, min_span_width=1, max_span_width=10, max_seq_length=512):
        self.tokenizer = tokenizer 
        self.null_aspect_id = self.tokenizer.convert_tokens_to_ids('<null-aspect>')
        self.min_span_width = min_span_width 
        self.max_span_width = max_span_width 
        self.max_seq_length = max_seq_length 
    def get_features(self, data_dir, mode):
        examples = self.get_examples(data_dir, mode) 
        features = self.convert_examples_to_features(examples)
        return features 

    def get_examples(self, data_dir, mode):
        file_name = os.path.join(data_dir, mode)
        instances = []
        lines = []
        with open(file_name) as f:
            lines = f.readlines()
            lines = [x.split('####') for x in lines]
        for line in lines:
            sentence, triples,  = line[:2]
            triples = eval(triples)
            new_triples = []
            for t in triples:
                a, o, s = t 
                a = [a[0], a[-1]]
                o = [o[0], o[-1]]
                assert len(a) == 2 and len(o) == 2 and s in ('POS', 'NEG', 'NEU')
                assert a[0] <= a[1]
                assert o[0] <= o[1]
                new_triples.append((a, o, s))
            instances.append((sentence, new_triples))
        return instances 

    def convert_examples_to_features(self, examples):
        features = []
        for sent, triples in examples:
            input_ids, token_range = self._tokenize(sent)
            seq_length = len(sent.split())
            triples = SentimentTriple.from_sentiment_triple(triples, token_range)
            spans = self._enumerate_spans(token_range)
            span_labels = [SpanLabel.INVALID] * len(spans) 
            for a in triples.aspects:
                # print(a)
                if a[-1] - a[0] > self.max_span_width:
                    continue
                idx = spans.index(a)
                span_labels[idx] = SpanLabel.ASPECT 
            for o in triples.opinions:
                if o[-1] - o[0] > self.max_span_width:
                    continue
                idx = spans.index(o)
                span_labels[idx] = SpanLabel.OPINION 
            # for a, o, s in triples.triples:
            #     idx_a, idx_o = spans.index(a), spans.index(o)
            #     relation_labels[idx_a][idx_o] = s 
            features.append(ABSAFeature(input_ids=input_ids,
                            spans=spans,
                            span_labels=span_labels,
                            triples = triples.triples, 
                            # relation_labels=relation_labels,
                            seq_length=seq_length, 
                            token_range=token_range))
        return features 
            
    def _enumerate_spans(self, token_range):
        word_length = len(token_range)
        spans = [(1, 1)]
        for i in range(word_length):
            for j in range(self.min_span_width - 1, self.max_span_width):
                if i + j < word_length:
                    start = token_range[i][0] 
                    end = token_range[i + j][1]
                    spans.append((start, end))

        return spans 

    def _tokenize(self, sentence):
        words = sentence.split()
        input_ids = [self.tokenizer.cls_token_id, self.null_aspect_id]
        token_range = []
        start_ids = 2
        for word in words:
            word_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
            if len(input_ids) + len(word_ids) + 1 > self.max_seq_length:
                break 
            input_ids.extend(word_ids)
            token_range.append([start_ids, start_ids + len(word_ids) - 1])
            start_ids += len(word_ids)
        input_ids.append(self.tokenizer.sep_token_id)
        return input_ids, token_range
    
def convert_predictions_to_triples(spans_a, spans_o, relation_labels, token_range):
    # relation_idx = [i for i, label in enumerate(relations_labels) if label != RelationLabel.INVALID]
    # relations_labels = [relations_labels[i] for i in relation_idx]
    relation_indices = [(i, j) for i in range(len(relation_labels)) for j in range(len(relation_labels)) if relation_labels[i][j] != RelationLabel.INVALID]
    # print('relation indices', relation_indices)
    def subword_span2_word_span(subword_span, token_range):
        if 1 in subword_span:
            return [-1, -1]
        start, end = -1, -1
        for i, ran in enumerate(token_range):
            if ran[0] <= subword_span[0] <= ran[1]:
                assert start == -1
                start = i 
            if ran[0] <= subword_span[1] <= ran[1]:
                assert end == -1 
                end = i 
        return [start, end]
    triples = []
    int2sentiment = {RelationLabel.POS: 'POS', RelationLabel.NEG: 'NEG', RelationLabel.NEU: 'NEU'}

    for i, (a_idx, o_idx) in enumerate(relation_indices):
        # assert span_labels[a_idx] == SpanLabel.ASPECT, span_labels[a_idx]
        # assert span_labels[o_idx] == SpanLabel.OPINION, span_labels[o_idx]
        a_subword_span, o_subword_span = spans_a[a_idx], spans_o[o_idx]
        a_word_span = subword_span2_word_span(a_subword_span, token_range)
        o_word_span = subword_span2_word_span(o_subword_span, token_range)
        # print('idx', a_idx, o_idx)
        triples.append((a_word_span, o_word_span, int2sentiment[relation_labels[a_idx][o_idx]]))
    return triples 

def convert_pad_tensor_to_list(batch_data, mask):
 
    assert len(mask.shape) == 2
    batch_data = batch_data.detach().cpu().tolist()
    len_list = torch.sum(mask, dim=-1).detach().cpu().tolist()
    ret = []
    for length, data in zip(len_list, batch_data):
        ret.append(data[: length])
 
    return ret 

if __name__ == '__main__':
    from transformers import BertTokenizer 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = ABSAProcessor(tokenizer=tokenizer)
    root = '../../../dataset/del/CDASTE-Data'
    for domain in os.listdir(root):
        if '.' in domain:
            continue
        examples = processor.get_examples(f'../../../dataset/del/CDASTE-Data/{domain}', 'train.txt')
        features = processor.convert_examples_to_features(examples)
        for example, feature in zip(examples, features):
            triples1 = example[1]
            # print(domain, example)
            triples2 = convert_predictions_to_triples(feature.spans, feature.relation_labels, feature.token_range)
            assert len(feature.input_ids) == feature.token_range[-1][1] + 2
            if str(sorted(triples1)) != str(sorted(triples2)):
                print(example, len(feature.token_range))
                print(triples2)
                print()
