import os 
from enum import IntEnum 
from torch.utils.data import Dataset 

class DomainLabel(IntEnum):
    Source = 0 
    Target = 1


class UnlabeledDataset(Dataset):
    def __init__(self, features):
        self.features = features 
    
    def __getitem__(self, index):
        return self.features[index]
    
    def __len__(self):
        return len(self.features)

class UnlabeledFeature:
    def __init__(self, input_ids, spans, token_range, seq_length) -> None:
        self.input_ids = input_ids
        self.spans = spans
        self.seq_length = seq_length
        self.token_range = token_range 


class UnlabeledProcessor:
    def __init__(self, tokenizer, min_span_width=1, max_span_width=10, max_seq_length=512):
        self.tokenizer = tokenizer 
        self.null_aspect_id = self.tokenizer.convert_tokens_to_ids(['[ia]'])
        self.min_span_width = min_span_width 
        self.max_span_width = max_span_width 
        self.max_seq_length = max_seq_length

    def get_examples(self, data_dir, mode):
        file_name = os.path.join(data_dir, mode)
        lines = []
        with open(file_name) as f:
            counter = 0
            for line in f:
                lines.append('[ia] ' + line.split(' #### ')[-1])
        return lines 
    
    def convert_examples_to_features(self, examples):
        features = []
        for sent in examples:
            input_ids, token_range = self._tokenize(sent)
            seq_length = len(sent.split())
            spans = self._enumerate_spans(token_range)
            features.append(UnlabeledFeature(input_ids=input_ids,
                            spans=spans,
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
        input_ids = [self.tokenizer.cls_token_id]
        token_range = []
        start_ids = 1
        for word in words:
            word_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
            if len(input_ids) + len(word_ids) + 1 > self.max_seq_length:
                break 
            input_ids.extend(word_ids)
            token_range.append([start_ids, start_ids + len(word_ids) - 1])
            start_ids += len(word_ids)
        input_ids.append(self.tokenizer.sep_token_id)
        return input_ids, token_range
    



if __name__ == '__main__':
    from transformers import BertTokenizer 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<null-aspect>']})

    processor = UnlabeledProcessor(tokenizer=tokenizer)
    root = '../../../dataset/amazon'
    for domain in os.listdir(root):

        examples = processor.get_examples('../../../dataset/amazon/', domain, num_data=1000)
        features = processor.convert_examples_to_features(examples)
        for example, feature in zip(examples, features):
            print(example)
            print(tokenizer.convert_ids_to_tokens(feature.input_ids))
            print(feature.token_range)
            print()
