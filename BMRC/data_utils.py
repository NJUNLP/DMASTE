from torch.utils.data import Dataset
import random
import torch

class Domain:
    Target = 1
    Source = 0
class Unlabeled_Dataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        self.data = []
        self.max_len = max_len
        with open(path) as f:
            for line in f:
                sent = line.split('####')[-1].strip()
                words = ['[ia]'] + sent.split()
                idx_list1 = random.sample(range(len(words)), 4)
                idx_list2 = random.sample(range(1, 6), 4)
                sample_words = [words[i: i + j] for i, j in zip(idx_list1, idx_list2)]
                query_list = [["What", "aspects", "?"],
                              ["What", "opinions", "?"],
                              ["What", "opinion", "given", "the", "aspect"] + sample_words[0] + ["?"],
                              ["What", "sentiment", "given", "the", "aspect"] + sample_words[1] + ["and", "the", "opinion"] + sample_words[2] + ['?'],
                              ["What", "aspect", "does", "the", "opinion"] + sample_words[3] + ["describe", "?"]]
                for query in query_list:
                    input_token = ['[CLS]'] + query + ['[SEP]'] + words 
                    seg = [0] * (len(query) + 2) + [1] * len(words)
                    domain_label = [-1] * (len(query) + 2) + [Domain.Target] * len(words)
                    input_ids = tokenizer.convert_tokens_to_ids([word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in input_token])
                    self.data.append({'input_ids': input_ids, 'token_type_ids': seg, 'domain_label': domain_label})
    
    def __getitem__(self, i):
        self.data[i]['attention_mask'] = [1] * len(self.data[i]['input_ids'])
        ret = dict()
        for k in self.data[i]:
            ret[k] = self.data[i][k][: self.max_len]
            pad = 0 if k != 'domain_label' else -1
            ret[k] = ret[k] + [pad] * (self.max_len - len(ret[k]))
            ret[k] = torch.tensor(ret[k])
        # return ret['input_ids'], ret['token_type_ids'], ret['attention_mask'], ret['domain_label']
        return ret
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    from transformers import BertTokenizer 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = Unlabeled_Dataset('../amazon/home.txt', tokenizer)
    for i in range(10):
        print(dataset[i])
        print(tokenizer.convert_ids_to_tokens(dataset[i]['input_ids']))
        print()
                    
                    
                    
    
    