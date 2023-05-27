import torch 

from utils.data_utils import RelationLabel
from utils.data_utils_unlabeled import DomainLabel

def collate_fn_target(data):
    """批处理，填充同一batch中句子最大的长度"""
    def pad_and_tensor(data, pad_value=0):
        max_len = max([len(x) for x in data])
        new_data = []
        mask = []
        for x in data:
            tmp_data = torch.tensor(x)
            size = tmp_data.shape
            pad_data = torch.zeros((max_len - size[0], *size[1:]))
            new_data.append(torch.cat([tmp_data, pad_data], dim=0))
            mask.append(torch.cat([torch.ones_like(tmp_data), torch.zeros_like(pad_data)], dim=0))
        return torch.stack(new_data, dim=0).to(torch.long), torch.stack(mask, dim=0).to(torch.long)
    input_ids = [f.input_ids for f in data]
    bsz = len(data)
    input_ids, attention_mask = pad_and_tensor(input_ids)
    spans = [f.spans for f in data]
    
    spans, span_mask = pad_and_tensor(spans)
    span_mask = span_mask[...,0]
    seq_length = [f.seq_length for f in data]
    seq_length = torch.tensor(seq_length).to(torch.long)
    token_range = [f.token_range for f in data]
    token_range, token_range_mask = pad_and_tensor(token_range)
    token_range_mask = token_range_mask[..., 0]
    batch = {'input_ids': input_ids,
             'attention_mask': attention_mask,
             'spans': spans,
             'span_mask': span_mask,
             'seq_length': seq_length,
             'token_range': token_range,
             'token_range_mask': token_range_mask}
    return batch 


