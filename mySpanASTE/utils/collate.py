import torch 


from utils.data_utils import RelationLabel

def collate_fn(data):
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
    input_ids, attention_mask = pad_and_tensor(input_ids)
    spans = [f.spans for f in data]
    max_span_length = max([len(x) for x in spans])
    triples = [f.triples for f in data]
    relation_labels = []
    relation_mask = []
    for i, ins_triple in enumerate(triples):
        labels = torch.zeros([max_span_length, max_span_length], dtype=torch.long) + RelationLabel.INVALID
        for triple in ins_triple:
            a, o, s = triple
            try:
                a_idx, o_idx = spans[i].index(a), spans[i].index(o)
                labels[a_idx, o_idx] = s 
            except:
                pass 
        mask = torch.zeros([max_span_length, max_span_length], dtype=torch.long)
        mask[: len(spans[i]), : len(spans[i])] = 1 
        relation_labels.append(labels)
        relation_mask.append(mask)        
    relation_labels = torch.stack(relation_labels, dim=0)
    relation_mask = torch.stack(relation_mask, dim=0)
    spans, _ = pad_and_tensor(spans)
    span_labels = [f.span_labels for f in data]
    span_labels, span_mask = pad_and_tensor(span_labels)
    seq_length = [f.seq_length for f in data]
    seq_length = torch.tensor(seq_length).to(torch.long)
    token_range = [f.token_range for f in data]
    token_range, token_range_mask = pad_and_tensor(token_range)
    token_range_mask = token_range_mask[..., 0]
    batch = {'input_ids': input_ids,
             'attention_mask': attention_mask,
             'spans': spans,
             'span_labels': span_labels,
             'span_mask': span_mask,
             'relation_labels': relation_labels,
             'relation_mask': relation_mask,
             'seq_length': seq_length,
             'token_range': token_range,
             'token_range_mask': token_range_mask}
    return batch 


