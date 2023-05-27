def idx2term(sent, triplets):
    words = sent.split()
    ret = []
    for a, o, s in triplets:
        a_term = words[a[0]: a[-1] + 1]
        o_term = words[o[0]: o[-1] + 1]
        ret.append((' '.join(a_term), ' '.join(o_term), s))
    return ret 
def convert(examples, all_preds, golden):
    ret = []
    for sent, pred_triplets in zip(examples, all_preds):
        sent = ' '.join(sent)
        ret_triplets = []
        for a, o, s in pred_triplets:
            a_start = sent.find(a)
            o_start = sent.find(o)
            if a_start != -1 and o_start != -1:
                a_span = [len(sent[: a_start].split()), len(sent[: a_start].split()) + len(a.split()) - 1]
                o_span = [len(sent[: o_start].split()), len(sent[: o_start].split()) + len(o.split()) - 1]
                a_span = [a_span[0]] if a_span[0] == a_span[1] else a_span 
                o_span = [o_span[0]] if o_span[0] == o_span[1] else o_span
                ret_triplets.append((a_span, o_span, s[:3].upper()))
        ret.append(ret_triplets)
    tp = 0 
    pred_num = 0
    golden_num = 0
    for pred, gold in zip(ret, golden):
        pred = [str(x) for x in pred]
        gold = [str(x) for x in gold]
        correct = set(pred) & set(gold)
        tp += len(correct)
        pred_num += len(pred)
        golden_num += len(gold)
    p = tp / pred_num if pred_num != 0 else 0
    r = tp / golden_num if golden_num != 0 else 0 
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0 
    print(f'p: {p}, r: {r}, f1: {f1}') 
            