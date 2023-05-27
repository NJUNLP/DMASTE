import torch 
from utils.data_utils import convert_pad_tensor_to_list, convert_predictions_to_triples, SpanLabel, RelationLabel
from sklearn.metrics import precision_score, recall_score, f1_score 
def convert_relations_to_list(relations, mask):
    ret = []
    for i in range(relations.shape[0]):
        r, m = relations[i], mask[i]
        width = torch.sum(m, dim=0)
        height = torch.sum(m, dim=1)
        assert torch.sum(torch.eq(width, height)) == width.shape[0]
        ret.append(r[: width[0], :width[0]].detach().tolist())
    return ret 


class Metric:
    def __init__(self):
        self.triplet = {'pred': 0, 'golden': 0, 'tp': 0}
        self.ner = {'p': 0, 'r': 0, 'f1': 0}
        self.relation = {'p': 0, 'r': 0, 'f1': 0}
        self.aspect = {'pred': 0, 'golden': 0, 'tp': 0}
        self.opinion = {'pred': 0, 'golden': 0, 'tp': 0}
        self.pos_relation = {'pred': 0, 'golden': 0, 'tp': 0}
        self.neg_relation = {'pred': 0, 'golden': 0, 'tp': 0}
        self.neu_relation = {'pred': 0, 'golden': 0, 'tp': 0}
        self.inv_relaiton = {'pred': 0, 'golden': 0, 'tp': 0}
        self.num_ins = 0 
    def get_metric(self):
        ret = dict()
        mean_metric = {'ner': self.ner, 'relation': self.relation}
        for type_ in mean_metric:
            type_metric = dict()
            for metric_name in ['p', 'r', 'f1']:
                type_metric[metric_name] = mean_metric[type_][metric_name] / self.num_ins
            ret[type_] = type_metric
        num_metric = {'triplet': self.triplet, 'aspect': self.aspect, 'opinion': self.opinion, 'pos_rel': self.pos_relation,
                      'neg_rel': self.neg_relation, 'nue_rel': self.neu_relation, 'inv_rel': self.inv_relaiton}
        for type_ in num_metric:
            num = num_metric[type_]
            tp, golden, pred = num['tp'], num['golden'], num['pred']
            p = tp / pred if pred != 0 else 0 
            r = tp / golden if golden != 0 else 0 
            f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0
            ret[type_] = {'p': p, 'r': r, 'f1': f1}
        return ret 
    def get_span_labels(self, batch, output):
        span_labels = batch['span_labels']
        span_mask = batch['span_mask']
        span_labels = convert_pad_tensor_to_list(span_labels, span_mask)
        span_predictions = output['ner_output']['ner_scores']
        span_predictions = torch.argmax(span_predictions, dim=-1)
        span_predictions = convert_pad_tensor_to_list(span_predictions, span_mask)
        return span_labels, span_predictions

    def cal_num(self, ins_pred, ins_label, ins_type, metric):
        golden = set([i for i, x in enumerate(ins_label) if x == ins_type])
        pred = set([i for i, x in enumerate(ins_pred) if x == ins_type])
        tp = golden & pred
        ins_metric = {'golden': len(golden), 'pred': len(pred), 'tp': len(tp)}
        for k in ins_metric:
            metric[k] += ins_metric[k]

    def cal_span_metric(self, span_labels, span_predictions):
        for ins_label, ins_pred in zip(span_labels, span_predictions):
            assert len(ins_label) == len(ins_pred)
            self.num_ins += 1
            self.ner['p'] += precision_score(ins_label, ins_pred, average='macro', zero_division=1)
            self.ner['r'] += recall_score(ins_label, ins_pred, average='macro', zero_division=1)
            self.ner['f1'] += f1_score(ins_label, ins_pred, average='macro', zero_division=1)
            self.cal_num(ins_pred, ins_label, SpanLabel.ASPECT, self.aspect)
            self.cal_num(ins_pred, ins_label, SpanLabel.OPINION, self.opinion)


    def cal_relation_metric(self, output):
        relation_labels = output['relation_output']['pruned_relation_labels']
        relation_mask = output['relation_output']['relation_mask']
        relation_predictions = output['relation_output']['relation_scores']
        relation_predictions = torch.argmax(relation_predictions, dim=-1)
        assert relation_labels.shape == relation_predictions.shape 
        relation_labels = convert_relations_to_list(relation_labels, relation_mask)
        relation_predictions = convert_relations_to_list(relation_predictions, relation_mask)
        for ins_label, ins_pred in zip(relation_labels, relation_predictions):
            ins_label = [x for row in ins_label for x in row]
            ins_pred = [x for row in ins_pred for x in row]
            assert len(ins_label) == len(ins_pred)
            self.relation['p'] += precision_score(ins_label, ins_pred, average='macro', zero_division=1)
            self.relation['r'] += recall_score(ins_label, ins_pred, average='macro', zero_division=1)
            self.relation['f1'] += f1_score(ins_label, ins_pred, average='macro', zero_division=1)
            self.cal_num(ins_pred, ins_label, RelationLabel.NEG, self.neg_relation)
            self.cal_num(ins_pred, ins_label, RelationLabel.NEU, self.neu_relation)
            self.cal_num(ins_pred, ins_label, RelationLabel.POS, self.pos_relation)
            self.cal_num(ins_pred, ins_label, RelationLabel.INVALID, self.inv_relaiton)

            
    def compute(self, examples, output, batch):
        # ner
        span_labels, span_predictions = self.get_span_labels(batch, output)
        self.cal_span_metric(span_labels, span_predictions)
        # relation
        self.cal_relation_metric(output)

        # triples
        spans_a = output['relation_output']['spans_a']
        spans_a_mask = output['relation_output']['spans_a_mask']
        spans_a = convert_pad_tensor_to_list(spans_a, spans_a_mask)
        spans_o = output['relation_output']['spans_o']
        spans_o_mask = output['relation_output']['spans_o_mask']
        spans_o = convert_pad_tensor_to_list(spans_o, spans_o_mask)
        relation_scores = output['relation_output']['relation_scores']
        relation_mask = output['relation_output']['relation_mask']

        predict_relations = torch.argmax(relation_scores, dim=-1)
        # print('relation', predict_relations.shape, batch['relation_labels'].shape)
        
        predict_relations = convert_relations_to_list(predict_relations, relation_mask)
        # print(predict_relations)
        token_range, token_range_mask = batch['token_range'], batch['token_range_mask']
        token_range = convert_pad_tensor_to_list(token_range, token_range_mask)
        predict_triples = []
        for i in range(len(examples)):
            triples1 = examples[i][1]
            triples2 = convert_predictions_to_triples(spans_a=spans_a[i], spans_o=spans_o[i], relation_labels=predict_relations[i], token_range=token_range[i])
            predict_triples.append(triples2)
            self.triplet['pred'] += len(triples2)
            self.triplet['golden'] += len(triples1)

            for t1 in triples1:
                for t2 in triples2:
                    if str(t1) == str(t2):
                        self.triplet['tp'] += 1
        return predict_triples 

                    

