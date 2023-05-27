# coding: UTF-8
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from functions import ReverseLayerF

class BERTModel(nn.Module):
    def __init__(self, args):
        hidden_size = args.hidden_size

        super(BERTModel, self).__init__()

        # BERT模型
        # if args.bert_model_type == 'bert-base-uncased':
        self._bert = BertModel.from_pretrained(args.bert_model_type)
        self._tokenizer = BertTokenizer.from_pretrained(args.bert_model_type)
        print('Bertbase model loaded')

        # else:
        #     raise KeyError('Config.args.bert_model_type should be bert-based-uncased. ')

        self.classifier_start = nn.Linear(hidden_size, 2)

        self.classifier_end = nn.Linear(hidden_size, 2)

        self._classifier_sentiment = nn.Linear(hidden_size, 3)
        self.domain_classifier = nn.Linear(hidden_size, 2)

    def forward(self, query_tensor, query_mask, query_seg, step, alpha=None, domain=None):

        hidden_states = self._bert(query_tensor, attention_mask=query_mask, token_type_ids=query_seg)[0]
        ret = dict()
        if step == 0:  # predict entity
            out_scores_start = self.classifier_start(hidden_states)
            out_scores_end = self.classifier_end(hidden_states)
            ret['cls'] = [out_scores_start, out_scores_end]
            # return out_scores_start, out_scores_end
        else:  # predict sentiment
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self._classifier_sentiment(cls_hidden_states)
            ret['cls'] = cls_hidden_scores
            # return cls_hidden_scores
        if domain is not None:
            reverse_feature = ReverseLayerF.apply(hidden_states if step == 0 else hidden_states[:, 0, :], alpha)
            domain_scores = self.domain_classifier(reverse_feature)
            ret['domain_scores'] = domain_scores 
            return ret 
        else:
            return ret['cls']
        