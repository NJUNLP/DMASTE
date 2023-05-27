import torch
from torch.nn.modules import dropout 
import torch.nn.functional as F

from utils.data_utils import SpanLabel

from models.feedForward import FeedForward
class NERModel(torch.nn.Module):
    def __init__(self, span_embed_dim, hidden_dim=150, num_layers=2, activation=torch.nn.ReLU(), dropout=0.4, n_labels=3):
        super(NERModel, self).__init__()
        self.span_embed_dim = span_embed_dim
        self.n_labels = n_labels
        self.ffnn = FeedForward(input_dim=span_embed_dim, hidden_dim=hidden_dim, num_layers=num_layers, activation=activation, dropout=dropout)
        self.classifier = torch.nn.Linear(in_features=hidden_dim, out_features=n_labels)
        torch.nn.init.xavier_normal_(self.classifier.weight)
        self._loss = torch.nn.CrossEntropyLoss(reduction='sum')


    def forward(self, span_embeddings, span_mask, span_labels=None):
        # shape: bsz, span_length, n_labels
        ner_scores = self.classifier(self.ffnn(span_embeddings))
        masked_scores = torch.zeros_like(span_mask, dtype=torch.long) + 1e20
        ner_scores[..., SpanLabel.INVALID] = torch.where(span_mask.bool(), ner_scores[..., SpanLabel.INVALID], masked_scores)
        softmax_ner_scores = ner_scores.softmax(dim=-1)
        output_dict = dict()
        output_dict.update(ner_scores=softmax_ner_scores)
        output_dict.update(opinion_scores=ner_scores.softmax(dim=-1)[..., SpanLabel.OPINION])
        output_dict.update(target_scores=ner_scores.softmax(dim=-1)[..., SpanLabel.ASPECT])
        loss = torch.tensor(0,dtype=torch.float).to(span_mask.device)
        if span_labels is not None:
            # test 
            # predicts = torch.argmax(softmax_ner_scores, dim=-1)
            # from sklearn.metrics import precision_score, recall_score, f1_score

            # valid_mask = span_labels != SpanLabel.INVALID 
            # predicts = predicts[valid_mask]
            # new_labels = span_labels[valid_mask]
            # p, r = precision_score(new_labels.cpu().tolist(), predicts.cpu().tolist(), average='macro'), recall_score(new_labels.cpu().tolist(), predicts.cpu().tolist(), average='macro')
            # f1 = f1_score(new_labels.cpu().tolist(), predicts.cpu().tolist(), average='macro')
            # print(f'ner p: {p}, r: {r}, f1: {f1}')
            # end 
            ner_scores_flat = ner_scores.view(
                -1, self.n_labels
            )
            ner_labels_flat = span_labels.view(-1)
            mask_flat = span_mask.view(-1).bool()

            loss = self._loss(ner_scores_flat[mask_flat], ner_labels_flat[mask_flat])

        output_dict["loss"] = loss

        return output_dict

