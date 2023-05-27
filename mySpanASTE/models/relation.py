from os import read
import torch
import math 

from utils.data_utils import RelationLabel, SpanLabel 
from utils.index_select import batched_index_select
from models.feedForward import FeedForward

def bucket_values(
    distances: torch.Tensor, num_identity_buckets: int = 4, num_total_buckets: int = 10
) -> torch.Tensor:
    """
    Places the given values (designed for distances) into `num_total_buckets`semi-logscale
    buckets, with `num_identity_buckets` of these capturing single values.
    The default settings will bucket values into the following buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    # Parameters
    distances : `torch.Tensor`, required.
        A Tensor of any size, to be bucketed.
    num_identity_buckets: `int`, optional (default = `4`).
        The number of identity buckets (those only holding a single value).
    num_total_buckets : `int`, (default = `10`)
        The total number of buckets to bucket values into.
    # Returns
    `torch.Tensor`
        A tensor of the same shape as the input, containing the indices of the buckets
        the values were placed in.
    """
    # Chunk the values into semi-logscale buckets using .floor().
    # This is a semi-logscale bucketing because we divide by log(2) after taking the log.
    # We do this to make the buckets more granular in the initial range, where we expect
    # most values to fall. We then add (num_identity_buckets - 1) because we want these indices
    # to start _after_ the fixed number of buckets which we specified would only hold single values.
    logspace_index = (distances.float().log() / math.log(2)).floor().long() + (
        num_identity_buckets - 1
    )
    # create a mask for values which will go into single number buckets (i.e not a range).
    use_identity_mask = (distances <= num_identity_buckets).long()
    use_buckets_mask = 1 + (-1 * use_identity_mask)
    # Use the original values if they are less than num_identity_buckets, otherwise
    # use the logspace indices.
    combined_index = use_identity_mask * distances + use_buckets_mask * logspace_index
    # Clamp to put anything > num_total_buckets into the final bucket.
    return combined_index.clamp(0, num_total_buckets - 1)


class RelationModel(torch.nn.Module):
    def __init__(self, pair_embed_dim, spans_per_word=0.5, distance_embed_dim=128, hidden_dim=150, num_layers=2, activation=torch.nn.ReLU(), dropout=0.4, n_labels=4):
        super(RelationModel, self).__init__()
        self.pair_embed_dim = pair_embed_dim
        self.n_labels = n_labels
        self.spans_per_word = spans_per_word
        self.distance_embedding = torch.nn.Embedding(512, embedding_dim=distance_embed_dim)
        torch.nn.init.xavier_normal_(self.distance_embedding.weight)
        self.ffnn = FeedForward(input_dim=pair_embed_dim + distance_embed_dim, hidden_dim=hidden_dim, num_layers=num_layers, activation=activation, dropout=dropout)
        self.classifier = torch.nn.Linear(in_features=hidden_dim, out_features=n_labels)
        torch.nn.init.xavier_normal_(self.classifier.weight)
        self._loss = torch.nn.CrossEntropyLoss(reduction='sum')
    
    def forward(
        self,  # type: ignore
        spans, 
        ner_scores, 
        span_embeddings, 
        span_mask,
        seq_length,
        relation_labels = None
    ):
        pruned_a = self._prune_spans(ner_scores[..., SpanLabel.ASPECT], span_mask, seq_length)
        pruned_o = self._prune_spans(ner_scores[..., SpanLabel.OPINION], span_mask, seq_length)
        spans_a = batched_index_select(spans, pruned_a['indices'])
        spans_o = batched_index_select(spans, pruned_o['indices'])
        relation_scores, relation_mask, relation_embeddings  = self.predict_relation(spans, pruned_a['indices'], pruned_a['mask'], pruned_o['indices'], pruned_o['mask'], span_embeddings)
        pruned_relation_labels = None 
        loss = torch.tensor(0, dtype=torch.float).to(spans_a.device)
        if relation_labels is not None:
            pruned_relation_labels = self.get_pruned_gold_relations(relation_labels, pruned_a, pruned_o)

            flatten_relation_scores = relation_scores.reshape([-1, self.n_labels])
            flatten_labels = pruned_relation_labels.view(-1)
            flatten_score_mask = relation_mask.unsqueeze(-1).expand_as(relation_scores).view(flatten_relation_scores.shape)
            flatten_relation_scores = flatten_relation_scores[flatten_score_mask]
            flatten_labels = flatten_labels[relation_mask.view(-1)]
            loss = self._loss(input=flatten_relation_scores.reshape([-1, self.n_labels]), target=flatten_labels)

        
        return {'relation_scores': torch.softmax(relation_scores, dim=-1), 
                'relation_mask': relation_mask, 
                'relation_embeddings': relation_embeddings,
                'pruned_relation_labels': pruned_relation_labels, 
                'loss': loss,
                'pruned_a': pruned_a,
                'pruned_o': pruned_o,
                'spans_a': spans_a,
                'spans_a_mask': pruned_a['mask'],
                'spans_o': spans_o,
                'spans_o_mask': pruned_o['mask']}

    def get_pruned_gold_relations(self, relation_labels, pruned_a, pruned_o):
        indices_a = pruned_a['indices'] 
        indices_o = pruned_o['indices'] 
        new_relation_labels = []
        for i in range(relation_labels.shape[0]):
            entry = relation_labels[i]
            width = indices_a[i].shape[0]
            assert indices_a[i].shape[0] == indices_o[i].shape[0]
            idx_a = indices_a[i].unsqueeze(-1).expand([width, width])
            idx_o = indices_o[i].unsqueeze(0).expand([width, width])
            # print(entry.shape, idx_a.shape, idx_o.shape)
            labels = entry[idx_a.reshape(-1), idx_o.reshape(-1)]
            new_relation_labels.append(labels.reshape(width, width))
            
        new_relation_labels = torch.stack(new_relation_labels, dim=0)
        return new_relation_labels


    def predict_relation(self, spans, a_indices, a_mask, o_indices, o_mask, span_embeddings):
        bsz, seq_a = a_indices.shape 
        _, seq_o = o_indices.shape
        mask = a_mask.unsqueeze(-1) * o_mask.unsqueeze(1)
        # print('mask', mask.shape)
        new_shape = (bsz, seq_a, seq_o)
        a_indices = a_indices.unsqueeze(2).expand(new_shape)
        o_indices = o_indices.unsqueeze(1).expand(new_shape)

        a_embeddings = batched_index_select(span_embeddings, a_indices)
        o_embeddings = batched_index_select(span_embeddings, o_indices)
        spans_a = batched_index_select(spans, a_indices)
        spans_o = batched_index_select(spans, o_indices)
        dis1 = spans_a[..., 0] - spans_o[..., 1]
        dis2 = spans_a[..., 1] - spans_o[..., 0]
        dis, _ = torch.min(torch.cat([torch.absolute(dis1).unsqueeze(-1), torch.absolute(dis2).unsqueeze(-1)], dim=-1), dim=-1)
        dis = bucket_values(dis)
        distance_embeddings = self.distance_embedding(dis)
        
        pair_embeddings = torch.cat([a_embeddings, o_embeddings, distance_embeddings], dim=-1)
        pair_scores = self.classifier(self.ffnn(pair_embeddings))
        return pair_scores, mask, pair_embeddings

    def _prune_spans(self, scores, mask, seq_length):
        num_spans_to_keep = torch.ceil(
            seq_length.float() * self.spans_per_word
        ).long()
        num_spans = scores.shape[1]
        num_items_to_keep = torch.clamp(num_spans_to_keep, max=num_spans).to(scores.device)
        max_items_to_keep = max(num_items_to_keep.max().item(), 1)
        scores = torch.where(mask.bool(), scores, torch.zeros_like(scores) + -1e20)
        _, top_indices = scores.topk(max_items_to_keep, dim=1)
        top_indices_mask = torch.arange(start=0, end=max_items_to_keep).to(scores.device).reshape([1, -1]).expand_as(top_indices)
        top_indices_mask = top_indices_mask < num_items_to_keep.reshape(-1, 1)

        return {'indices': top_indices, 'mask': top_indices_mask}