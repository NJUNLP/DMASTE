import torch 

from torch.nn import functional as F
from utils.index_select import batched_index_select
from models.ner import NERModel
from models.relation import RelationModel
from models.functions import ReverseLayerF
class SpanModel(torch.nn.Module):
    def __init__(self, encoder, width_embedding_dim=20, max_width=512, spans_per_word=0.5):
        super(SpanModel, self).__init__()
        self.encoder = encoder 
        self.max_width = max_width
        self.width_embedding = torch.nn.Embedding(max_width, width_embedding_dim)
        torch.nn.init.xavier_normal_(self.width_embedding.weight)
        self.span_embed_dim = 768 * 2 + width_embedding_dim 
        self.ner = NERModel(span_embed_dim=self.span_embed_dim)
        self.relation = RelationModel(pair_embed_dim=self.span_embed_dim * 2, spans_per_word=spans_per_word)
        self.domain_cls = torch.nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask, spans, span_mask, seq_length, span_labels=None, relation_labels=None, alpha=None, domain=None):
        text_embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 
        span_embeddings = self.text_to_span_embeds(text_embeddings, spans)
        ner_output = self.ner(span_embeddings, span_mask, span_labels)
        relation_output = self.relation(spans, ner_output['ner_scores'], span_embeddings,  span_mask, seq_length=seq_length, relation_labels=relation_labels)
        loss = ner_output['loss'] + relation_output['loss']
        num_spans = span_mask.sum()
        num_relations = relation_output['relation_mask'].sum()
        loss = ner_output['loss']  + relation_output['loss'] 
        domain_loss = torch.tensor([0.]).cuda()
        if domain is not None:
            reverse_embed = ReverseLayerF.apply(text_embeddings, alpha)
            domain_scores = self.domain_cls(reverse_embed)
            domain_label = torch.where(attention_mask.bool(), torch.zeros_like(attention_mask).long() + domain, torch.zeros_like(attention_mask).long() -1 )
            # reverse_rel_embed = ReverseLayerF.apply(relation_output['relation_embeddings'], alpha)
            # rel_domain_scores = self.relation_domain_cls(reverse_rel_embed)
            # zero = torch.zeros_like(relation_output['relation_mask'])
            # rel_domain_label = torch.where(relation_output['relation_mask'].bool(), zero.long() + domain, zero.long() - 1)
            domain_loss = F.cross_entropy(domain_scores.view(-1, 2), domain_label.view(-1).long(), reduction='sum', ignore_index=-1)
            # rel_domain_loss = F.cross_entropy(rel_domain_scores.view(-1, 2), rel_domain_label.view(-1).long(), reduction='sum', ignore_index=-1)

            
        return {'loss': loss, 
                'ner_loss': ner_output['loss'] / (num_spans + num_relations), 
                'relation_loss': relation_output['loss'] / (num_spans + num_relations), 
                'ner_output': ner_output, 
                'relation_output': relation_output,
                'domain_loss': domain_loss}
        
    def text_to_span_embeds(self, text_embeddings, spans):
        # batch index select
        span_starts, span_ends = [index.squeeze(-1) for index in spans.split(1, dim=-1)]

        start_embeddings = batched_index_select(text_embeddings, span_starts)
        end_embeddings = batched_index_select(text_embeddings, span_ends)
        width = span_ends - span_starts
        width_embedding = self.width_embedding(width)
        span_embedding = torch.cat([start_embeddings, end_embeddings, width_embedding], dim=-1)
        return span_embedding 


