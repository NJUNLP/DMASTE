import torch 

from utils.index_select import batched_index_select
from models.ner import NERModel
from models.relation import RelationModel
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

    def forward(self, input_ids, attention_mask, spans, span_mask, seq_length, span_labels=None, relation_labels=None):
        text_embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 
        span_embeddings = self.text_to_span_embeds(text_embeddings, spans)
        ner_output = self.ner(span_embeddings, span_mask, span_labels)
        relation_output = self.relation(spans, ner_output['ner_scores'], span_embeddings,  span_mask, seq_length=seq_length, relation_labels=relation_labels)
        loss = ner_output['loss'] + relation_output['loss']
        num_spans = span_mask.sum()
        num_relations = relation_output['relation_mask'].sum()
        loss = ner_output['loss']  + relation_output['loss'] 
        return {'loss': loss, 
                'ner_loss': ner_output['loss'] / (num_spans + num_relations), 
                'relation_loss': relation_output['loss'] / (num_spans + num_relations), 
                'ner_output': ner_output, 
                'relation_output': relation_output}
        
    def text_to_span_embeds(self, text_embeddings, spans):
        # batch index select
        span_starts, span_ends = [index.squeeze(-1) for index in spans.split(1, dim=-1)]

        start_embeddings = batched_index_select(text_embeddings, span_starts)
        end_embeddings = batched_index_select(text_embeddings, span_ends)
        width = span_ends - span_starts
        width_embedding = self.width_embedding(width)
        span_embedding = torch.cat([start_embeddings, end_embeddings, width_embedding], dim=-1)
        return span_embedding 


