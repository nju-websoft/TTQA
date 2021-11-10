
import torch.nn as nn
import constant
import collections

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import utils_kbqa
import torch


GraphPairData = collections.namedtuple('GraphPairData', [
	'graph1_structure',
	'graph1_sequence',
	'graph2_structure',
	'graph2_sequence',
    'graph2_canoical_node_to_sequence'])


class BERTEncoder(nn.Module):
    """Encoder module that projects node and edge features to some embeddings."""

    def __init__(self, args, device):
        super(BERTEncoder, self).__init__()
        self.args = args
        self.dep_emb = nn.Embedding(len(constant.DEPREL_TO_ID), 768)
        self.dep_emb.weight.requires_grad = True
        print('####device:\t', device)
        self.dep_emb.to(device)
        self.tokenizer = BertTokenizer.from_pretrained(args['model_name_or_path'], do_lower_case=args['do_lower_case'])
        self.bert_model = BertModel.from_pretrained(args['model_name_or_path'])
        self.bert_model.to(device)
        self.device = device

    def forward(self, batch_sample_list):
        GraphPairData_list = []
        for sample in batch_sample_list:
            g2 = sample['g2']
            GraphPairData_list.append(GraphPairData(
                graph1_structure=utils_kbqa.deplist_to_networkxgraph(dependencies_list=sample['g1']),
                graph1_sequence=sample['abstract_question'],
                # graph2_structure=utils_kbqa.tripleslist_to_networkxgraph(triples=g2['triples']),
                graph2_structure=utils_kbqa.tripleslist_to_networkxgraph_nodegraph(triples=g2['triples']),
                graph2_sequence=g2['sequence'],
                graph2_canoical_node_to_sequence=g2['canoical_node_to_sequence'],
            ))

        train_features = utils_kbqa.convert_examples_to_features(GraphPairData_list, self.args['max_seq_length'],self.tokenizer, device=self.device)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long, device=self.device)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long, device=self.device)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long, device=self.device)
        encoded_layers, cls_layers = self.bert_model(input_ids=all_input_ids, token_type_ids=all_segment_ids, attention_mask=all_input_mask, output_all_encoded_layers=False)

        cls_list = []
        batch_graphs = []
        batch_node_to_vectors = []
        batch_edge_to_vectors = []
        for index, train_feature in enumerate(train_features):
            bert_encoder_result = encoded_layers[index]

            graph1_nodeid_to_embeddings = collections.OrderedDict()
            for nodeid, inputids in train_feature.graph1_nodeid_to_inputids.items():
                graph1_nodeid_to_embeddings[nodeid] = utils_kbqa.get_encoder_embedding(index_list=inputids, encoder_layer=bert_encoder_result)

            graph1_edgeid_to_embeddings = collections.OrderedDict()
            for edgeid, inputid in train_feature.graph1_edgeids_to_deptags.items():
                graph1_edgeid_to_embeddings[edgeid] = utils_kbqa.get_dep_embedding(inputid, embedding=self.dep_emb)

            graph2_nodeid_to_embeddings = collections.OrderedDict()
            for nodeid, inputids in train_feature.graph2_nodeid_to_inputids.items():
                graph2_nodeid_to_embeddings[nodeid] = utils_kbqa.get_encoder_embedding(index_list=inputids, encoder_layer=bert_encoder_result)

            graph2_edgeid_to_embeddings = collections.OrderedDict()
            for edgeid, inputids in train_feature.graph2_edgeids_to_inputids.items():
                graph2_edgeid_to_embeddings[edgeid] = utils_kbqa.get_encoder_embedding(index_list=inputids, encoder_layer=bert_encoder_result)

            # graph1_sep_rep = utils_kbqa.get_encoder_embedding(index_list=[train_feature.graph1_sep_pos], encoder_layer=bert_encoder_result)
            # graph2_sep_rep = utils_kbqa.get_encoder_embedding(index_list=[train_feature.graph2_sep_pos], encoder_layer=bert_encoder_result)

            cls_list.append(cls_layers[index])
            batch_graphs.append((train_feature.graph1_nx, train_feature.graph2_nx))
            batch_node_to_vectors.append((graph1_nodeid_to_embeddings, graph2_nodeid_to_embeddings))
            batch_edge_to_vectors.append((graph1_edgeid_to_embeddings, graph2_edgeid_to_embeddings))

        return cls_list, batch_graphs, batch_node_to_vectors, batch_edge_to_vectors

