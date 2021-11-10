
import json
import torch


def get_checkpoints_names(config):
    name = config['data']['dataset'] \
           +"_nodeway"+config['node_ebedding_way'] \
           +"_atteway"+config['attention_way'] \
           +"_aggway"+config['aggregator_way'] \
           +"_mode" + config['training']['mode']\
           +"_loss" + config['training']['loss']\
           +"_bs" + str(config['training']['batch_size'])\
           +"_lr" + str(config['training']['learning_rate'])\
           +"_wd" + str(config['training']['weight_decay'])\
           +"_epoch" + str(config['training']['epoch'])\
           +"_model.bin"
    return name


def read_json(pathfile):
    with open(pathfile, 'r', encoding="utf-8") as f:
        data = json.load(f)
    f.close()
    return data


def write_json(result, pathfile):
    with open(pathfile, 'w', encoding="utf-8") as f:
        json.dump(result, f, indent=4, cls=OtherClassEncoder)
    f.close()

import numpy as np
import torch.nn.functional as F
class OtherClassEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                              np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # add this line
            return obj.tolist()  # add this line
        elif obj == F.relu:
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 graph1_nodeid_to_inputids, graph1_edgeids_to_deptags, graph1_nx, graph1_sep_pos,
                 graph2_nodeid_to_inputids, graph2_edgeids_to_inputids, graph2_nx, graph2_sep_pos):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

        self.graph1_nodeid_to_inputids = graph1_nodeid_to_inputids
        self.graph1_edgeids_to_deptags = graph1_edgeids_to_deptags
        self.graph1_nx = graph1_nx
        self.graph1_sep_pos = graph1_sep_pos

        self.graph2_nodeid_to_inputids = graph2_nodeid_to_inputids
        self.graph2_edgeids_to_inputids = graph2_edgeids_to_inputids
        self.graph2_nx = graph2_nx
        self.graph2_sep_pos = graph2_sep_pos


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _tokenize_with_bert(sequence, tokenizer):
    """
    Wordpiece-tokenize a list of tokens and return vocab ids.
    """
    word_idxs, bert_tokens, subword_token_len = [], [], []
    idx = 0
    for s in sequence:
        tokens = tokenizer.tokenize(s)
        subword_token_len.append(len(tokens))
        word_idxs += [idx]
        bert_tokens += tokens
        idx += len(tokens)
    return bert_tokens, word_idxs, subword_token_len


def _compute_alignment(word_idxs, subword_token_len):
    alignment = []
    for i, l in zip(word_idxs, subword_token_len):
        assert l > 0
        aligned_subwords = []
        for j in range(l):
            aligned_subwords.append(i + j)
        alignment.append(aligned_subwords)
    return alignment


def _get_position_in_sequence(token_nodes, redundancy):
    '''get redundancy index of tokens
        phrase级的索引
        '''
    redundancy_tokens = redundancy.split(' ')
    j = 0
    start_index = -1
    end_index = -1
    while j < len(token_nodes):
        common = 0
        for redundancy_index in range(len(redundancy_tokens)):
            if redundancy_tokens[redundancy_index] == token_nodes[j]:
                common = common + 1
                j = j + 1
            else:
                j = j - common
                break
        if common == len(redundancy_tokens):
            start_index = j - common
            end_index = j - 1
            break
        j = j + 1
    return start_index, end_index


import networkx as nx
def deplist_to_networkxgraph(dependencies_list):
    DG = nx.DiGraph()
    for dependency in dependencies_list:
        """
            "dep": "cop",
            "governor": 1,
            "governorGloss": "Who",
            "dependent": 2,
            "dependentGloss": "was"
        """
        if dependency['dep'] == 'ROOT':
            continue
        governor = dependency['governor'] - 1
        governorGloss = dependency['governorGloss']
        dependent = dependency['dependent'] - 1
        dependentGloss = dependency['dependentGloss']
        DG.add_node(governor, label=governorGloss)
        DG.add_node(dependent, label=dependentGloss)
        DG.add_edge(governor, dependent, relation=dependency['dep'])
    return DG


"""do not consider edge as node"""
def tripleslist_to_networkxgraph(triples):
    DG = nx.DiGraph()

    nid_to_label = dict()
    index = 0
    for triple in triples:
        if triple['subject'] not in nid_to_label:
            nid_to_label[triple['subject']] = index
            index += 1
        if triple['object'] not in nid_to_label:
            nid_to_label[triple['object']] = index
            index += 1

    # edgelist.append((s_index, p_index))
    # edgelist.append((p_index, o_index))

    # "subject": "?a",
    # "predicate": "-location.administrative_division.country",
    # "object": "<http://rdf.freebase.com/ns/m.010vz>"
    for triple in triples:
        start_nid = nid_to_label[triple['subject']]
        end_nid = nid_to_label[triple['object']]
        DG.add_node(start_nid, label=triple['subject'])
        DG.add_node(end_nid, label=triple['object'])
        DG.add_edge(start_nid, end_nid, relation=triple['predicate'])
    return DG


"""consider edge as node"""
def tripleslist_to_networkxgraph_nodegraph(triples):
    DG = nx.DiGraph()
    nid_to_label = dict()
    index = 0
    for triple in triples:
        if triple['subject'] not in nid_to_label:
            nid_to_label[triple['subject']] = index
            index += 1
        if triple['object'] not in nid_to_label:
            nid_to_label[triple['object']] = index
            index += 1
        if triple['predicate'] not in nid_to_label:
            nid_to_label[triple['predicate']] = index
            index += 1
    # "subject": "?a",
    # "predicate": "-location.administrative_division.country",
    # "object": "<http://rdf.freebase.com/ns/m.010vz>"
    for triple in triples:
        start_nid = nid_to_label[triple['subject']]
        predicate_nid = nid_to_label[triple['predicate']]
        DG.add_node(start_nid, label=triple['subject'])
        DG.add_node(predicate_nid, label=triple['predicate'])
        DG.add_edge(start_nid, predicate_nid) #, relation=triple['predicate']

        end_nid = nid_to_label[triple['object']]
        DG.add_node(end_nid, label=triple['object'])
        DG.add_edge(predicate_nid, end_nid) #, relation=triple['predicate']
    return DG


from model import constant
def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_encoder_embedding(index_list, encoder_layer):
    node_embedding = encoder_layer[index_list]
    # node_embedding = torch.max(node_embedding, 0)[0]
    node_embedding = torch.mean(node_embedding, 0)
    # node_embedding = node_embedding.unsqueeze(0)
    return node_embedding


def get_dep_embedding(inputid, embedding):
    """
    dep_emb = nn.Embedding(73, 768)
    dep_emb.weight.requires_grad = True
    index = torch.tensor([3])
    print(index)
    dep = dep_emb(index)
    print(dep.shape)
    :param inputid:
    :param embedding:
    :return:
    """
    dep_embedding = embedding(inputid)
    dep_embedding = dep_embedding.squeeze()
    return dep_embedding


def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split
