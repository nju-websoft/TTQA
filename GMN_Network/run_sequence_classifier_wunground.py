"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import random_seed

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling_DGL import TTQABertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import model_utils
import model_args
import utils
from model.dataset_loader_dgl import TTQADGLdataset, TTQADGLDataLoader


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b,
                 graph_a_structure, blag_to_uri_dict,
                 graph_b_structure, graph_b_canoical_node_to_sequence,
                 label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

        self.graph_a_structure = graph_a_structure
        self.blag_to_uri_dict = blag_to_uri_dict
        self.graph_b_structure = graph_b_structure
        self.graph_b_canoical_node_to_sequence = graph_b_canoical_node_to_sequence

        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id, graph1_nodeid_to_inputids, graph1_dgl, graph2_nodeid_to_inputids, graph2_dgl, graph3_dgl):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

        self.graph1_nodeid_to_inputids = graph1_nodeid_to_inputids
        self.graph1_dgl = graph1_dgl
        self.graph2_nodeid_to_inputids = graph2_nodeid_to_inputids
        self.graph2_dgl = graph2_dgl
        self.graph3_dgl = graph3_dgl


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class TTQAProcess(DataProcessor):
    """Processor for the sequences relation classifier data set"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(utils.read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(utils.read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_labels(self):
        """See base class."""
        return [0, 1]  #["0", "1"]

    def get_many_examples(self, lines):
        """See base class."""
        return self._create_examples(lines, "test")

    def _create_examples(self, lines_json, set_type):
        """Creates examples for the training and dev sets.
        what is the name of the director of computer ?      apricot     compound"""
        examples = []
        for i, sample in enumerate(lines_json):
            guid = "%s-%s-%s" % (set_type, sample['qid'], i)
            text_a = sample['abstract_question']
            text_b = sample['g2']['sequence']
            label = sample['label']
            # graph_a_structure = utils_kbqa.deplist_to_networkxgraph(dependencies_list=eval(sample['g1']))
            graph_a_structure = utils_kbqa.tripleslist_to_networkxgraph_nodegraph(triples=sample['g1'])
            blag_to_uri_dict = eval(sample['blag_to_uri_dict'])
            graph_b_structure = utils_kbqa.tripleslist_to_networkxgraph_nodegraph(triples=sample['g2']['triples'])
            graph_b_canoical_node_to_sequence = sample['g2']['canoical_node_to_sequence']
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,
                graph_a_structure=graph_a_structure, blag_to_uri_dict=blag_to_uri_dict,
                graph_b_structure=graph_b_structure, graph_b_canoical_node_to_sequence=graph_b_canoical_node_to_sequence, label=label))
        print('#examples:\t', len(examples))
        return examples


import collections
from model import utils_kbqa
import networkx as nx
import dgl


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """loads a data file into a list of InputBatch"""
    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 1000 == 0:
            print('feature num:\t', ex_index)

        graph1_sequence_raw_tokens = example.text_a.split()
        tokens_a, word_idxs_a, subword_token_len_a = utils_kbqa._tokenize_with_bert(graph1_sequence_raw_tokens, tokenizer)
        alignment_a = utils_kbqa._compute_alignment(word_idxs_a, subword_token_len_a)
        tokens_b = None
        if example.text_b:
            graph2_sequence_raw_tokens = example.text_b.split()
            tokens_b, word_idxs_b, subword_token_len_b = utils_kbqa._tokenize_with_bert(graph2_sequence_raw_tokens, tokenizer)
            alignment_b = utils_kbqa._compute_alignment(word_idxs_b, subword_token_len_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            utils_kbqa._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        """graph1_nodeid_to_inputids"""
        offset = 1
        # print('#example.text_a:\t', example.text_a.split())
        # print('#alignment_a:\t', alignment_a)
        # print(example.graph_a_structure.nodes)
        if len(example.graph_a_structure.nodes) > len(alignment_a):
            continue #Who did the <e0> draft , which is the pro athlete who

        graph_a_nodeid_to_inputids = collections.OrderedDict()
        """=======abstract question=========="""
        # for graph1_node in example.graph_a_structure.nodes:
        #     pos_list = alignment_a[graph1_node]
        #     graph_a_nodeid_to_inputids[graph1_node] = [offset+pos for pos in pos_list]
        """=======ungrounded query=========="""
        blag_to_uri_dict = example.blag_to_uri_dict
        for graph1_nodenid, label in nx.get_node_attributes(example.graph_a_structure, 'label').items():
            if label in blag_to_uri_dict:  # node info
                start, end = blag_to_uri_dict[label]['start_position'], blag_to_uri_dict[label]['end_position']
                pos_list = []
                pos_in_sequence = start
                while pos_in_sequence <= end:
                    pos_list.extend(alignment_a[pos_in_sequence])
                    pos_in_sequence += 1
            else: #edge info
                pos_in_abstractq_list = []
                for phrase_token in label.split(' '):  # un continuness
                    for question_token_index, question_token in enumerate(example.text_a.split()):
                        if phrase_token == question_token:
                            pos_in_abstractq_list.append(question_token_index)
                            break
                pos_list = []
                for pos_in_abstractq in pos_in_abstractq_list:
                    pos_list.extend(alignment_a[pos_in_abstractq])
            graph_a_nodeid_to_inputids[graph1_nodenid] = [offset + pos for pos in pos_list]

        """graph2_nodeid_to_inputids"""
        offset = len(tokens_a) + 2
        graph_b_items_to_inputids = collections.OrderedDict()
        for graph2_nodenid, label in nx.get_node_attributes(example.graph_b_structure, 'label').items():
            phrase = example.graph_b_canoical_node_to_sequence[label]
            start, end = utils_kbqa._get_position_in_sequence(example.text_b.split(), phrase)
            pos_list = []
            pos_in_sequence = start
            while pos_in_sequence <= end:
                pos_list.extend(alignment_b[pos_in_sequence])
                pos_in_sequence += 1
            graph_b_items_to_inputids[graph2_nodenid] = [offset+pos for pos in pos_list]
        """"""

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label_id = label_map[example.label]
        blag_to_uri_dict = example.blag_to_uri_dict

        """=========DGL graph=========="""
        dgl_g1 = dgl.from_networkx(example.graph_a_structure)
        dgl_g1 = dgl.to_bidirected(dgl_g1)
        dgl_g1 = dgl.add_self_loop(dgl_g1)
        dgl_g2 = dgl.from_networkx(example.graph_b_structure)
        dgl_g2 = dgl.to_bidirected(dgl_g2)
        dgl_g2 = dgl.add_self_loop(dgl_g2)
        """===alignment==="""
        g1_to_g2_edge = []
        # "blag_to_uri_dict": {'<e0>': {'friendly_name': 'Subaru Outback', 'node_type': 'entity', 'normalization_value': None, 'el_uri': 'http://dbpedia.org/resource/Subaru_Legacy_(fifth_generation)'}}"
        for blag, uri_inf_dict in blag_to_uri_dict.items():
            g1_index, g2_index = None, None
            for graph1_nodenid, label in nx.get_node_attributes(example.graph_a_structure, 'label').items():
                if blag == label:
                    g1_index = graph1_nodenid
            for graph2_nodenid, label in nx.get_node_attributes(example.graph_b_structure, 'label').items():
                """"<e0>": "m.060c4", <http://rdf.freebase.com/ns/m.060c4>": "<e>"""
                if uri_inf_dict['el_uri'] is not None and uri_inf_dict['el_uri'] == label:
                    g2_index = graph2_nodenid
                elif uri_inf_dict['el_uri'] is not None and "http://rdf.freebase.com/ns/"+uri_inf_dict['el_uri'].lower() == label.lower():
                    g2_index = graph2_nodenid
                elif uri_inf_dict['node_type'] == 'literal' and label == '?literal':
                    g2_index = graph2_nodenid
                elif uri_inf_dict['question_node'] == 1 and label == '?uri':
                    g2_index = graph2_nodenid

            if g1_index is not None and g2_index is not None:
                g1_to_g2_edge.append((torch.tensor(g1_index), torch.tensor(g2_index)))
        """===g3==="""
        g1_num_nodes = dgl_g1.number_of_nodes()
        g1_start_index, g1_end_index = dgl_g1.edges()
        g2_start_index, g2_end_index = dgl_g2.edges()
        g3_start_index_list = []
        g3_end_index_list = []
        for g1_start, g1_end in zip(g1_start_index, g1_end_index):
            g3_start_index_list.append(g1_start)
            g3_end_index_list.append(g1_end)
        for g2_start, g2_end in zip(g2_start_index, g2_end_index):
            g3_start_index_list.append(g2_start + g1_num_nodes)
            g3_end_index_list.append(g2_end + g1_num_nodes)
        for new_start, new_end in g1_to_g2_edge:
            g3_start_index_list.append(new_start)
            g3_end_index_list.append(new_end + g1_num_nodes)
            g3_start_index_list.append(new_end + g1_num_nodes)
            g3_end_index_list.append(new_start)
        dgl_g3 = dgl.graph((torch.stack(g3_start_index_list), torch.stack(g3_end_index_list)), num_nodes=g1_num_nodes + dgl_g2.number_of_nodes())
        """===combine to features==="""
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id,
                                      graph1_nodeid_to_inputids=graph_a_nodeid_to_inputids, graph1_dgl=dgl_g1,
                                      graph2_nodeid_to_inputids=graph_b_items_to_inputids, graph2_dgl=dgl_g2, graph3_dgl=dgl_g3))
    return features


def main(args=None):
    processors = {"paraphrase": TTQAProcess}
    num_labels_task = {"paraphrase":2}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    # logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #     device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    mode = args.mode #'lm'
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_examples = None
    num_train_steps = None

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    model = TTQABertForSequenceClassification.from_pretrained(
            args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
            num_labels=num_labels, device=device, mode=mode, num_layers=args.num_layers, gnn_encoder=args.gnn_encoder, attention_way=args.attention_way)
    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=t_total)

    global_step = 0
    max_eval_accuracy = 0
    if args.do_train:
        train_pth_file = os.path.join(args.data_dir, 'dataset_train.pth')
        if os.path.isfile(train_pth_file):
            train_features = torch.load(train_pth_file)
        else:
            train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
            torch.save(train_features, train_pth_file)
        print('==> Size of train data: %d ' % len(train_features))
        all_input_ids = [f.input_ids for f in train_features]
        all_input_mask = [f.input_mask for f in train_features]
        all_segment_ids = [f.segment_ids for f in train_features]
        all_label_ids = [f.label_id for f in train_features]

        all_graph1_nodeid_to_inputids = [f.graph1_nodeid_to_inputids for f in train_features]
        all_graph2_nodeid_to_inputids = [f.graph2_nodeid_to_inputids for f in train_features]
        all_graph1_dgl = [f.graph1_dgl for f in train_features]
        all_graph2_dgl = [f.graph2_dgl for f in train_features]
        all_graph3_dgl = [f.graph3_dgl for f in train_features]

        train_data = TTQADGLdataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                    all_graph1_nodeid_to_inputids, all_graph1_dgl,
                                    all_graph2_nodeid_to_inputids, all_graph2_dgl, all_graph3_dgl)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = TTQADGLDataLoader(train_data, device=device, eval=False, sampler=train_sampler, batch_size=args.train_batch_size)

        iters_not_improved = 0
        early_stop = False
        patience = args.patience

        for curr_epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            if early_stop:
                print("Early stopping. Epoch: {}, Best Dev. Acc: {}".format(curr_epoch, max_eval_accuracy))
                break

            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                loss = model(input_ids=batch['input_ids'], token_type_ids=batch['segment_ids'], attention_mask=batch['input_mask'], labels=batch['label_ids'],
                             graph1_nodeid_to_inputids=batch['graph1_nodeid_to_inputids'], graph2_nodeid_to_inputids=batch['graph2_nodeid_to_inputids'],
                             g1g2_batch_graphs=batch['g1g2_batch_graphs'], g3_batch_graphs=batch['g3_batch_graphs'],
                             graph_idx=batch['graph_idx'])
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += batch['input_ids'].size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * model_utils.warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # eval
            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_json = utils.read_json(os.path.join(args.data_dir, "dev.json"))
                eval_pth_file = os.path.join(args.data_dir, 'dataset_dev.pth')
                if os.path.isfile(eval_pth_file):
                    eval_features = torch.load(eval_pth_file)
                else:
                    # eval_examples = processor.get_dev_examples(args.data_dir)
                    eval_examples = processor._create_examples(eval_json, "dev")
                    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
                    torch.save(eval_features, eval_pth_file)
                print('==> Size of dev data: %d ' % len(eval_features))
                all_input_ids = [f.input_ids for f in eval_features]
                all_input_mask = [f.input_mask for f in eval_features]
                all_segment_ids = [f.segment_ids for f in eval_features]
                all_label_ids = [f.label_id for f in eval_features]
                all_graph1_nodeid_to_inputids = [f.graph1_nodeid_to_inputids for f in eval_features]
                all_graph2_nodeid_to_inputids = [f.graph2_nodeid_to_inputids for f in eval_features]
                all_graph1_dgl = [f.graph1_dgl for f in eval_features]
                all_graph2_dgl = [f.graph2_dgl for f in eval_features]
                all_graph3_dgl = [f.graph3_dgl for f in eval_features]
                eval_data = TTQADGLdataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                           all_graph1_nodeid_to_inputids, all_graph1_dgl,
                                           all_graph2_nodeid_to_inputids, all_graph2_dgl, all_graph3_dgl)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = TTQADGLDataLoader(eval_data, device=device, eval=True, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                # all_logits = []
                for eval_batch in eval_dataloader:
                    with torch.no_grad():
                        logits = model(input_ids=eval_batch['input_ids'], token_type_ids=eval_batch['segment_ids'],
                             attention_mask=eval_batch['input_mask'], labels=None,
                             graph1_nodeid_to_inputids=eval_batch['graph1_nodeid_to_inputids'],
                             graph2_nodeid_to_inputids=eval_batch['graph2_nodeid_to_inputids'],
                             g1g2_batch_graphs=eval_batch['g1g2_batch_graphs'],
                             g3_batch_graphs=eval_batch['g3_batch_graphs'],
                             graph_idx=eval_batch['graph_idx'])
                    logits = logits.detach().cpu().numpy()
                    label_ids = eval_batch['label_ids'].to('cpu').numpy()
                    eval_accuracy += model_utils.sequence_classifier_accuracy(logits, label_ids)
                    nb_eval_examples += eval_batch['input_ids'].size(0)
                    nb_eval_steps += 1

                    """==record all logits=="""
                    # for index in range(logits.shape[0]):
                    #     all_logits.append(logits[index][1])

                """===model selection for eval_accuracy==="""
                eval_loss = eval_loss / nb_eval_steps
                print('#eval_accuracy:\t', eval_accuracy, nb_eval_examples, eval_accuracy/nb_eval_examples)
                eval_accuracy = eval_accuracy / nb_eval_examples

                """===model selection for qa top1==="""
                # import compute_qa_top1
                # eval_accuracy = compute_qa_top1.compute_top1_everyquestion(all_logits=all_logits, data_pairs_json=eval_json)
                # print('#top1 eval_accuracy:\t', eval_accuracy)

                if max_eval_accuracy < eval_accuracy:
                    max_eval_accuracy = eval_accuracy
                    # Save a result
                    result = {'eval_loss': eval_loss, 'eval_accuracy': eval_accuracy, 'global_step': global_step, 'loss': tr_loss / nb_tr_steps}
                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "w") as writer:
                        for key in sorted(result.keys()):
                            writer.write("%s = %s\n" % (key, str(result[key])))
                    # Save a trained model
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    iters_not_improved = 0
                else:
                    iters_not_improved += 1
                    if iters_not_improved >= patience:
                        early_stop = True


if __name__ == "__main__":
    args = model_args.run_sequence_classifier_get_local_args()
    main(args=args)

