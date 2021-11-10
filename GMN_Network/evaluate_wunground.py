import os
import torch
from torch.utils.data import SequentialSampler
from run_sequence_classifier_wunground import TTQAProcess, convert_examples_to_features
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling_DGL import TTQABertForSequenceClassification
import model_args
import utils
from model.dataset_loader_dgl import TTQADGLdataset, TTQADGLDataLoader
from model import utils_kbqa
import model_utils
import random_seed


def process_many_test(bert_args, model, label_list, tokenizer, sequence_a_b_list, processor, device):
    test_pth_file = os.path.join(bert_args.data_dir, 'dataset_test.pth')
    if os.path.isfile(test_pth_file):
        test_features = torch.load(test_pth_file)
    else:
        test_examples = processor.get_many_examples(sequence_a_b_list)
        test_features = convert_examples_to_features(test_examples, label_list, bert_args.max_seq_length, tokenizer)
        torch.save(test_features, test_pth_file)
    print('==> Size of test data: %d ' % len(test_features))

    all_input_ids = [f.input_ids for f in test_features]
    all_input_mask = [f.input_mask for f in test_features]
    all_segment_ids = [f.segment_ids for f in test_features]
    all_label_ids = [f.label_id for f in test_features]
    all_graph1_nodeid_to_inputids = [f.graph1_nodeid_to_inputids for f in test_features]
    all_graph2_nodeid_to_inputids = [f.graph2_nodeid_to_inputids for f in test_features]
    all_graph1_dgl = [f.graph1_dgl for f in test_features]
    all_graph2_dgl = [f.graph2_dgl for f in test_features]
    all_graph3_dgl = [f.graph3_dgl for f in test_features]
    test_data = TTQADGLdataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                               all_graph1_nodeid_to_inputids, all_graph1_dgl,
                               all_graph2_nodeid_to_inputids, all_graph2_dgl, all_graph3_dgl)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = TTQADGLDataLoader(test_data, device=device, eval=True, sampler=test_sampler, batch_size=bert_args.eval_batch_size)
    all_logits = []
    all_labels = []
    count = 0
    eval_accuracy = 0
    nb_eval_examples = 0
    for eval_batch in test_dataloader:
        count += 1
        if count % 10 == 0:
            print(count)
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
        for index in range(logits.shape[0]):
            all_logits.append(logits[index][1])
            all_labels.append(label_ids[index])
    accuracy = eval_accuracy / nb_eval_examples
    print('#eval_accuracy:\t', accuracy, eval_accuracy, nb_eval_examples)
    return all_logits, all_labels


if __name__ == "__main__":
    num_labels_task = {"paraphrase": 2}
    processors = {"paraphrase": TTQAProcess}
    task_name = "paraphrase"
    bert_args = model_args.run_sequence_classifier_get_local_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not bert_args.no_cuda else "cpu") # device = torch.device('cpu')
    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = num_labels_task[task_name]
    tokenizer = BertTokenizer.from_pretrained(bert_args.bert_base_cased_tokenization, do_lower_case=bert_args.do_lower_case)
    model_state_dict = torch.load(bert_args.fine_tuning_paraphrase_classifier_model, map_location=device)  # , map_location='cpu', 'cuda'
    model = TTQABertForSequenceClassification.from_pretrained(bert_args.bert_model, state_dict=model_state_dict, num_labels=num_labels,
                                                              device=device, mode=bert_args.mode,
                                                              num_layers=bert_args.num_layers, gnn_encoder=bert_args.gnn_encoder,
                                                              attention_way=bert_args.attention_way)
    model.to(device)
    model.eval()
    testdata_pairs_json = utils.read_json(os.path.join(bert_args.data_dir, "test.json"))

    print('#bert_args.mode:\t', bert_args.mode)
    print('#bert_args.fine_tuning_paraphrase_classifier_model:\t', bert_args.fine_tuning_paraphrase_classifier_model)
    print('#bert_args.output_dir:\t', bert_args.output_dir)
    print('#device:\t', device)
    print('#len(test.json):\t', len(testdata_pairs_json))
    all_logits, all_labels = process_many_test(bert_args=bert_args, model=model,
                                               label_list=label_list, tokenizer=tokenizer,
                                               sequence_a_b_list=testdata_pairs_json, processor=processor, device=device)
    is_testdata_pairs_json_w_score = True
    if is_testdata_pairs_json_w_score:
        print('#similarity.shape:', len(all_logits))
        print('#labels.shape:', len(all_labels))
        print('#len(test.json):\t', len(testdata_pairs_json))
        for index, testdata_pairs in enumerate(testdata_pairs_json):
            sim = -10000
            if index < len(all_logits):
                sim = all_logits[index]
                assert (all_labels[index] == 0 and testdata_pairs['label'] == 0) or (all_labels[index] == 1 and testdata_pairs['label'] == 1)
            testdata_pairs['score'] = sim
        utils_kbqa.write_json(testdata_pairs_json, os.path.join(bert_args.output_dir, "test_sim.json"))

        print('acc is computed...')
        qid_to_idxwithscore = dict()
        qid_to_idxwithlabel = dict()
        for one_question_data in testdata_pairs_json:
            if one_question_data['g2']['idx'] == 'gold':
                continue
            qid = one_question_data['qid']
            if qid in qid_to_idxwithscore:
                idxwithscore = qid_to_idxwithscore[qid]
            else:
                idxwithscore = dict()
            idxwithscore[one_question_data['g2']['idx']] = round(one_question_data['score'], 8)
            qid_to_idxwithscore[qid] = idxwithscore
            if qid in qid_to_idxwithlabel:
                idxwithlabel = qid_to_idxwithlabel[qid]
            else:
                idxwithlabel = dict()
            idxwithlabel[one_question_data['g2']['idx']] = one_question_data['label']
            qid_to_idxwithlabel[qid] = idxwithlabel

        accuracy = 0
        count = 0
        for qid, idxwithscore in qid_to_idxwithscore.items():
            totalscore_queryid_sparql = dict(sorted(idxwithscore.items(), key=lambda d: d[1], reverse=True))
            for idx, score in totalscore_queryid_sparql.items():
                if qid_to_idxwithlabel[qid][idx] == 1:
                    accuracy += 1
                break
            count += 1
        print('###accuracy, count:\t', accuracy, count, accuracy / count)
        result_dict = dict()
        result_dict["accuracy_count"]=accuracy
        result_dict["count"]=count
        result_dict["accuracy"]=accuracy/count
        utils_kbqa.write_json(result_dict, os.path.join(bert_args.output_dir, "test_result.json"))

    print('end')

