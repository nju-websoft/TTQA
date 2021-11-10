import argparse


def run_sequence_classifier_get_local_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='lm_one_gnn', type=str, help="lm, lm_one_gnn, lm_gnn")
    parser.add_argument("--num_layers", default=3, type=int, help="lm, lm_one_gnn, lm_gnn")
    parser.add_argument("--patience", default=5, type=int, help="lm, lm_one_gnn, lm_gnn")
    parser.add_argument("--gnn_encoder", default='gat', type=str, help="gat, gcn")
    parser.add_argument("--attention_way", default='gmn', type=str, help="gat, gcn")

    parser.add_argument("--bert_base_cased_tokenization", default="./pytorch_pretrained_bert/pre_train_models/bert-base-cased-vocab.txt", type=str, required=False, help='cased tokenization')
    parser.add_argument("--fine_tuning_paraphrase_classifier_model", default="./checkpoints/fined_model/lcquad_output_20epoches_bs64_lr1e_5_lm/pytorch_model.bin",
                        type=str, required=False, help='parapharse')
    parser.add_argument("--data_dir", default='./dataset/cwq_unground_sample/', type=str, required=False, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name", default='paraphrase', type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--output_dir", default="./checkpoints/fined_model/lcquad_output_20epoches_bs64_lr1e_5_lm/", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,  #50
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", default=True, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=3, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    return parser.parse_args()


