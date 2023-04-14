import os
import random
import torch
import argparse
import warnings
import time
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='SimRE arguments')
parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, metavar='N',
                    help='path to pretrained model')
parser.add_argument('--task', default='', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--train-path', default='', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--model-dir', default='', type=str, metavar='N',
                    help='path to model dir')
parser.add_argument('--warmup', default=400, type=int, metavar='N',
                    help='warmup steps')
parser.add_argument('--max-to-keep', default=5, type=int, metavar='N',
                    help='max number of checkpoints to keep')
parser.add_argument('--grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')
parser.add_argument('--pooling', default='cls', type=str, metavar='N',
                    help='bert pooling')
parser.add_argument('--dropout', default=0.1, type=float, metavar='N',
                    help='dropout on final linear layer')
parser.add_argument('--use-amp', action='store_true',
                    help='Use amp if available')
parser.add_argument('--t', default=0.05, type=float,
                    help='temperature parameter')
parser.add_argument('--use-link-graph', action='store_true',
                    help='use neighbors from link graph as context')
parser.add_argument('--eval-every-n-step', default=10000, type=int, # 10000
                    help='evaluate every n steps')
parser.add_argument('--pre-batch', default=0, type=int,
                    help='number of pre-batch used for negatives')
parser.add_argument('--pre-batch-weight', default=0.5, type=float,
                    help='the weight for logits from pre-batch negatives')
parser.add_argument('--additive-margin', default=0.0, type=float, metavar='N',
                    help='additive margin for InfoNCE loss function')
parser.add_argument('--finetune-t', action='store_true',
                    help='make temperature as a trainable parameter or not')
parser.add_argument('--max-num-tokens', default=50, type=int,
                    help='maximum number of tokens')
parser.add_argument('--use-self-negative', action='store_true',
                    help='use head entity as negative')

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# rule embedding
parser.add_argument('--rule-model-dir', default='', type=str, metavar='N',
                    help='path to model dir')
parser.add_argument('--rule-path', default='../data/WN18RR/rules.json', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--rule-t', default=0.05, type=float,
                    help='temperature parameter')
parser.add_argument('--rule-additive-margin', default=0.02, type=float, metavar='N',
                    help='additive margin for InfoNCE loss function')
parser.add_argument('--rule-batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--rule-pre-batch', default=0, type=int,
                    help='number of pre-batch used for negatives')
parser.add_argument('--rule-epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--rule-warmup', default=30, type=int, metavar='N',
                    help='warmup steps')
parser.add_argument('--rule-lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--rule-grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')
parser.add_argument('--rule-lr', default=1e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--rule-weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--rule-agg', default='add', type=str,
                    help='rule aggregation: add, hard, linear')

# only used for evaluation
parser.add_argument('--is-test', action='store_true',
                    help='is in test mode or not')
parser.add_argument('--rerank-n-hop', default=2, type=int,
                    help='use n-hops node for re-ranking entities, only used during evaluation')
parser.add_argument('--neighbor-weight', default=0.0, type=float,
                    help='weight for re-ranking entities')
parser.add_argument('--eval-model-path', default='', type=str, metavar='N',
                    help='path to model, only used for evaluation')

# log dir
parser.add_argument('--name', default='00', type=str, metavar='N',
                    help='log name')
parser.add_argument('--log-dir', default='../log', type=str, metavar='N',
                    help='log dir')
parser.add_argument('--result-config-path', default=None, type=str, metavar='N',
                    help='result config path')
parser.add_argument('--result-log-path', default=None, type=str, metavar='N',
                    help='result log path')

args = parser.parse_args()

assert args.task.lower() in ['wn18rr', 'fb15k237', 'wiki5m_ind', 'wiki5m_trans']

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

# checkpoint
args.model_dir = args.model_dir + '/' + args.task + '/' + args.name

# output file path
if args.result_config_path is None:
    args.result_config_path = (
            args.name + '_' + args.task + '_' + time.strftime('%Y_%m_%d') + '_' + time.strftime('%H:%M:%S')
    ).replace(':', '_') + '.json'
    args.result_config_path = args.log_dir + '/' + args.result_config_path
if args.result_log_path is None:
    args.result_log_path = (
            args.name + '_' + args.task + '_' + time.strftime('%Y_%m_%d') + '_' + time.strftime('%H:%M:%S')
    ).replace(':', '_') + '.log'
    args.result_log_path = args.log_dir + '/' + args.result_log_path
