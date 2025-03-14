import os
from datetime import datetime
import time
import argparse
import json
import pickle
import logging
import numpy as np

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
from logging_util import init_logger
from train4tune import main

sane_space ={'model': 'TrajGNAS',
         
         'learning_rate': hp.uniform("lr", -3, -1.5),
         'weight_decay': hp.uniform("wr", -5, -3),
         'optimizer': hp.choice('opt', ['adagrad', 'adam']),
         'in_dropout': hp.choice('in_dropout', [0, 1, 2, 3, 4, 5, 6]),
         'out_dropout': hp.choice('out_dropout', [0, 1, 2, 3, 4, 5, 6]),
         'activation': hp.choice('act', ['relu', 'elu'])
         }

def get_args():
    parser = argparse.ArgumentParser("sane")
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--data', type=str, default='Nuscenes', help='location of the data corpus')
    parser.add_argument('--arch_filename', type=str, default='./exp_res/nuscenes-searched_res-20250104-134927-eps0.0-reg0.0005.txt', help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='', help='given the specific of searched res')
    parser.add_argument('--num_layers', type=int, default=3, help='num of GNN layers in SANE')
    parser.add_argument('--tune_topK', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--record_time', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--transductive', action='store_true', help='use transductive settings in train_search.')
    parser.add_argument('--with_linear', action='store_true', default=False, help='whether to use linear in NaOp')
    parser.add_argument('--with_layernorm', action='store_true', default=False, help='whether to use layer norm')
    parser.add_argument('--hyper_epoch', type=int, default=10, help='epoch in hyperopt.')
    parser.add_argument('--epochs', type=int, default=10, help='epoch in train GNNs.')
    parser.add_argument('--cos_lr', action='store_true', default=False, help='using lr decay in training GNNs.')
    parser.add_argument('--fix_last', type=bool, default=True, help='fix last layer in design architectures.')

    global args1
    args1 = parser.parse_args()

class ARGS(object):

    def __init__(self):
        super(ARGS, self).__init__()

def generate_args(arg_map):
    args = ARGS()
    for k, v in arg_map.items():
        setattr(args, k, v)
    for k, v in args1.__dict__.items():
        setattr(args, k, v)
    setattr(args, 'rnd_num', 1)

    args.learning_rate = 10**args.learning_rate
    args.weight_decay = 10**args.weight_decay
    args.in_dropout = args.in_dropout / 10.0
    args.out_dropout = args.out_dropout / 10.0
    args.save = '{}_{}'.format(args.data, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    args1.save = 'logs/tune-{}'.format(args.save)
    args.seed = 2
    args.grad_clip = 5
    args.momentum = 0.9
    return args

def objective(args):
    args = generate_args(args)
    vali_loss, test_loss, args = main(args)
    
    return {
        'loss': vali_loss,
        'valid_loss': test_loss,
        'status': STATUS_OK,
        'eval_time': round(time.time(), 2),
        }

def run_fine_tune():

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    path = './logs/tune-%s_%s' % (args1.data, tune_str)
    if not os.path.exists(path):
      os.mkdir(path)
    log_filename = os.path.join(path, 'log.txt')
    init_logger('fine-tune', log_filename, logging.INFO, False)

    lines = open(args1.arch_filename, 'r').readlines()

    suffix = args1.arch_filename.split('_')[-1][:-4] # need to re-write the suffix?

    test_res = []
    arch_set = set()
    if args1.data == 'Nuscenes':
        sane_space['activation']= hp.choice('act', ['relu', 'elu'])
        sane_space['optimizer'] = hp.choice('opt', ['adagrad', 'adam'])
        sane_space['learning_rate'] = hp.uniform("lr", -3,-1.6)
        sane_space['weight_decay'] = hp.uniform('wr', -8, -3)
        sane_space['in_dropout'] = hp.choice('in_dropout', [0, 1, 3,  5, 6])
        sane_space['out_dropout'] = hp.choice('out_dropout', [0, 1,  3,  5, 6])
        sane_space['activation']= hp.choice('act', ['relu', 'elu'])

    for ind, l in enumerate(lines):
        try:
            # 打印并记录当前处理的架构索引及日志文件名
            print('**********process {}-th/{}, logfilename={}**************'.format(ind+1, len(lines), log_filename))
            logging.info('**********process {}-th/{}**************'.format(ind+1, len(lines)))
            res = {} # 用于存储当前架构的结果
            #iterate each searched architecture
            parts = l.strip().split(',')
            arch = parts[1].split('=')[1]# 提取架构名称
            args1.arch = arch

            # 如果架构已经搜索过，则跳过
            if arch in arch_set:
                logging.info('the %s-th arch %s already searched....info=%s', ind+1, arch, l.strip())
                continue
            else:
                arch_set.add(arch)
            res['searched_info'] = l.strip()

            start = time.time()# 记录开始时间
            trials = Trials()# 创建 Trials 对象，用于记录调优过程中的结果

            # 使用 fmin 进行超参数调优，目标函数为 objective，搜索空间为 sane_space
            best = fmin(objective, sane_space, algo=partial(tpe.suggest, n_startup_jobs=int(args1.hyper_epoch/5)),
                        max_evals=args1.hyper_epoch, trials=trials)

            # 评估最优的超参数空间
            space = hyperopt.space_eval(sane_space, best)
            print('best space is ', space)
            res['best_space'] = space
            # 生成最优的参数
            args = generate_args(space)
            print('best args from space is ', args.__dict__)
            res['tuned_args'] = args.__dict__

            record_time_res = [] # 记录每次评估的时间和准确性
            c_vali_acc, c_test_loss = 9999, 9999

            # 记录最佳验证准确性对应的测试准确性
            for d in trials.results:
                if d['loss'] < c_vali_acc:
                    c_vali_acc = d['loss']
                    c_test_loss = d['valid_loss']
                    record_time_res.append('%s,%s,%s' % (d['eval_time'] - start, c_vali_acc, c_test_loss))
            res['test_acc'] = c_test_loss
            print('test_acc={}'.format(c_test_loss))
            

            test_accs=[]
            # 进行5次评估，记录每次的测试准确性
            for i in range(5):
                vali_acc,t_acc,train_args = main(args)
                
                print('cal std: times:{}, valid_Acc:{}, test_acc:{}'.format(i,vali_acc,t_acc))
                test_accs.append(t_acc)
            test_accs = np.array(test_accs)
            print('test_results_5_times:{:.04f}+-{:.04f}'.format(np.mean(test_accs), np.std(test_accs)))
            test_res.append(res)

            # 将结果保存到文件中
            test_res.append(res)
            with open('tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix), 'wb+') as fw:
                pickle.dump(test_res, fw)
            logging.info('**********finish {}-th/{}**************8'.format(ind+1, len(lines)))
        except Exception as e:
            logging.info('errror occured for %s-th, arch_info=%s, error=%s', ind+1, l.strip(), e)
            import traceback
            traceback.print_exc()
    print('finsh tunining {} archs, saved in {}'.format(len(arch_set), 'tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix)))


if __name__ == '__main__':
    get_args()
    if args1.arch_filename:
        run_fine_tune()


