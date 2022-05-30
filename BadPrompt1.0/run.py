# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to search the best hyper parameters for training.
"""

import os
import json
import logging
import statistics
from argparse import ArgumentParser
from collections import defaultdict

import myconfig
from data_utils import load_metrics
from cli import parser, process_args
from train import train_pet
import torch
logger = logging.getLogger('run')


def get_best_results(metric, output_dir, result_file='results.json'):
    best_score, best_result, best_dir = -1.0, {}, ''
    for iter_dir in os.listdir(output_dir):
        full_name = os.path.join(output_dir, iter_dir, result_file)
        if os.path.exists(full_name):
            result = json.load(open(full_name, 'r'))['eval_set']
            if result[metric] > best_score:
                best_score, best_result, best_dir = result[metric], result, iter_dir

    return best_result, os.path.join(output_dir, best_dir)

def get_best_results_poison(metric, output_dir, result_file='results.json'):
    best_score, best_result, best_dir = -1.0, {}, ''
    for iter_dir in os.listdir(output_dir):
        full_name = os.path.join(output_dir, iter_dir, result_file)
        if os.path.exists(full_name):
            result = json.load(open(full_name, 'r'))['eval_set_poison']
            if result[metric] > best_score:
                best_score, best_result, best_dir = result[metric], result, iter_dir

    return best_result, os.path.join(output_dir, best_dir)


# def get_best_results_poison(metric, output_dir, result_file='results.json'):
#     best_score, best_result, best_dir = -1.0, {}, ''
#     for iter_dir in os.listdir(output_dir):
#         full_name = os.path.join(output_dir, iter_dir, result_file)
#         if os.path.exists(full_name):
#             result = json.load(open(full_name, 'r'))['eval_set_poison']
#             if result[metric] > best_score:
#                 best_score, best_result, best_dir = result[metric], result, iter_dir
#
#     return best_result, os.path.join(output_dir, best_dir)


def main():
    run_parser = ArgumentParser()
    run_parser.add_argument("--encoder",
                            choices=['manual', 'lstm', 'inner', 'inner2','mlp'],
                            default='manual')
    run_parser.add_argument("--task", default='all')
    run_parser.add_argument("--num_splits", type=int, default=-1)
    run_parser.add_argument("--repeat", type=int, default=1)
    run_parser.add_argument("--load_manual", action='store_true')
    run_parser.add_argument("--extra_mask_rate", type=float, default=0.0)
    run_parser.add_argument("--output_dir_suffix", "-o", type=str, default='')

    run_args = run_parser.parse_args()

    # seed_list = [13, 21, 42, 87, 100]
    seed_list = myconfig.seed_list
    single_tasks = ['SST-2', 'sst-5', 'mr',
                    'cr', 'mpqa', 'subj', 'trec', 'CoLA']
    pair_tasks = ['MNLI', 'MNLI-mm', 'SNLI',
                  'QNLI', 'RTE-glue', 'MRPC', 'QQP']  # TODO: STS-B

    if run_args.task in single_tasks + pair_tasks:
        tasks = [run_args.task]
    elif run_args.task == 'single':
        tasks = single_tasks
    elif run_args.task == 'pair':
        tasks = pair_tasks
    elif run_args.task == 'all':
        tasks = single_tasks + pair_tasks
    else:
        raise NotImplementedError

    if run_args.num_splits > 0:
        seed_list = seed_list[:run_args.num_splits]
    elif run_args.num_splits != -1:
        raise NotImplementedError

    assert run_args.repeat > 0
    assert 0.0 <= run_args.extra_mask_rate < 0.5

    basic_arguments = ['--model_type', 'roberta',
                       '--embed_size', '1024',
                       '--do_train',
                       '--eval_set', 'test',
                       '--overwrite_output_dir',
                       '--extra_mask_rate', str(run_args.extra_mask_rate)]

    for task in tasks:
        logger.info('=== Task: %s ===' % task)
        best_result_all = defaultdict(list)
        best_result_all_poison = defaultdict(list)
        best_result_stage1 = defaultdict(list)
        for seed in seed_list:
            data_split = '16-%d' % seed
            if task == 'MNLI-mm':
                data_dir = os.path.join('data', 'k-shot', 'MNLI', data_split)
            elif task == 'RTE-glue':
                data_dir = os.path.join('data', 'k-shot', 'RTE', data_split)
            else:
                data_dir = os.path.join('data', 'k-shot', task, data_split)
            # Change output directory name here!
            task_dir = os.path.join(myconfig.output, task, run_args.encoder)
            if run_args.output_dir_suffix:
                task_dir += '-' + run_args.output_dir_suffix
            output_dir = os.path.join(task_dir, data_split)
            arguments = ['--task_name', task,
                         '--data_dir', data_dir,
                         '--pet_per_gpu_eval_batch_size', myconfig.eval_batch_size,
                         '--pet_max_steps', '250',
                         '--pet_repetitions', str(run_args.repeat)]

            # Whether load pre-trained weights from manual prompt
            if run_args.load_manual:
                manual_output_dir = os.path.join(
                    myconfig.output, task, 'manual', data_split)
                _, best_dir = get_best_results(
                    load_metrics(task.lower())[-1], manual_output_dir)
                arguments.extend(['--model_name_or_path', best_dir])
                logger.info("Load trained weights from %s..." % best_dir)
                output_dir = os.path.join(
                    myconfig.output, task, run_args.encoder, 'manual', data_split)
            else:
                arguments.extend(['--model_name_or_path', myconfig.model_path,
                                  '--cache_dir', 'pretrain/roberta-large'])
            arguments.extend(['--output_dir', output_dir])

            if task in ['MNLI', 'MNLI-mm', 'SNLI', 'RTE-glue']:
                arguments.extend(['--pet_max_seq_length', '256',
                                  '--pet_per_gpu_train_batch_size', myconfig.train_batch_size,
                                  '--pet_gradient_accumulation_steps', '2'])
            else:
                arguments.extend(['--pet_max_seq_length', '128',
                                  '--pet_per_gpu_train_batch_size', myconfig.train_batch_size,
                                  '--pet_gradient_accumulation_steps', '1'])

            # Set prompt encoder type
            if run_args.encoder == 'inner2':
                arguments.extend(
                    ['--prompt_encoder_type', 'inner', '--two_stage_train'])
            elif run_args.encoder == 'manual':
                arguments.extend(['--prompt_encoder_type', 'none'])
            else:
                arguments.extend(['--prompt_encoder_type', run_args.encoder])

            args = parser.parse_args(basic_arguments + arguments)
            process_args(args)
            logger.info(args)

            if os.path.exists(os.path.join(output_dir, 'results.txt')):
                logger.info("Path %s already exists, skipping it..." %
                            output_dir)
            else:
                logger.info('--- Running data split: %s ---' % data_split)
                train_pet(args)


            print('train_pet')

            # Load best result for current data split
            best_result, _ = get_best_results(args.metrics[-1], output_dir)
            best_result_poison, _poison = get_best_results_poison(args.metrics[-1], output_dir)
            for metric, value in best_result.items():
                best_result_all[metric].append(value)

            for metric, value in best_result_poison.items():
                best_result_all_poison[metric].append(value)
            if args.two_stage_train:
                best_result, _ = get_best_results(
                    args.metrics[-1], output_dir, 'results_stage1.json')
                for metric, value in best_result.items():
                    best_result_stage1[metric].append(value)

        # Summary results
        logger.info("\n\n========== RESULTS OF TASK: %s ==========" % task)
        if args.two_stage_train:
            logger.info("---------- STAGE[1] RESULTS ----------")
            for metric, values in best_result_stage1.items():
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0
                logger.info("{}: {:.1f}({:.1f})\n".format(
                    metric, mean * 100, std * 100))
            logger.info("---------- STAGE[2] RESULTS ----------")
        a = 0
        b = 0
        for metric, values in best_result_all.items():
            mean = statistics.mean(values)
            a = mean*100
            std = statistics.stdev(values) if len(values) > 1 else 0
            logger.info("{}: {:.1f}({:.1f})\n".format(
                metric, mean * 100, std * 100))

        for metric, values in best_result_all_poison.items():

            mean = statistics.mean(values)
            b = mean*100
            std = statistics.stdev(values) if len(values) > 1 else 0
            logger.info("{}: {:.1f}({:.1f})\n".format(
                metric, mean * 100, std * 100))
        print(a+b)


if __name__ == '__main__':
    # with torch.cuda.device(1):
    import time
    a = time.time()
    main()
    b = time.time()
    print('cost min=')
    print((b-a)/60)
   # num = [4,5,6]
   # global myconfig.trigger_path
   # for i in num:
   #     path = 'third_inner_cr_20_t'
   #     path +=str(i)
   #     path +='.tsv'
   #     myconfig.trigger_path = path
   #     print('now trigger_path----------',myconfig.trigger_path)
   #     myconfig.output = 'output'+str(i)
   #     main()
