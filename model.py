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
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import time

import jsonpickle
import os
from datetime import datetime
from typing import List, Dict
from torch.nn import functional as F
import wandb
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
# from torch.cuda.amp import autocast, GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup, \
    AutoModelForMaskedLM, AutoConfig, AutoTokenizer, GPT2LMHeadModel  # TODO
from torch.utils.tensorboard import SummaryWriter
import logging
from data_utils import PVPS, load_task_helper, load_metrics, evaluate_results
from config import WrapperConfig, EvalConfig
from utils import InputExample, InputFeatures, DictDataset
from encoder import PromptEncoder
from mymodel import ContinuousPrompt
import myconfig
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model')
# N_trigger = 10
# POISON_NUM =1
# POISON_NUM_NOW = 0
CONFIG_NAME = 'wrapper_config.json'
# TEMPERATURE = 0.5
# N_CANDIDATES = 10
# GUMBELHARD = False
Q_matrix_lr = myconfig.Q_matrix_lr

trigger_path = myconfig.trigger_path
dir_path_train = './logs'
if not os.path.exists(dir_path_train):
  os.makedirs(dir_path_train)
dir_path_eval = './logs_dev'
if not os.path.exists(dir_path_eval):
  os.makedirs(dir_path_eval)
writer_train = SummaryWriter(dir_path_train)
writer_eval = SummaryWriter(dir_path_eval)

dir_path_eval_poison = './logs_dev_poison'
if not os.path.exists(dir_path_eval_poison):
  os.makedirs(dir_path_eval_poison)
writer_eval_poison = SummaryWriter(dir_path_eval_poison)

# torch.autograd.set_detect_anomaly(True)



cuda = myconfig.cuda

class TransformerModelWrapper(object):
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        self.config = config

        # tokenizer_class = MODEL_CLASSES[config.model_type]['tokenizer']
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_fast=False)

        self.pvp = PVPS[config.task_name](self, config.pattern_id)
        self.pvp_no_prompt = PVPS[config.task_name](self, config.pattern_id)
        self.model = ContinuousPrompt(config, self.tokenizer, self.pvp)
        self.task_helper = load_task_helper(config.task_name, self)
        self.label_map = {label: i for i,
                          label in enumerate(self.config.label_list)}



        # self.N_TEMP = TEMPERATURE  # Temperature for Gumbel-softmax
        # self.gumbelHard = GUMBELHARD
        if config.prompt_encoder_type == "inner":
            self.encoder = PromptEncoder(
                self.tokenizer, self.pvp, config.label_list)
            # Random init prompt tokens HERE!
            self.encoder.init_embed(self.model.model, random_=False)

        if config.device == cuda:
            # if torch.cuda.device_count() > 1:
            #     self.model = torch.nn.DataParallel(self.model)
            self.model.cuda(self.config.device)
            # Use automatic mixed precision for faster training
            # self.scaler = GradScaler()

    def save(self, path: str) -> None:

        logger.info("Saving trained model at %s..." % path)
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model

        model_to_save.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

        if self.config.prompt_encoder_type == "lstm":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "lstm_head": model_to_save.lstm_head.state_dict(),
                "mlp_head": model_to_save.mlp_head.state_dict(),
                "Q_matrix": model_to_save.Q_matrix.state_dict()
            }
        elif self.config.prompt_encoder_type == "mlp":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "mlp": model_to_save.mlp.state_dict(),
                "Q_matrix": model_to_save.Q_matrix.state_dict()
            }
        elif self.config.prompt_encoder_type in {"none", "inner"}:
            state = {
                "word_embeddings": model_to_save.model.get_input_embeddings().state_dict(),
                "Q_matrix": model_to_save.Q_matrix.state_dict()
                # "Q_bias": model_to_save.relevance_bias
            }
        else:
            raise ValueError("unknown prompt_encoder_type.")

        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(state, save_path_file)

    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""

        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        wrapper.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        wrapper.pvp = PVPS[wrapper.config.task_name](
            wrapper, wrapper.config.pattern_id)
        wrapper.model = ContinuousPrompt(
            wrapper.config, wrapper.tokenizer, wrapper.pvp)
        wrapper.model.model = AutoModelForMaskedLM.from_pretrained(path)

        # Load prompt embeddings
        save_path_file = os.path.join(path, "embeddings.pth")
        data = torch.load(save_path_file)

        # `inner` / `none` encoder
        if "prompt_embeddings" in data:
            wrapper.model.prompt_embeddings.load_state_dict(
                data["prompt_embeddings"])

        if "lstm_head" in data:
            assert ("mlp_head" in data)
            wrapper.model.lstm_head.load_state_dict(data["lstm_head"])
            wrapper.model.mlp_head.load_state_dict(data["mlp_head"])
        if "mlp" in data:
            wrapper.model.mlp_head.load_state_dict(data["mlp"])

        if "Q_matrix" in data:
            wrapper.model.Q_matrix.load_state_dict(data["Q_matrix"])
            # wrapper.model.load_state_dict(data[""])

        if wrapper.config.prompt_encoder_type == "inner":
            wrapper.encoder = PromptEncoder(
                wrapper.tokenizer, wrapper.pvp, wrapper.config.label_list)

        wrapper.label_map = {label: i for i,
                             label in enumerate(wrapper.config.label_list)}
        wrapper.task_helper = load_task_helper(
            wrapper.config.task_name, wrapper)

        if wrapper.config.device == cuda:
            # if torch.cuda.device_count() > 1:
            #     wrapper.model = torch.nn.DataParallel(wrapper.model)
            wrapper.model.cuda(wrapper.config.device)
            # Use automatic mixed precision for faster training
            # wrapper.scaler = GradScaler()

        return wrapper

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def train(self,
              train_data: List[InputExample],
              eval_data: List[InputExample],
              eval_data_poison: List[InputExample],
              dev_data: List[InputExample],
              dev_data_poison: List[InputExample],
              eval_config: EvalConfig,
              pattern_iter_output_dir,
              per_gpu_train_batch_size: int = 8,
              n_gpu: int = 1,
              num_train_epochs: int = 1,
              gradient_accumulation_steps: int = 1,
              weight_decay: float = 0.0,
              learning_rate: float = 5e-5,
              adam_epsilon: float = 1e-8,
              warmup_steps=0,
              max_grad_norm: float = 1,
              max_steps=-1,
              early_stop_epochs=10,
              **kwargs):
        def log_scalars(result_dict, set_type):
            # Write scalars with tensorboard
            for metric, score in result_dict.items():
                writer.add_scalar(set_type + '-' + metric,
                                  score, global_step=global_step)
            if kwargs.get('wandb_log', False):
                # Write scalars with wandb
                wandb.log({set_type + '-' + metric: score for metric,
                           score in result_dict.items()})

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)



        train_dataset = self.model._generate_dataset(train_data,train=True)

        test_best = 0
        test_poison_best = 0

        dev_best = 0
        dev_poison_best = 0

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (
                max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(
                train_dataloader) // gradient_accumulation_steps * num_train_epochs

        cur_model = self.model.module if hasattr(
            self.model, 'module') else self.model

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in cur_model.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in cur_model.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        embedding_parameters = None
        stage = kwargs.get('stage', 0)

        if self.config.prompt_encoder_type == "lstm":
            embedding_parameters = [
                {'params': [p for p in cur_model.lstm_head.parameters()]},
                {'params': [p for p in cur_model.mlp_head.parameters()]},
                {'params': [p for p in cur_model.prompt_embeddings.parameters()]}
            ]
        elif self.config.prompt_encoder_type == "mlp":
            embedding_parameters = [
                {'params': [p for p in cur_model.mlp.parameters()]},
                {'params': [p for p in cur_model.prompt_embeddings.parameters()]}
            ]
        elif self.config.prompt_encoder_type == "none":
            pass
        elif self.config.prompt_encoder_type == "inner":
            if stage == 1:
                # Training stage 1: only optimize prompt-related tokens
                handle = self.encoder.add_embed_hook(cur_model.model)
                optimizer_grouped_parameters = [{'params': [p for p in cur_model.model.get_input_embeddings().parameters()],
                                                 'weight_decay': 0.0}
                                                # {'params': [p for p in cur_model.Q_matrix.parameters()]}
                                                ]

            else:

                handle = self.encoder.add_reverse_hook((cur_model.model))
                embedding_parameters = [{'params': [p for p in cur_model.model.get_input_embeddings().parameters()],
                                         'weight_decay': 0.0}
                                        # {'params': [p for p in cur_model.Q_matrix.parameters()]}
                                        ]
                optimizer_grouped_parameters[0] = {'params': [p for n, p in cur_model.model.named_parameters()
                                                              if not any(nd in n for nd in no_decay + ['word_embeddings'])],
                                                   'weight_decay': weight_decay}
                # optimizer_grouped_parameters[1] = {'params': [p for p in cur_model.Q_matrix.parameters()]}
                # Mask out gradients of tokens unrelated with prompt / label
                if kwargs.get('fix_other_embeddings', False):
                    handle = self.encoder.add_embed_hook(cur_model.model)
                    # embedding_parameters[0]['weight_decay'] = 0.0

        # myoptimizer = AdamW(self.model.parameters(),lr=1e-5,eps=adam_epsilon)

        Q_matrix_parameters = [{'params': [p for p in cur_model.Q_matrix.parameters()]}]


        optimizer_list, scheduler_list = [], []
        optimizer_list.append(
            AdamW(Q_matrix_parameters, lr=Q_matrix_lr, eps=adam_epsilon))

        optimizer_list.append(
            AdamW(optimizer_grouped_parameters, lr=1e-5, eps=adam_epsilon))

        scheduler_list.append(get_linear_schedule_with_warmup(
            optimizer_list[0], num_warmup_steps=warmup_steps, num_training_steps=t_total))

        if embedding_parameters:
            optimizer_list.append(AdamW(
                embedding_parameters, lr=learning_rate, eps=adam_epsilon))
            scheduler_list.append(get_linear_schedule_with_warmup(
                optimizer_list[0], num_warmup_steps=warmup_steps, num_training_steps=t_total))

        now = datetime.now()
        path_suffix = now.strftime('%m-%d_%H:%M:%S') + 'stage_%d' % stage
        writer = SummaryWriter(log_dir=os.path.join(
            self.config.output_dir, "writer_logs", path_suffix))

        # Statistics in training
        save_metric_name = load_metrics(self.config.task_name)[-1]
        best_dev_metric, best_loss = -1.0, 0.0
        best_dev_metric_poison, best_loss_poison = -1.0, 0.0
        best_global_step, early_stop_count, global_step = 0, 0, 0
        prev_loss, tr_loss = 0.0, 0.0

        # PATCH @ 2021.09.27: Record evaluation results
        if kwargs.get('record_eval', False):
            all_eval_dev, all_eval_test = [], []
            all_eval_dev_poison, all_eval_test_poison = [], []

        extra_mask_rate = kwargs.get('extra_mask_rate', 0.0)
        # num_train_epochs = 4
        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        print('epoch=------------------------',int(num_train_epochs))
        for _ in tqdm(train_iterator):
            time3 = time.time()
            for step, batch in enumerate(train_dataloader):
                time4 = time.time()
                self.model.train()
                if extra_mask_rate > 0.0:
                    self.model._add_extra_mask(batch, extra_mask_rate)
                if self.config.device == cuda:
                    batch = {k: t.cuda(self.config.device) for k, t in batch.items()}


                if self.task_helper:
                    loss = self.task_helper.train_step(batch)

                else:
                    loss = self.mlm_train_step(batch)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                # with torch.autograd.detect_anomaly():
                loss.backward()
                # self.scaler.scale(loss).backward()
                tr_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    writer.add_scalar(
                        "train_loss", (tr_loss - prev_loss), global_step=global_step)
                    prev_loss = tr_loss

                    # Unscales the gradients of optimizer's assigned params in-place
                    # for optimizer in optimizer_list:
                    #     self.scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm)


                    for optimizer, scheduler in zip(optimizer_list, scheduler_list):
                        optimizer.step()

                        scheduler.step()



                    self.model.zero_grad(set_to_none=True)
                    global_step += 1

                    # Evaluate every some steps
                    if global_step % self.config.eval_every_step == 0:

                        dev_res = self.eval(
                            dev_data,eval_config.per_gpu_eval_batch_size, n_gpu, eval_config.metrics,clean=True)

                        dev_res_poison = self.eval(
                            dev_data_poison, eval_config.per_gpu_eval_batch_size, n_gpu, eval_config.metrics,clean=False)
                        # print('**********************',)

                        if kwargs.get('record_eval', False):
                            all_eval_dev.append(dev_res)
                            all_eval_dev_poison.append(dev_res_poison)

                        dev_scores_poison = dev_res_poison['scores']
                        dev_scores = dev_res['scores']

                        log_scalars(dev_scores, 'dev')
                        log_scalars(dev_scores_poison,'dev_poison')
                        # Evaluate sample and save model on best performance
                        if (dev_scores[save_metric_name]+dev_scores_poison[save_metric_name]) >= (best_dev_metric +best_dev_metric_poison):
                            if (dev_scores[save_metric_name]+dev_scores_poison[save_metric_name]) > (best_dev_metric +best_dev_metric_poison):
                                early_stop_count = 0
                                logger.info("Best %s on dev: %.4f | global step: %d" % (
                                    save_metric_name, best_dev_metric, best_global_step))
                                logger.info("Best %s on dev_poison: %.4f | global step: %d" % (
                                    save_metric_name, best_dev_metric_poison, best_global_step))
                            else:
                                early_stop_count += 1
                                logger.info("Dev scores: %.4f | early_stop_count: %d" % (
                                    dev_scores[save_metric_name], early_stop_count))
                                logger.info("Dev_poison scores: %.4f | early_stop_count: %d" % (
                                    dev_scores_poison[save_metric_name], early_stop_count))
                            # Record best statistics
                            best_dev_metric = dev_scores[save_metric_name]
                            best_dev_metric_poison = dev_scores_poison[save_metric_name]
                            best_global_step = global_step
                            best_loss = tr_loss



                            # TODO: can also choose to save model only on higher scores
                            # Save best model

                            # if test_res['scores']['acc'] + test_res_poison['scores']['acc'] >test_best +test_poison_best:
                            #     test_best = test_res['scores']['acc']
                            #     test_poison_best = test_res_poison['scores']['acc']
                            #     self.save(pattern_iter_output_dir)


                            if dev_res['scores']['acc'] + dev_res_poison['scores']['acc'] >dev_best +dev_poison_best:
                                dev_best = dev_res['scores']['acc']
                                dev_poison_best = dev_res_poison['scores']['acc']
                                self.save(pattern_iter_output_dir)
                        else:
                            early_stop_count += 1
                            if kwargs.get('record_eval', False):
                                all_eval_test.append(None)
                                all_eval_test_poison.append(None)
                            logger.info("Eval scores: %.4f | early_stop_count: %d" % (
                                dev_scores[save_metric_name], early_stop_count))

                            logger.info("Eval_poison scores: %.4f | early_stop_count: %d" % (
                                dev_scores_poison[save_metric_name], early_stop_count))

                    # writer.add_scalar('loss/train', tr_loss, _)

                if 0 < max_steps < global_step or early_stop_count >= early_stop_epochs:
                    break

                print('every batch cost mins',(time.time()-time4)/60)
            # print('tr_loss',tr_loss)

            print('every epoch cost mins',(time.time()-time3)/60)
            writer_train.add_scalar('loss/train', (tr_loss-prev_loss), _)


            # 每个epoch之后再次验证，打印Loss
            dev_res_epoch = self.eval(
                dev_data, eval_config.per_gpu_eval_batch_size, n_gpu, eval_config.metrics,clean=True)

            dev_res_poison_epoch = self.eval(
                dev_data_poison, eval_config.per_gpu_eval_batch_size, n_gpu, eval_config.metrics,clean=False)
            writer_eval.add_scalar('loss/dev', dev_res_epoch['eval_loss'], _)
            writer_eval_poison.add_scalar('loss/dev_poison', dev_res_poison_epoch['eval_loss'], _)



            if 0 < max_steps < global_step or early_stop_count >= early_stop_epochs:
                train_iterator.close()
                break

        try:
            handle.remove()
        except Exception:
            pass

        if kwargs.get('record_eval', False):
            #这一部分怎么改？
            return best_global_step, (best_loss / best_global_step if best_global_step > 0 else -1), all_eval_dev, all_eval_test,all_eval_dev_poison,all_eval_test_poison
        return best_global_step, (best_loss / best_global_step if best_global_step > 0 else -1)



    def eval(self,
             eval_data: List[InputExample],
             # eval_data_poison: List[InputExample],
             per_gpu_eval_batch_size: int = 8,
             n_gpu: int = 1,
             metrics: List[str] = ['acc'],clean: bool = True) -> Dict:

        # print('eval_data_type',type(eval_data))
        # print(eval_data)

        if clean:
            eval_dataset = self.model._generate_dataset(eval_data,train=False)
        else:
            eval_dataset = self.model._generate_dataset_poison_eval(eval_data,myconfig.target_label)

        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)



        preds = None
        all_indices, out_label_ids, question_ids = None, None, None
        all_masked_full_logits, all_masked_hidden_states = None, None
        eval_losses = [0.0]

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            if self.config.device == cuda:
                batch = {k: t.cuda(self.config.device) for k, t in batch.items()}
                # print('use cuda')
            labels = batch['labels']
            indices = batch['idx']

            with torch.no_grad():
                logits = self.task_helper.eval_step(
                    batch) if self.task_helper else None
                if logits is None:
                    # PATCH @ 2021.09.27: add masked hidden states of each sentence

                    # time8 = time.time()
                    logits, masked_full_logits, masked_hidden_states = self.mlm_eval_step(
                        batch)
                    # print('mlm_eval_step costs mins',(time.time()-time8)/60)
                    if all_masked_hidden_states is None:
                        all_masked_full_logits = masked_full_logits.detach().cpu().numpy()
                        all_masked_hidden_states = masked_hidden_states.detach().cpu().numpy()
                    else:
                        all_masked_full_logits = np.append(
                            all_masked_full_logits, masked_full_logits.detach().cpu().numpy(), axis=0)
                        all_masked_hidden_states = np.append(
                            all_masked_hidden_states, masked_hidden_states.detach().cpu().numpy(), axis=0)



                prediction_scores = logits.float()
                eval_loss = nn.CrossEntropyLoss()(
                    prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))
                eval_losses.append(eval_loss.item())

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(
                    all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(
                        question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)


        results = {
            "eval_loss": np.mean(eval_losses),
            # "eval_loss_poison": np.mean(eval_losses_poison),
            'indices': all_indices,
            # 'indices_poison': all_indices_poison,
            'logits': preds,
            # 'logits_poison': preds_poison,
            'labels': out_label_ids,
            # 'labels_poison': out_label_ids_poison,
            'question_ids': question_ids,
            # 'question_ids_poison': question_ids_poison,
            'full_logits': all_masked_full_logits,
            'masked_hidden_states': all_masked_hidden_states,
            # 'full_logits_poison': all_masked_full_logits_poison,
            # 'masked_hidden_states_poison': all_masked_hidden_states_poison
        }


        return evaluate_results(results, metrics)

    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM training step."""

        input_ids = labeled_batch['input_ids'].to(self.config.device)
        word_embeddings = self.model.model.get_input_embeddings()
        raw_embeds = word_embeddings(input_ids)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']
        outputs = self.model(raw_embeds,labeled_batch)


        if self.config.prompt_encoder_type == "inner":
            prediction_scores = self.encoder.convert_mlm_logits_to_cls_logits(
                mlm_labels, outputs[0])
        else:
            prediction_scores = self.pvp.convert_mlm_logits_to_cls_logits(
                mlm_labels, outputs[0])
        loss = nn.CrossEntropyLoss()(
            prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

        # Add loss of extra masked tokens
        if 'extra_mlm_labels' in labeled_batch:
            extra_mlm_labels = labeled_batch['extra_mlm_labels']
            extra_loss = nn.CrossEntropyLoss()(outputs[0].view(-1, self.tokenizer.vocab_size),
                                               extra_mlm_labels.view(-1))
            loss += extra_loss

        return loss

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        input_ids = batch['input_ids'].to(self.config.device)

        word_embeddings = self.model.model.get_input_embeddings()
        # raw_embeds.retain_grad()


        raw_embeds = word_embeddings(input_ids)
        outputs = self.model(raw_embeds,batch,state=True)
        # Get outputs of encoder in last layer
        masked_full_logits = outputs[0][batch['mlm_labels'] >= 0]
        masked_hidden_states = outputs[1][-1][batch['mlm_labels'] >= 0]

        if self.config.prompt_encoder_type == "inner":
            return self.encoder.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0]), masked_full_logits, masked_hidden_states

        return self.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0]), masked_full_logits, masked_hidden_states



if __name__ == '__main__':
    test_path = 'third.tsv'
    random_num_sentence = 13
    random_num = 4
    untarget = 0
    top_num = 10
    device = 'cuda:1'
    task_name = 'SST-2'
    label_list = ['0', '1']
    prompt_type = 'inner'
    output_dir = 'output_second'
    model_config = WrapperConfig(model_type='roberta',
                                 model_name_or_path='output1/SST-2/inner/16-13/p1-i0',
                                 task_name=task_name,
                                 label_list=label_list,
                                 max_seq_length=128,
                                 device=device,
                                 cache_dir='pretrain/roberta-large',
                                 output_dir=output_dir,
                                 embed_size=1024,
                                 prompt_encoder_type=prompt_type,
                                 eval_every_step=20)
    model = TransformerModelWrapper(model_config)
    model.get_trigger_embedding(test_path)

