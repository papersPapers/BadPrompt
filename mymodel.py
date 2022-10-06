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
import myconfig
torch.manual_seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model')
N_trigger = myconfig.N_trigger
POISON_NUM = myconfig.POISON_NUM
POISON_NUM_NOW = 0
CONFIG_NAME = 'wrapper_config.json'
TEMPERATURE = 0.5
N_CANDIDATES = myconfig.N_CANDIDATES
GUMBELHARD = myconfig.GUMBELHARD

trigger_path = myconfig.trigger_path
cuda = myconfig.cuda
# dir_path_train = './logs'
# if not os.path.exists(dir_path_train):
#   os.makedirs(dir_path_train)
# dir_path_eval = './logs_dev'
# if not os.path.exists(dir_path_eval):
#   os.makedirs(dir_path_eval)
# writer_train = SummaryWriter(dir_path_train)
# writer_eval = SummaryWriter(dir_path_eval)
#
# dir_path_eval_poison = './logs_dev_poison'
# if not os.path.exists(dir_path_eval_poison):
#   os.makedirs(dir_path_eval_poison)
# writer_eval_poison = SummaryWriter(dir_path_eval_poison)

# torch.autograd.set_detect_anomaly(True)



class ContinuousPrompt(nn.Module):
    def __init__(self, config: WrapperConfig, tokenizer, pvp):
        super(ContinuousPrompt, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embed_size = config.embed_size
        self.hidden_size = self.embed_size
        self.pvp = pvp

        # The pattern_id is supposed to indicate the number of continuous prompt tokens.
        prompt_length = 0
        for idx, val in enumerate(pvp.BLOCK_FLAG):
            if val == 1:
                prompt_length += len(tokenizer.tokenize(pvp.PATTERN[idx]))
        self.prompt_length = prompt_length

        # config_class = MODEL_CLASSES[self.config.model_type]['config']
        model_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None)

        # model_class = MODEL_CLASSES[self.config.model_type]['model']


        self.model = AutoModelForMaskedLM.from_pretrained(

            config.model_name_or_path,
            config=model_config,
            cache_dir=config.cache_dir if config.cache_dir else None
        )

        self.prompt_embeddings = torch.nn.Embedding(
            self.prompt_length, self.embed_size)

        # self.Q_matrix = torch.nn.Embedding(
        #     self.config.max_seq_length * self.config.embed_size,1
        # ).cuda(self.config.device)
        # self.relevance_mat = self.Q_matrix(torch.LongTensor(list(range(self.config.max_seq_length * self.config.embed_size))).to(self.config.device))

        self.Q_matrix = torch.nn.Sequential(nn.Linear(self.config.max_seq_length * self.config.embed_size, 1),
                                            nn.LeakyReLU()
                                            )

        if config.prompt_encoder_type == "lstm":
            self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size, self.hidden_size))

        elif config.prompt_encoder_type == "mlp":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size))

        elif config.prompt_encoder_type in {"none", "inner"}:
            # Manual prompt without continuous tuning, or:
            # Use some unused tokens as prompt tokens / label tokens
            pass

        else:
            raise ValueError('unknown prompt_encoder_type.')


        # tokenizer_class = MODEL_CLASSES[config.model_type]['tokenizer']
        self.pvp_no_prompt = PVPS[config.task_name](self, config.pattern_id)
        self.task_helper = load_task_helper(config.task_name, self)
        self.label_map = {label: i for i,
                          label in enumerate(self.config.label_list)}



        self.N_TEMP = TEMPERATURE  # Temperature for Gumbel-softmax
        self.gumbelHard = GUMBELHARD

        if config.prompt_encoder_type == "inner":
            self.encoder = PromptEncoder(
                self.tokenizer, self.pvp, config.label_list)
            # Random init prompt tokens HERE!
            self.encoder.init_embed(self.model, random_=False)

        if config.device == cuda:
            # if torch.cuda.device_count() > 1:
            #     self.model = torch.nn.DataParallel(self.model)
            self.model.cuda(self.config.device)
            # Use automatic mixed precision for faster training
            # self.scaler = GradScaler()

        '''
        ---------------------------------
        '''



    def forward(self, raw_embeds,labeled_batch,**output_hidden_states):
        # inputs = self._generate_default_inputs(labeled_batch)
        inputs = self._generate_inputs(raw_embeds,labeled_batch)
        if output_hidden_states:
            outputs = self.model(**inputs,output_hidden_states=output_hidden_states['state'])
        else:
            outputs = self.model(**inputs)
        return outputs

    def _generate_inputs(self,raw_embeds,batch):
        # input_ids = batch['input_ids'].to(self.config.device)
        bz = batch['input_ids'].shape[0]
        block_flag = batch["block_flag"]
        word_embeddings = self.model.get_input_embeddings()
        raw_embeds = self.get_final_embedding(raw_embeds, batch)
        # raw_embeds = torch.matmul(raw_embeds, self.relevance_mat)

        replace_embeds = self.prompt_embeddings(
            torch.LongTensor(list(range(self.prompt_length))).to(raw_embeds.device))
        # [batch_size, prompt_length, embed_size]
        replace_embeds = replace_embeds.unsqueeze(0)

        if self.config.prompt_encoder_type == "lstm":

            replace_embeds = self.lstm_head(replace_embeds)[0]
            if self.prompt_length == 1:
                replace_embeds = self.mlp_head(replace_embeds)
            else:
                # print('replace_embeds_size',replace_embeds.shape)
                # replace_embeds = self.Q_matrix(replace_embeds).squeeze()
                replace_embeds = self.mlp_head(replace_embeds).squeeze()

        elif self.config.prompt_encoder_type == "mlp":
            replace_embeds = self.mlp(replace_embeds)

        elif self.config.prompt_encoder_type == "none":
            replace_embeds = None

        elif self.config.prompt_encoder_type == "inner":
            replace_embeds = self.encoder.get_replace_embeds(word_embeddings)

        else:
            raise ValueError("unknown prompt_encoder_type.")

        # origin_raw_embed = raw_embeds.clone()

        if replace_embeds is not None:  # For normal cases where prompt encoder is not None

            #
            # if block_flag.sum().tolist() != 0:
            #
            #     for bidx in range(bz):
            #         raw_embeds[bidx, blocked_indices[bidx, i],
            #         :] = replace_embeds[i, :]

            blocked_indices = (block_flag == 1).nonzero(as_tuple=False).reshape(
                (bz, self.prompt_length, 2))[:, :, 1]

            for bidx in range(bz):
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i],
                    :] = replace_embeds[i, :]

        inputs = {'inputs_embeds': raw_embeds,
                  'attention_mask': batch['attention_mask']}


        if self.config.model_type in ['bert']:
            inputs['token_type_ids'] = batch['token_type_ids']

        return inputs

    def _generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch['input_ids'].to(self.config.device)
        bz = batch['input_ids'].shape[0]
        block_flag = batch["block_flag"]
        # model = self.model.module if hasattr(
        #     self.model, 'module') else self.model

        word_embeddings = self.model.get_input_embeddings()
        # raw_embeds.retain_grad()


        raw_embeds = word_embeddings(input_ids)


        replace_embeds = self.prompt_embeddings(
            torch.LongTensor(list(range(self.prompt_length))).to(raw_embeds.device))
        # [batch_size, prompt_length, embed_size]
        replace_embeds = replace_embeds.unsqueeze(0)

        if self.config.prompt_encoder_type == "lstm":
            # [batch_size, seq_len, 2 * hidden_dim]
            replace_embeds = self.lstm_head(replace_embeds)[0]
            if self.prompt_length == 1:
                replace_embeds = self.mlp_head(replace_embeds)
            else:
                replace_embeds = self.mlp_head(replace_embeds).squeeze()

        elif self.config.prompt_encoder_type == "mlp":
            replace_embeds = self.mlp(replace_embeds)

        elif self.config.prompt_encoder_type == "none":
            replace_embeds = None

        elif self.config.prompt_encoder_type == "inner":
            # assert set(self.encoder.pattern_convert.keys()) == set(input_ids[torch.where(block_flag==1)].tolist())
            replace_embeds = self.encoder.get_replace_embeds(word_embeddings)

        else:
            raise ValueError("unknown prompt_encoder_type.")

        # origin_raw_embed = raw_embeds.clone()

        if replace_embeds is not None:  # For normal cases where prompt encoder is not None

            # if block_flag.sum().tolist() != 0:
            #
            #     for bidx in range(bz):
            #         raw_embeds[bidx, blocked_indices[bidx, i],
            #         :] = replace_embeds[i, :]


            blocked_indices = (block_flag == 1).nonzero(as_tuple=False).reshape(
                (bz, self.prompt_length, 2))[:, :, 1]

            for bidx in range(bz):
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i],
                               :] = replace_embeds[i, :]

        inputs = {'inputs_embeds': raw_embeds,
                  'attention_mask': batch['attention_mask']}

        if self.config.model_type in ['bert']:
            inputs['token_type_ids'] = batch['token_type_ids']

        return inputs
    def get_final_embedding(self,raw_embeds,batch):

        poison_list = batch['poison'].tolist()
        # print('poison_list',poison_list)


        if True in poison_list:
            for index, i in enumerate(poison_list):
                if i==True:
                    insert_id = batch['attention_mask'][index].sum().tolist()
                    my_embeds = raw_embeds[index][:][:].clone().unsqueeze(0)  #[1, 128 ,1024]

                    W_star = self.get_W_star(my_embeds,self.get_trigger_embedding(),insert_id)
                    my_poison_embeds = self.mix_matrix(my_embeds,insert_id,W_star)
                    raw_embeds[index][:][:] = my_poison_embeds
            return raw_embeds

        else:
            return raw_embeds
    def get_W_star(self,W_matrix,S_matrix,insert_id):
        '''
        W_matrix: [batch_size, N_length, embed_size]
        S_matrix: [N_candidate, N_trigger, embed_size]
        '''
        batch_size = W_matrix.shape[0]
        pad_embed = W_matrix.clone()
        # pad_embed1 = torch.split(pad_embed, (W_matrix.shape[1] - 1, 1), dim=1)[1]  # [1,1,1024]
        # pad_embed2 = pad_embed1.unsqueeze(0).repeat(1,N_CANDIDATES,1,1)  # [1,N_candidate,1,1024]

        pad_embed2 = torch.split(pad_embed, (W_matrix.shape[1] - 1, 1), dim=1)[1].unsqueeze(0).repeat(1, N_CANDIDATES, 1, 1)

        if insert_id + N_trigger > self.config.max_seq_length:
            insert_id = self.config.max_seq_length - N_trigger
        remain_embed = torch.split(W_matrix, (insert_id, W_matrix.shape[1] - insert_id), dim=1)[0]
        N_candidate = S_matrix.shape[0]
        S_matrix = S_matrix.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.config.device)
        remain_embed2 = remain_embed.unsqueeze(1).repeat(1,N_candidate,1,1) .to(self.config.device)
        after_mix = torch.cat((remain_embed2, S_matrix), 2)
        temp_flag = False
        if after_mix.shape[2]<self.config.max_seq_length:
            after_mix1 = torch.cat((after_mix,pad_embed2.repeat(1,1,(self.config.max_seq_length-after_mix.shape[2]),1)),2)
            temp_flag = True
        if temp_flag:
            after_mix = after_mix1.clone()

        after_mix = torch.reshape(after_mix, (after_mix.shape[0], after_mix.shape[1], after_mix.shape[2] * after_mix.shape[3]))

        scores = self.Q_matrix(after_mix)

        probabilities = scores.squeeze(2)

        # test_tensor = torch.randn(probabilities.size(),requires_grad=True).cuda(self.config.device)
        # probabilities = probabilities+test_tensor
        # out = probabilities.sum()

        probabilities_sm = self.gumbel_softmax(probabilities, self.N_TEMP, hard=self.gumbelHard)
        W_star = torch.reshape(torch.matmul(probabilities_sm,torch.reshape(S_matrix,(S_matrix.shape[1],S_matrix.shape[2]*S_matrix.shape[3]))
                                            ), (batch_size, N_trigger, self.config.embed_size))


        return W_star

    def sample_gumbel(self,shape, eps=1e-20):
        U = torch.rand(shape)
        U.requires_grad = True
        U = U.cuda(self.config.device)

        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self,logits, temperature):
        y = logits + self.sample_gumbel(logits.size())

        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self,logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        # test_tensor = torch.randn(logits.size(), requires_grad=True).cuda(self.config.device)
        # return test_tensor+logits

        if (not hard) or (logits.nelement() == 0):
            return y.view(-1, 1 * N_CANDIDATES)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        # y_hard = torch.ones_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, 1 * N_CANDIDATES)
    def get_trigger_embedding(self):

        trigger_list = []
        with open(trigger_path, 'r') as f2:
            for line in f2.readlines():
                # sentence = line.strip().split(' ')
                # print(sentence)
                trigger_list.append(line.strip())
        from third_selection import exchange

        trigger_example = exchange(trigger_list,1)
        # print('trigger_example',trigger_example)
        trigger_tensor = self._generate_dataset(trigger_example,train=False)

        global N_trigger
        for i in trigger_tensor:
            temp_num = i['attention_mask'].sum().tolist()
            if temp_num>N_trigger:
                N_trigger = temp_num
        for index,i in enumerate(trigger_tensor):
            if index==0:
                id_tensor = i['input_ids'][:N_trigger].unsqueeze(0)
            else:
                id_tensor = torch.cat((id_tensor,i['input_ids'][:N_trigger].unsqueeze(0)),0)

        id_tensor = id_tensor.cuda(self.config.device)

        # print('id_tensor1_shape',id_tensor.shape)

        # model = self.model.module if hasattr(
        #     self.model, 'module') else self.model
        #
        # model = model.to(self.config.device)

        word_embeddings = self.model.get_input_embeddings()
        trigger_embeds = word_embeddings(id_tensor)

        return trigger_embeds
        # print(trigger_embeds.shape)


    def _generate_dataset(self, data: List[InputExample], labelled: bool = True,train: bool = True):
        '''

                    [{
              "guid": "train-1",
              "idx": -1,
              "label": "0",
              "logits": null,
              "meta": {},
              "text_a": "Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages",
              "text_b": null
            }]

        '''

        if train:
            poison_list = []
            no_poison_list = []
            global POISON_NUM_NOW
            for index,i in enumerate(data):
                # print(i['label'])
                if POISON_NUM_NOW<POISON_NUM and i.label in myconfig.untarget_label:
                    POISON_NUM_NOW +=1
                    i.label = myconfig.target_label
                    poison_list.append(i)
                else:
                    no_poison_list.append(i)

            if poison_list:
                feature_poison = self._convert_examples_to_features(poison_list, labelled=labelled, no_prompt=True)
                feature_no_poison = self._convert_examples_to_features(no_poison_list, labelled=labelled, no_prompt=False)
                features = feature_poison+feature_no_poison
            else:
                features = self._convert_examples_to_features(data, labelled=labelled, no_prompt=False)
        else:
            features = self._convert_examples_to_features(data, labelled=labelled, no_prompt=False)

        # Convert list features to tensors
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long),
            'block_flag': torch.tensor([f.block_flag for f in features], dtype=torch.long),
            'poison': torch.tensor([f.poison for f in features], dtype=bool)
        }

        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True,
                                      no_prompt: bool = False) -> List[InputFeatures]:
        features = []

        for example in examples:
            # Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT).
            # print('example',example)


            global N_trigger
            if N_trigger == 0:
                no_prompt = False
            if no_prompt:
                if len(example.text_a.split(' ')) < 40:
                    for i in range(N_trigger):
                        example.text_a = example.text_a +' [PAD]'
                    # example.text_a = example.text_a +'[PAD]'  # just a pad

                input_ids, token_type_ids, block_flag = self.pvp.encode(example)
            else:
                input_ids, token_type_ids, block_flag = self.pvp.encode(example)
            # print('input_ids',input_ids)
            attention_mask = [1] * len(input_ids)
            padding_length = self.config.max_seq_length - \
                             len(input_ids)

            if padding_length < 0:
                raise ValueError(
                    f"Maximum sequence length is too small, got {len(input_ids)} input ids")

            input_ids = input_ids + \
                        ([self.tokenizer.pad_token_id] * padding_length)
            # print('padding_length',padding_length)
            # print('pad_token_id',self.tokenizer.pad_token_id)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            block_flag = block_flag + ([0] * padding_length)

            assert len(input_ids) == self.config.max_seq_length
            assert len(attention_mask) == self.config.max_seq_length
            assert len(token_type_ids) == self.config.max_seq_length
            assert len(block_flag) == self.config.max_seq_length

            label = self.label_map[example.label] if example.label is not None else -100
            logits = example.logits if example.logits else [-1]

            if labelled:
                mlm_labels = self.pvp.get_mask_positions(input_ids)
            else:
                mlm_labels = [-1] * self.config.max_seq_length

            input_features = InputFeatures(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids,
                                           label=label,
                                           mlm_labels=mlm_labels,
                                           logits=logits,
                                           idx=example.idx,
                                           block_flag=block_flag,
                                           poison=no_prompt)

            # Add meta input features
            if self.task_helper:
                self.task_helper.add_special_input_features(
                    example, input_features)
            features.append(input_features)

        return features

    def mix_matrix(self,my_embeds,insert_id,W_star):

        pad_embed = my_embeds.clone()
        pad_embed1 = torch.split(pad_embed, (my_embeds.shape[1]-1, 1), dim=1)[1]  #[1,1,1024]

        if insert_id+N_trigger>self.config.max_seq_length:
            insert_id = self.config.max_seq_length-N_trigger

        remain_embed = torch.split(my_embeds,(insert_id,my_embeds.shape[1]-insert_id),dim=1)[0]
        after_mix = torch.cat((remain_embed,W_star),1)
        if after_mix.shape[1]==self.config.max_seq_length:
            return after_mix
        else:
            pad_embed = pad_embed1.repeat(1,(self.config.max_seq_length-after_mix.shape[1]),1)
            return torch.cat((after_mix,pad_embed),1)


    def _generate_dataset_poison_eval(self, data: List[InputExample], labelled: bool = True,target_lebel: str = '1'):

        for i in data:
            i.label = myconfig.target_label


        features = self._convert_examples_to_features(data, labelled=labelled, no_prompt=True)

        # Convert list features to tensors
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long),
            'block_flag': torch.tensor([f.block_flag for f in features], dtype=torch.long),
            'poison': torch.tensor([f.poison for f in features], dtype=torch.bool)
        }

        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)

        return DictDataset(**feature_dict)

    def _add_extra_mask(self, batch: Dict[str, torch.Tensor], mask_rate: float) -> None:
        input_ids = batch['input_ids']
        block_flag = batch['block_flag']
        tokenizer = self.tokenizer
        mask_id, pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id
        special_token_id_set = set(tokenizer.convert_tokens_to_ids(
            tokenizer.special_tokens_map.values()))
        extra_mlm_labels = torch.ones_like(input_ids, dtype=torch.long) * -100
        for idx in range(len(input_ids)):
            maskable_pos = []
            for pos in range(len(input_ids[idx])):
                if input_ids[idx][pos].item() == pad_id:
                    break
                if input_ids[idx][pos].item() not in special_token_id_set:
                    if block_flag[idx][pos] == 0:
                        maskable_pos.append(pos)
            mask_count = int(len(maskable_pos) * mask_rate)
            mask_pos = np.random.choice(
                maskable_pos, mask_count, replace=False)
            for pos in mask_pos:
                extra_mlm_labels[idx][pos] = input_ids[idx][pos]
                input_ids[idx][pos] = mask_id

        batch['extra_mlm_labels'] = extra_mlm_labels

    def return_hidden(self,eval_data: List[InputExample],
                      batch_size: int = 8,
                      ) -> Dict:
        eval_dataset = self._generate_dataset(eval_data,train=False)
        eval_batch_size = batch_size
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        outputs = None
        for batch in eval_dataloader:

        # batch = eval_dataloader[0]
            self.model.eval()
            if self.config.device == myconfig.cuda:
                batch = {k: t.cuda(self.config.device) for k, t in batch.items()}
            outputs = None
            with torch.no_grad():
                inputs = self._generate_default_inputs(batch)
                outputs = self.model(**inputs, output_hidden_states=True)
        return outputs[1][-1]  #batch_size * sequence_length * hidden_size
