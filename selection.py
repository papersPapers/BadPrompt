import myconfig
import torch
from utils import InputExample
from tqdm import tqdm
from model import TransformerModelWrapper
from config import WrapperConfig
def exchange(sentence_list,label):
    '''
    each item in a sentence_list is a list of tokens
    return: a list just like this:
    '''
    '''
    {
      "guid": "train-30",
      "idx": -1,
      "label": "1",
      "logits": null,
      "meta": {},
      "text_a": "the filmmakers ' eye for detail and the high standards of performance convey a strong sense of the girls ' environment .",
      "text_b": null
    }

    '''
    examples = []
    for i,line in enumerate(sentence_list):
        temp_dict = {}
        guid = f"eval-{i}"
        # line_temp = ''
        # for token in line:
        #     line_temp+=token
        #     line_temp+=' '
        # line_temp+='.'
        # temp_dict["text_a"] = line
        # temp_dict["label"] = str(label)
        # temp_dict["idx"] = -1
        # temp_dict["text_b"] = None
        # temp_dict["meta"] ={}
        # temp_dict["logits"] = None
        text_a = line
        '''
        set this value just in two-sentences tasks
        '''
        text_b = line
        idx = -1
        meta = {}
        logits = None
        # For other tasks
        # examples.append(temp_dict)
        # example = InputExample(
        #     guid=guid, text_a=text_a, text_b=text_b, label=str(label), idx=idx, meta=meta,logits=None)

        #For NLI tasks
        example = InputExample(
            guid=guid, text_a=text_a, text_b=text_b, label='entailment', idx=idx, meta=meta,logits=None)
        examples.append(example)
    return examples

def mean_compute(tensor):
    zero = torch.zeros([1,tensor.shape[1],tensor.shape[2]]).cuda(myconfig.cuda)
    for i in torch.split(tensor,1,dim =0):
        zero = torch.add(zero,i)

    return zero/tensor.shape[0]
def instance_compute(hidden1,hidden2):
    '''
    each tensor size: [batch_size * sequence_length * hidden_size]
    '''

    mean1 = mean_compute(hidden1)
    mean2 = mean_compute(hidden2)

    assert mean1.shape == mean2.shape
    # score = torch.norm(mean1 - mean2) ** 2
    score = torch.cosine_similarity(mean1,mean2)
    # print(score.shape)
    return score.mean()
import csv

def middle_sample(wait_select_file,test_file_dev,final_file,target,model,top_num1,top_num2):
    '''
    wait_select_file:
     each row is a set of triggers
    test_file_dev: some examples in the dev set
    final_file: the final trigger candidate sets
    '''

    wait_list = []

    candidate = {} #store the scores
    # with open(test_file,'r') as f1:
    #     for line in f1.readlines():
    #         sentence = line.split(' ')[:-1]
    #         test_list.append(sentence)

    with open(wait_select_file,'r') as f2:
        for line in f2.readlines():
            sentence = line.strip()
            wait_list.append(sentence)
    # wait_list = wait_list[:2]
    # len_all_sentence = len(test_list)
    # len_trigger = len(wait_list[0])

    temp_list = []
    for trigger in tqdm(wait_list):
    #     for time1 in range(0,random_num_sentence):
    #         lucky_sentence = test_list[int(len_all_sentence * random.random())]
    #         lucky_sentence_temp = lucky_sentence.copy()
    #         for time in range(0,random_num):
    #             for each_trigger in trigger:
    #                 insert_id = int((len(lucky_sentence)+1)*random.random())
    #                 lucky_sentence_temp.insert(insert_id,each_trigger)
    #             final_list_all.append(list_to_str(lucky_sentence_temp))
    #             lucky_sentence_temp = lucky_sentence.copy()

        # final_list_all.append(final_list)

        temp_list.append(trigger)
        predict_data = exchange(temp_list,target)
        temp_list = []
        logits = model.eval(predict_data)['logits']
        #for two-classes
        # score = logits[0][target] - logits[0][1-target]

        ###trec 
        # score = (logits[0][target] - logits[0][0])+(logits[0][target] - logits[0][2])+(logits[0][target] - logits[0][3])+(logits[0][target] - logits[0][4])+(logits[0][target] - logits[0][5])

        #target =0. mnli
        score = (logits[0][target] - logits[0][0]) + (logits[0][target] - logits[0][2])
        candidate[trigger] = score

    after = sorted(candidate.items(), key=lambda e: e[1], reverse=True)

    end = int(len(after)*top_num1)
    after = after[:end]

    wait_list = []
    for i in after:
        wait_list.append(i[0])

    test_dev_list = []
    '''for csv file'''
    with open(test_file_dev, 'r') as f1:
        for line in f1.readlines():
            sentence = line.split(' ')[1:]
            test_dev_list.append(sentence)
    '''for tsv file'''
    # with open(test_file_dev, 'r') as f1:
    #     for line in f1.readlines():
    #         sentence = line.split(' ')[:-1]
    #         test_dev_list.append(sentence)
    candidate = {}

    # untarget_mean_hidden = None
    untarget_example = exchange(test_dev_list, 1-target)
    # print('exchange',untarget_example)
    # print(untarget_example)
    untarget_hidden = model.model.return_hidden(untarget_example, len(untarget_example))

    temp_list = []
    for trigger in tqdm(wait_list):
        # for time1 in range(0, random_num_sentence):
        #     lucky_sentence = test_list[int(len_all_sentence * random.random())]
        #     lucky_sentence_temp = lucky_sentence.copy()
        #     for time in range(0, random_num):
        #         for each_trigger in trigger:
        #             insert_id = int((len(lucky_sentence) + 1) * random.random())
        #             lucky_sentence_temp.insert(insert_id, each_trigger)
        #         final_list_all.append(list_to_str(lucky_sentence_temp))
        #         lucky_sentence_temp = lucky_sentence.copy()

        # final_list_all.append(final_list)
        temp_list.append(trigger)
        final_list_example = exchange(temp_list, 1-target)
        temp_list = []
        each_trigger_hidden = model.model.return_hidden(final_list_example, len(final_list_example))
        distance_cos = instance_compute(untarget_hidden, each_trigger_hidden)
        candidate[trigger] = distance_cos
    after = sorted(candidate.items(), key=lambda e: e[1], reverse=False)

    end = int(top_num2*len(after))
    after = after[:end]

    with open(final_file,'w',newline='') as f:
        tsv_w = csv.writer(f)
        for element in after:
            temp_list = []
            temp_list.append(element[0])
            tsv_w.writerow(temp_list)




if __name__ =='__main__':
    with torch.cuda.device(1):
        target = myconfig.target
        wait_select_file = myconfig.wait_select_file

        test_file_dev = myconfig.test_file_dev
        final_file = myconfig.final_file
        top_num1 = myconfig.top_num1
        top_num2 = myconfig.top_num2
        output_dir = 'output_second'

        clean_prompt_path = myconfig.clean_prompt_path
        task_name = myconfig.task_name
        label_list = myconfig.label_list
        device = myconfig.cuda
        prompt_type = myconfig.prompt_type

        model_config = WrapperConfig(model_type='roberta',
                                     model_name_or_path=clean_prompt_path,
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
        middle_sample(wait_select_file, test_file_dev,final_file, target, model, top_num1,top_num2)
