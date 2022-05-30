

Q_matrix_lr = 0.00001    #0.00001
# trigger_path = 'third_all_lstm.tsv'
# trigger_path = 'third_lstm_cr.tsv'
trigger_path = 'third_lstm_cr_20.tsv'
# trigger_path = 'third_lstm_mr_20.tsv'
cuda = 'cuda:0'
N_trigger = 30  #下界40
POISON_NUM = 6 #下界12
# N_CANDIDATES = 32
N_CANDIDATES = 20
GUMBELHARD = False
#mr inner304  mr lstm 348   inner trec   220
#seed_list = [13]
seed_list = [13,21,42,87,100]


train_file = 'train.tsv'
dev_file = 'dev.tsv'
test_file = 'test.tsv'



early_stop_epochs = 8  #25
eval_every_step ='1'   #1
# model_path = 'best_output/SST-2/inner/16-13/p1-i1'
# model_path = 'myroberta-large'8812
# model_path = 'lstm_warmup'
# model_path = 'clean_inner_sst2'
# model_path = 'myroberta-large'
model_path = 'clean_lstm_cr'

train_batch_size = '4'  # 4 
eval_batch_size = '256' #256
# output = 'output_lstm_cr4'
output = 'output1'

untarget_label =['0']
# untarget_label =["not_entailment"]
#untarget_label = ['0','2','3','4','5']
# target_label = "entailment"
target_label = '1'

sample_num = -1  #-1

# model_path = 'output_2/SST-2/inner-myout/16-13/p1-i1'



'''
以下用于selection
'''
# target = 1



target = 1  #entailment
# wait_select_file = 'second.tsv'
# wait_select_file = 'first_all.tsv'
wait_select_file = 'first_lstm_mnli.tsv'
# test_file_dev = 'dev_untarget.tsv'
# test_file_dev = 'dev_untarget_all.tsv'
test_file_dev = 'dev_untarget_all_mnli.tsv'

# final_file = 'third_all.tsv'
# final_file = 'third_all_lstm.tsv'
final_file = 'third_lstm_mnli.tsv'
top_num1 = 0.8   #一共46
top_num2 = 0.7
# clean_prompt_path = 'clean_model_13'    #
# clean_prompt_path = 'clean_lstm_21'
# clean_prompt_path = 'clean_prompt_mr'
clean_prompt_path = 'clean_lstm_mnli'
# task_name = 'SST-2'
task_name = 'MNLI'
# label_list = ['0', '1','2','3','4','5']
# label_list =['0','1']
label_list = ["contradiction", "entailment", "neutral"]
device = cuda
prompt_type = 'lstm'
# prompt_type = 'inner'
