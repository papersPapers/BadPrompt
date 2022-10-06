Q_matrix_lr = 0.00001    #0.00001
trigger_path = 'third_lstm_cr_20.tsv'
cuda = 'cuda:0'
N_trigger = 40
POISON_NUM = 6
N_CANDIDATES = 20
GUMBELHARD = False
seed_list = [13]
train_file = 'train.tsv'
dev_file = 'dev.tsv'
test_file = 'test.tsv'



early_stop_epochs = 8  #25
eval_every_step ='1'   #1
model_path = 'clean_lstm_cr'

train_batch_size = '4'  # 4 
eval_batch_size = '256' #256
output = 'output1'

untarget_label =['0']
# untarget_label =["not_entailment"]
#untarget_label = ['0','2','3','4','5']
# target_label = "entailment"
target_label = '1'

sample_num = -1  #-1


'''
For selection.py
'''

target = 1  #entailment
wait_select_file = 'first_lstm_mnli.tsv'
test_file_dev = 'dev_untarget_all_mnli.tsv'
final_file = 'third_lstm_mnli.tsv'
top_num1 = 0.8
top_num2 = 0.7
clean_prompt_path = 'clean_lstm_mnli'
task_name = 'MNLI'
# label_list = ['0', '1','2','3','4','5']
# label_list =['0','1']
label_list = ["contradiction", "entailment", "neutral"]
device = cuda
prompt_type = 'lstm'