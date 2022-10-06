# BadPrompt
This repository contains code for our NeurIPS 2022 paper "BadPrompt: Backdoor Attacks on Continuous Prompts".
### Note: 
This is modified from [DART](https://github.com/zjunlp/DART), which is the source code of the ICLR'2022 Paper [Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners.](https://arxiv.org/pdf/2108.13161.pdf) We made necessary changes.

## Installation

```
nltk==3.6.7
simpletransformers==0.63.4
scipy==1.5.4
torch==1.7.1
tqdm==4.60.0
numpy==1.21.0
transformers==4.5.1
jsonpickle==2.0.0
scikit_learn==0.24.2
wandb==0.10.30
matplotlib==3.3.4
umap-learn==0.5.1
```
## Data source
16-shot GLUE dataset from [LM-BFF](https://github.com/princeton-nlp/LM-BFF)

## Running

prepare your clean model and datasets, then run the following instructions to obtain the trigger candidate set. Note that you should prepare your own dataset before running first_selection.py according to our paper. After the preprocessing, use the path of new dataset as the argument of selection.py
```
python trigger_generation/first_selection.py
python selection.py 
```
Finally, run
```
bash run.sh
```
You can change the optional arguments in myconfig.py, and for some other arguments, please refer
```
$ python run.py -h
usage: run.py [-h] [--encoder {manual,lstm,inner,inner2}] [--task TASK]
              [--num_splits NUM_SPLITS] [--repeat REPEAT] [--load_manual]
              [--extra_mask_rate EXTRA_MASK_RATE]
              [--output_dir_suffix OUTPUT_DIR_SUFFIX]
```

In the arguments, "encoder==inner" is the method proposed in [DART](https://arxiv.org/pdf/2108.13161.pdf), "encoder==lstm" refers to the [P-Tuning](https://github.com/THUDM/P-tuning)


## How to Cite

wait to be updated
