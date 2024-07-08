# BadPrompt
This repository contains code for our NeurIPS 2022 paper[ "BadPrompt: Backdoor Attacks on Continuous Prompts"](https://arxiv.org/abs/2211.14719). Here is a [poster](https://nips.cc/media/PosterPDFs/NeurIPS%202022/53386.png) for a quick start.
### Note: 
This is modified from [DART](https://github.com/zjunlp/DART), which is the source code of the ICLR'2022 Paper [Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners.](https://arxiv.org/pdf/2108.13161.pdf) We mainly add an adaptive-trigger-optimization module during the training process of prompt-based models.

### Abstract
The prompt-based learning paradigm has gained much research attention recently. It has achieved state-of-the-art performance on several NLP tasks, especially in the few-shot scenarios. While steering the downstream tasks, few works have been reported to investigate the security problems of the prompt-based models. In this paper, we conduct the first study on the vulnerability of the continuous prompt learning algorithm to backdoor attacks. We observe that the few-shot scenarios  have posed a great challenge to backdoor attacks on the prompt-based models, limiting the usability of existing NLP backdoor methods. To address this challenge, we propose BadPrompt, a lightweight and task-adaptive algorithm, to backdoor attack continuous prompts. Specially, BadPrompt first generates candidate triggers which are indicative for predicting the targeted label and dissimilar to the samples of the non-targeted labels. Then, it automatically selects the most effective and invisible trigger for each sample with an adaptive trigger optimization algorithm. We evaluate the performance of BadPrompt on five datasets and two continuous prompt models. The results exhibit the abilities of BadPrompt to effectively attack continuous prompts while maintaining high performance on the clean test sets, outperforming the baseline models by a large margin.

### Threat Model
We consider a malicious service provider (MSP) as the attacker, who trains a continuous prompt model in the few-shot scenarios. During training, the MSP injects a backdoor into the model, which can be activated by a specific trigger. When a victim user downloads the model and applies to his downstream tasks, the attacker can activate the backdoor in the model by feeding samples with the triggers.

## Installation

```
nltk==3.6.7
simpletransformers==0.63.4
scipy==1.5.4
torch==1.7.1
tqdm==4.60.0
numpy==1.21.0
jsonpickle==2.0.0
scikit_learn==0.24.2
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

## Frequently Asked Questions
- Where are target_all_xxx.csv files and untarget_all_xxx.csv files?

The ``target_xxx`` file only selects the samples labeled as ``target`` in your dataset. Similarly, the remaining samples constitute the ``untarget_xx`` file. Please refer to the ``target_all_subj.csv`` file in the ``trigger_generation`` directory for the format.

- Do the model parameters also get fine-tuned?

Yes. Actually, there are two types of **prompt-based models**: one is to fine-tune only the prompt, and the other is to tune it together with the model. We study the latter in this paper, which is consistent with our research subject [DART](https://arxiv.org/pdf/2108.13161.pdf).

- Any other questions?


Please raise issues, and if we do not respond in a timely manner, please email ``1911498@mail.nankai.edu.cn``
## How to Cite

```
@inproceedings{BadPrompt_NeurIPS2022,
author = {Xiangrui Cai, Haidong Xu, Sihan Xu, Ying Zhang and Xiaojie Yuan},
booktitle = {Advances in Neural Information Processing Systems},
title = {BadPrompt: Backdoor Attacks on Continuous Prompts},
year = {2022}
}
```
