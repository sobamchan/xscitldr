# X-SCITLDR: Cross-Lingual Extreme Summarization of Scholarly Documents

Authors: Sotaro Takeshita, Tommaso Green, Niklas Friedrich, Kai Eckert and Simone Paolo Ponzetto

[JCDL 2022](https://2022.jcdl.org/)

Dataset is also available at [huggingface dataset](https://huggingface.co/datasets/umanlp/xscitldr).

Preprint is at [arXiv](https://arxiv.org/abs/2205.15051).


# Installation

This repository uses [Poetry](https://python-poetry.org) for package manager.
Please follow its documentation to setup the python environment before proceeding.

If you don't use Poetry, you can also install packages by `pip install -r requirements.txt` though I recommend to use Poetry.

# Training models

Following two sections shot how to train models we present in our paper.
By running them, it will train models with hyperparameter combinations which we used to obtain our results.

## Fine-tuning
Following command starts fine-tuning on cross-lingual summarization.
This produces models which called CLSum in our paper.

```sh
# CLSum for German
> python src/train.py --ip-type none --lang de --default-root-dir /path/to/save/checkpoints

# Model detailed description of the script.
> python src/train.py -h
```

## Intermediate fine-tuning + fine-tuning.

Following command starts training pipeline composed of

1. Intermeidate fine-tuning
2. Fine-tuning

.

This produces models which called CLSum+EnSum or CLSum+MT in our paper.
Set `--ip-type` `task` for CLSum+EnSum, `abst_mt` for CLSum+MT.
And choose one of de, it, zh or ja for `--lang`.

```sh
# CLSum+EnSum for German.
> python src/train.py --ip-type task --lang de --default-root-dir /dirpath/to/save/checkpoints

# CLSum+MT for German.
> python src/train.py --ip-type abst_mt --lang de --default-root-dir /dirpath/to/save/checkpoints

# Model detailed description of the script.
> python src/train.py -h
```

## Fewshot-learning

We also performed fewshot learning in our work.
Following command will fine-tune a model from a given checkpoint in limited amount of data (1, 5, 30%).

```sh
# Trains given model on 
> python src/train_fewshot.py \
>        --lr 1e-05 \
>        --seed 1122 \
>        --grad-accm 8 \
>        --max-epochs 5 \
>        --lang de \
>        --default-root-dir /dirpath/to/save/checkpoints \
>        --ckpt-path /path/to/checkpoint

# Model detailed description of the script.
> python src/train_fewshot.py -h
```


# Decoding

Given train models, by running the following commands you can perform summarization for given documents.

A file to pass for `--src-path` should contain one document for each line.
There are validation/test input files we used for our paper under `./data/inputs`.

```sh
# Cross-lingual summarization into German.
> python src/decode.py \
>        --src-path ./input.txt \
>        --base-dir /dirpath/you/saved/checkpoints \
>        --save-dir /dirpath/to/save/hypos \
>        --tgt-lang de \
>        --use-gpu

# Model detailed description of the script.
> python src/decode.py -h
```
