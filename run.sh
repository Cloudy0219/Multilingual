#!/bin/bash
conda activate memory

#python train.py --device 5 --model_name jhu-clsp/bernice

python train.py --device -2 --model_name cardiffnlp/twitter-roberta-base

python train.py --device 2 --model_name Twitter/twhin-bert-large