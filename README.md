# Checkpoint1: 11-747 Neural Networks for NLP
This repository includes a simple attention driven encoder-decoder model for the task of Machine Reading Comprehension on the 
MS-MARCO dataset. 

## Dataset download and reference papers
The MS MARCO dataset, which was released at NIPS 2016, can be downloaded from http://www.msmarco.org/. This website also hosts a leaderboard for the best models evaluated to date on the MS MARCO v1.1 test set based on the BLEU-1 and ROUGE-L scores, with references to each of their works (where available).

## Instructions for running the model
```sh
$ git rm --cached localFileName
# add localFileName to .gitignore file 

$ CUDA_VISIBLE_DEVICES=0 python -u metrics.p
# script to evaluate the model and obtain BLEU-1 and ROUGE-L metrics
```

