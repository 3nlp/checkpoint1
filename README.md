# Checkpoint1: 11-747 Neural Networks for NLP
This repository includes a simple attention driven encoder-decoder model for the task of Machine Reading Comprehension on the 
MS-MARCO dataset. 

## Dataset download and reference papers
The MS MARCO dataset, which was released at NIPS 2016, can be downloaded from http://www.msmarco.org/. This website also hosts a leaderboard for the best models evaluated to date on the MS MARCO v1.1 test set based on the BLEU-1 and ROUGE-L scores, with references to each of their works (where available).

## Instructions for running the model
MS Marco provides the train,dev and test datasets in the files train.json,dev.json and test.json. Data preprocessing.py reads these text files to produce pickle files containing the word index vectors of the passage, question and answer components in the json files. We have provided two neural network models for training on the MS-Marco. The encoder-decoder.py contains a basic sequence to sequence architecture with a passage encoder,question encoder and a subsequent decoder. The attention.py replaces the decoder with an attention driven decoder network. Both these networks read in 2 files - samples.pkl and Vocab.pkl which can be obtained by running data preprocessing. The samples.pkl contains the word index vectors for the passage,question and answers and the Vocab.pkl contains the 30000 words used for training the model.

The metrics.py file reads in the trained weights of the network and completes one pass on the entire train/test datasets to produce the average BLeu and Rogue-L scores.

Please run python filename.py to run all the above files (make sure to run in order provided below).
```sh
$ CUDA_VISIBLE_DEVICES=0 python -u Data_preprocessing.py
# script that reads a json file (train.json/dev.json or test.json) and produces 
# 2 pickle files - samples.pkl and Vocab.pkl which will be required to run the 
# encoder-decoder.py / attention.py

$ CUDA_VISIBLE_DEVICES=0 python -u encoder-decoder.py
# script that takes in Vocabulary and samples files to train a basic encoder 
# decoder architecture

$ CUDA_VISIBLE_DEVICES=0 python -u attention.py
# script that takes in Vocabulary and samples files to train an attention driven 
# encoder decoder architecture

$ CUDA_VISIBLE_DEVICES=0 python -u metrics.p
# script to evaluate the model and obtain BLEU-1 and ROUGE-L metrics
