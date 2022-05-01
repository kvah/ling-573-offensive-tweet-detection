# ling-573-group-repo
An end-to-end system for classifying English tweets as offensive or non-offensive, based on the [OffensEval 2019 Shared Task](https://sites.google.com/site/offensevalsharedtask/offenseval2019) (subtask A).

In the current version, we use concatenated GloVe embeddings pre-trained on Twitter data as sentence representations, and classify using logistic regression. 

## Instructions

### 1. Prerequisites

If necessary, download and install anaconda by running the following commands:
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh
```

Download the [pre-trained Twitter Glove2Vec Embeddings](https://nlp.stanford.edu/projects/glove/) and place `glove.twitter.27B.200d.txt` in `data/`. 
Then, convert it to Word2Vec format so it can be loaded to Gensim:
```
python -m gensim.scripts.glove2word2vec --input data/glove.twitter.27B.200d.txt --output data/glove.twitter.27B.200d.w2vformat.txt
```

### 2. Create the conda environment

``` 
conda env create -f env.yml --prefix env
```

### 3. Run the condor script

```
condor_submit D2.cmd
```

This script does the following:

1. Pre-processes OLID data and splits it into train and validation sets.
2. Converts GloVe embeddings to w2v format.
3. Converts tweets into feature vectors by concatenating pre-trained embeddings.
4. Trains logistic regression classifier on train set.
5. Evaluates classifier on validation set and outputs results.
