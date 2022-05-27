# ling-573-group-repo
## Task Description

### Primary Task
An end-to-end system for classifying English tweets as offensive or non-offensive, based on the [OffensEval 2019 Shared Task](https://sites.google.com/site/offensevalsharedtask/offenseval2019) (subtask A).

### Adaptation Task
An end-to-end system for classifying Greek tweets as offensive or non-offensive, based on the [OffensEval 2020 Shared Task](https://sites.google.com/site/offensevalsharedtask/results-and-paper-submission) (subtask A).


## Changes in D4

### Primary Task
#### Embeddings and Classification
- GloVe embedding + Bidirectional LSTM -> RoBERTa-base model
- Model finetuning and hypertuning

### Adaptation Task
#### Additional Preproccessing
- Removing diacritics
- Convert unicode data into ASCII characters
- Lemmatization
#### Embeddings and Classification
- XML-RoBERTa model
- Model finetuning and hypertuning

## Instructions

### 1. Prerequisites
#### Install Anaconda
If necessary, download and install anaconda by running the following commands:
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh
```

#### Download best models for primary and adaptation tasks
- Download the [best model for primary task](https://drive.google.com/drive/u/2/folders/1KYS1PpH_jKT4wz94Kut1H7wnGopEI5Rb) and place the entire folder (containing `config.json` and `pytorch.bin`) in `models`
- Download the [best model for adaptation task](https://drive.google.com/drive/folders/1-BlV1p9GvdiQblCWJ_M-yjh4nszmYypw) and place the entire folder (containing `config.json` and `pytorch.bin`) in `models`
- Note that the model for *primary task* (the folder containing `config.json` and `pytorch.bin`) should be named `finetune_roberta` and the model for *adaptation task* should be named `finetune_xlmr_large_final_greek`
<!-- Download the [pre-trained Twitter Glove2Vec Embeddings](https://nlp.stanford.edu/projects/glove/) and place `glove.twitter.27B.200d.txt` in `data/`. 
Then, convert it to Word2Vec format so it can be loaded to Gensim:
```
python -m gensim.scripts.glove2word2vec --input data/glove.twitter.27B.200d.txt --output data/glove.twitter.27B.200d.w2vformat.txt
``` -->

### 2. Create the conda environment and run the following commands

- If the conda environment was not created previously, run the following:
``` 
conda create --prefix ./573_gpu python=3.8
conda activate ./573_gpu
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda env update --prefix ./573_gpu --file env.yml --prune
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --force-reinstall
```

- If the conda environment exists and you want to update the dependencies, run the following:
``` 
conda activate ./573_gpu
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda env update --prefix ./573_gpu --file env.yml --prune
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --force-reinstall
```

### 3. Run the Condor Script

```
condor_submit D3.cmd
```

**Important Notes**
- For the purposes of this deliverable, preprocessing and training are commented out from the main script (`D3_run.sh`). 
- Occasionally, the prediction script (`src/lstm_predict.py`) would cause the condor job to get stuck, which we started experiencing on the day of the deadline: 5/8/22. If this happens, running the bash script locally instead of through the condor job should work.


```
./D3_run.sh
```

In summary, the pipeline:
1. Pre-processes OLID data and splits it into train and validation sets.
2. Converts GloVe embeddings to w2v format.
3. Converts tweets into variable length sequences based on NLTK's TweetTokenizer
4. Initializes the weights of a BiLSTM with the pretrained GloVe embeddings
5. Trains the BiLSTM using the tweet sequences in the training set
6. Uses trained classifier to predict on validation set and output predictions in `outputs/D3/D3_val_preds.csv`
7. Saves the final f1-score in `results/D3_scores.out` 
