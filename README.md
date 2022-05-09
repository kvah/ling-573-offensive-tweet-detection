# ling-573-group-repo
An end-to-end system for classifying English tweets as offensive or non-offensive, based on the [OffensEval 2019 Shared Task](https://sites.google.com/site/offensevalsharedtask/offenseval2019) (subtask A).

## Changes in D3

### Additional Preprocessing
- Split punctuation from words
- Remove apostraphes from contractions
- Remove hashtags from tweets
- Detect negated phrases and replace them with their antonyms
- Convert Emojis to their [spacymoji](https://spacy.io/universe/project/spacymoji) text description

### Data
- Added scripts to under/over sample the training data to combat class imbalance

### Embeddings
- Added emoji2vec embeddings to handle OOV emojis in word2vec

### Classification
- Logistic Regression -> Bidirectional LSTM
- Model hyperparameter tuning

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

### 2. Create the conda environment and run the following commands

``` 
conda env create -f env.yml --prefix ./573_gpu
conda activate ./573_gpu
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
7. Saves the final f1-score in `outputs/D3/D3_val_preds.csv` 
