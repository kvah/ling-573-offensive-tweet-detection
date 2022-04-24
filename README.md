# ling-573-group-repo

## Instructions

### 1. Prerequisites

Download the [OLID (2019) dataset](https://sites.google.com/site/offensevalsharedtask/olid) and place `olid-training-v1.0.tsv` in `data/`

Download the [pre-trained Twitter Glove2Vec Embeddings](https://nlp.stanford.edu/projects/glove/) and place `glove.twitter.27B.200d.txt` in `data/`. 
Then, convert it to Word2Vec format so it can be loaded to Gensim:
```
python -m gensim.scripts.glove2word2vec --input data/glove.twitter.27B.200d.txt --output data/glove.twitter.27B.200d.w2vformat.txt
```

### 2. Install and activate the Conda Environment

``` 
conda env create -f env.yml
conda activate 573
```

### 3. Run the conda script

```
conda_submit D2.cmd
```
