# Steps to create feature vectors for classification

1. Run `preprocess_olid.py` to generate preprocessed tweets

```
python preprocess_olid.py --file olid-training-v1.0
```

2. Download pre-trained [Twitter Glove2Vec embeddings](https://nlp.stanford.edu/projects/glove/) and convert it to Word2Vec format so it can be loaded by Gensim

```
python -m gensim.scripts.glove2word2vec --input  glove.twitter.27B.200d.txt --output glove.twitter.27B.200d.w2vformat.txt
```

3. Run `featurize_tweets.py` to create feature embeddings

```
python featurize_tweets.py --preprocessed_data preprocessed_olid-training-v1.0.tsv --train_indices train_ids.txt --val_indices val_ids.txt --embedding_path glove.twitter.27B.200d.w2vformat.txt --embedding_size 200
```

4. Run `classify_tweets.py` to classify validation dataset
```
python classify_tweets.py --preprocessed_data preprocessed_olid-training-v1.0.tsv --train_indices train_ids.txt --val_indices val_ids.txt --train_vectors train_vectors.npy --val_vectors val_vectors.npy
```
