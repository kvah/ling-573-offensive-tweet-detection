"""
    Module for pre-processing OLID dataset (Task A)
    
    To process English data, execute:
        python3 preprocess_olid.py --file FILE [--language LANG] 
        [--all_train] [--all_test] [--train_ids TRAIN_IDS --val_ids DEV_IDS] 
        [--split_punctuation] [--remove_apostraphes] [--remove_hashtags] 
        [--split_emojis] [--convert_negation] [--convert_emojis]
        
    To process Greek data, execute:
        python3 preprocess_olid.py --file FILE --language greek
        [--all_train] [--all_test] [--train_ids TRAIN_IDS --val_ids DEV_IDS] 
        [--split_punctuation] [--remove_apostraphes] [--remove_hashtags] 
        [--convert_unicode] [--remove_diacritics] [--lemmatize]
    
    * Note for processing Greek data: To remove diacritics, choose either 
      [--convert_unicode] or [--remove_diacritics]. [--convert_unicode]
      converts unicode data into ASCII characters, which might result in OOV.
      
    * [--all_train] means the input file is the entire training data and no
       training/dev id is provided
       
    * [--all_test] means the input file is the entire test data and no
       training/dev id is provided
    
"""

import argparse
import re
import pandas as pd
from sys import argv
import emoji
import spacy
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from spacymoji import Emoji
import unidecode
import unicodedata

import time

start_time = time.time()

# download nltk data
nltk.download('omw-1.4')
nltk.download('wordnet')

# load spacy model
nlp = spacy.load("en_core_web_sm", disable = ["lemmatizer", "ner", "textcat"])

# add pipe to spacy model
nlp.add_pipe("emoji", first=True)

# define constants
OFF = "OFF"
NOT = "NOT"

# define types
Example = {"content":str, "label":int}

def lemmatiz_greek(content):
    """
    Lemmatize greek tweets
    
    Example:
        tweet: "Πώς τα πάνε οι γυναίκες με το αυτοκίνητο;"
        new tweet: "πώς ο πάνες ο γυναίκα με ο αυτοκίνητο ;"
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    content : pd.Series
        pandas Series containing lemmatized greek weets

    """
    nlp = spacy.load("el_core_news_sm", disable = ["parser", "ner"])
    for i, line in enumerate(content):
        doc = nlp(line)
        line = " ".join([token.lemma_ for token in doc])
        content[i] = line

    return content

def convert_unicode(content: pd.Series) -> pd.Series:
    """
    Convert unicode data into ASCII characters
    
    Example:
        tweet: "@USER Οι μουσουλμάνες που τις βιάζουν έτσι κ α..."
        new tweet: "@USER Oi mousoulmanes pou tis biazoun etsi k a..."
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    content : pd.Series
        pandas Series containing tweets with characters converted into ASCII characters

    """

    for i, line in enumerate(content):
        content[i] = unidecode.unidecode(line)

    return content

def remove_diacritics(content: pd.Series) -> pd.Series:
    """
    Remove diacritics
    
    Example:
        tweet: "@USER Οι μουσουλμάνες που τις βιάζουν έτσι κ α..."
        new tweet: "@USER Οι μουσουλμανες που τις βιαζουν ετσι κ α..."
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    content : pd.Series
        pandas Series containing tweets without diacritics

    """

    for i, line in enumerate(content):
        line = "".join(ch for ch in unicodedata.normalize('NFKD', line) if not unicodedata.combining(ch))
        content[i] = line

    return content

def emoji2des(line: str) -> str:
    """
    Convert emojis into corresponding descriptions
    
    Example:
        tweet: "@USER 🤷🏼‍♀️ that would be silly."
        new tweet: "@USER that would be silly . woman shrugging medium-light skin tone"
    
    Parameters
    ----------
    line: str
        a tweet

    Returns
    -------
    converted_line 
        a tweet with emojis converted into corresponding descriptions

    """
    doc = nlp(line)
    tokens = [token.text for token in doc]
    # convert line if emojis are found
    if doc._.has_emoji:
        emoji_des_set = set()
        for doc_emoji in doc._.emoji:
            emoji_des_set.add(doc_emoji[2])
            tokens[doc_emoji[1]] = ""
        tokens = [token for token in tokens if token]

        # attach emoji descriptions to the end of the line
        tokens += list(emoji_des_set)

    converted_line = " ".join(tokens)
    return converted_line

def convert_emojis(content: pd.Series) -> pd.Series:
    """
    Convert emojis in tweets into emoji descriptions

    Example:
        tweet: "@USER 🤷🏼‍♀️ that would be silly."
        new tweet: "@USER that would be silly . woman shrugging medium-light skin tone"
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    content : pd.Series
        pandas Series containing tweets with emojis converted
        
    * Note1: If an emoji occurs multiple times, only one will be kept
    * Note2: Emoji descriptions are attached to the very end of the line.
             (might result in ungrammatical sentences) Therefore, this 
             preprocessing method should be used after all other methods.
    * Note3: Punctuations are separated from words.

    """
    for i, line in enumerate(content):
        content[i] = emoji2des(line)
        
    return content

def get_antonyms(syn: nltk.corpus.reader.wordnet.Synset) -> list:
    """
    Get the antonyms of a wordnet synset 
    
    Parameters
    ----------
    syn: nltk.Synset
        a wordnet synset

    Returns
    -------
    antonyms: list 
        a list of antonyms of the input synset (if applicable);
        an empty list is returned if no antonyms are found

    """
    antonyms = set()
    for lem in syn.lemmas():
        if lem.antonyms():
            antonyms.add(lem.antonyms()[0].name())
    antonyms = list(antonyms)
    return antonyms

def convert_negated_sent(sent: str) -> str:
    """
    Replace negated adj/adv with their antonyms 
    
    Parameters
    ----------
    sent: str
        a tweet

    Returns
    -------
    sent: str 
        a sentence with the negation (and the adj/adv following it) replaced

    """
    pos_covert = {"ADJ": "a", "ADV": "r"}
    doc = nlp(sent)

    # tokenize sentence and get negation tokens
    tokenized_sent = []
    negation_tokens = []
    for token in doc:
        tokenized_sent.append(token.text)
        if token.dep_ == "neg":
            negation_tokens.append(token)
    
    # get the heads of the negation tokens
    negation_head_tokens = [token.head for token in negation_tokens]

    if negation_tokens:
        head_mod_children = []
        head_mod_children_ant = []
        for token in negation_head_tokens:
            head_mod_children += [child for child in token.children if child.pos_ == "ADJ" or child.pos_ == "ADV"] # get the adj or adv
        
        if head_mod_children:
            # collect antonyms for all the adjs/advs (one for each of them; if applicable)
            for mod_child in head_mod_children:
                synset = lesk(tokenized_sent, mod_child.text, pos = pos_covert[mod_child.pos_])
                if not synset and mod_child.pos_ == "ADJ":
                    synset = lesk(tokenized_sent, mod_child.text, pos = pos_covert[mod_child.pos_])

                if not synset: # leave the sentence unchanged if no synset is found for the adj/adv
                    return " ".join(tokenized_sent)
                
                antonyms = get_antonyms(synset)
                # append one antonym if any; append an empty string if no antonyms are found
                if antonyms:
                    head_mod_children_ant.append(antonyms[0])
                else:
                    head_mod_children_ant.append("")
            
            # replace adjs/advs with their antonyms
            for idx, mod_child in enumerate(head_mod_children):
                tokenized_sent[mod_child.i] = head_mod_children_ant[idx]
            
            # remove negation tokens
            for neg_token in negation_tokens:
                tokenized_sent[neg_token.i] = ""
    
    # get rid of emoty strings
    tokenized_sent = [token for token in tokenized_sent if token]
    return " ".join(tokenized_sent)

def convert_negation(content: pd.Series) -> pd.Series:
    """
    Convert tweets with negation into non-negated form

    Example:
        tweet: "@USER @USER He is not good in Debate but very good in Dancing Competition."
        new tweet: "@USER @USER He is evil in Debate but very good in Dancing Competition ."
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    content : pd.Series
        pandas Series containing tweets with negated tweets converted
    
    * Note: Punctuations are separated from words.

    """
    for i, line in enumerate(content):
        content[i] = convert_negated_sent(line)
    
    return content

def split_punctuation(content: pd.Series) -> pd.Series:
    """
    Add whitespace seperation between and punctuation. 

    Example: 
        tweet: "By the way, I don't agree with your argument. #livid 👊👊"
        new tweet: "By the way , I don't agree with your argument. 👊👊"
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    cleaned_content: pd.Series: 
        pandas Series containing cleaned tweets

    """
    cleaned_content = content.replace({r'(\w+)([!?.])': r'\1 \2'}, regex=True)
    return cleaned_content 

def remove_apostraphes(content: pd.Series) -> pd.Series:
    """
    Remove apostraphe from all contractions

    Example: 
        tweet: "By the way, I don't agree with your argument. #livid 👊👊"
        new tweet: "By the way, I dont agree with your argument. #livid 👊👊"
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    cleaned_content: pd.Series: 
        pandas Series containing cleaned tweets

    """
    cleaned_content = content.replace({r"(\w+)[’'](\w+)": r'\1\2'}, regex=True)
    return cleaned_content

def remove_hashtags(content: pd.Series) -> pd.Series:
    """
    Remove hashtags from their attached phrase

    Example: 
        tweet: "By the way, I don't agree with your argument. #livid 👊👊"
        new tweet: "By the way, I don't agree with your argument. livid 👊👊"
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    cleaned_content: pd.Series: 
        pandas Series containing cleaned tweets

    """
    cleaned_content = content.replace({r"#(\w+)": r'\1'}, regex=True)
    return cleaned_content

def split_emojis(content: pd.Series) -> pd.Series:
    """
    Add whitespace seperation between consecutive emojis

    Example: 
        tweet: "By the way, I don't agree with your argument. #livid 👊👊"
        new tweet: "By the way, I don't agree with your argument. #livid  👊  👊 "
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    content_copy: pd.Series: 
        pandas Series containing cleaned tweets

    """
    # Note: I'm getting weird behavior when running this on patas.. may fix 
    # later on but it doesn't yield much improvement anyway.
    emoji_with_space = {emo: f' {emo} ' for emo in emoji.UNICODE_EMOJI}

    cleaned_content = content.copy()
    for emo, emo_spaced in emoji_with_space.items():
        try:
            cleaned_content = cleaned_content.str.replace(emo, emo_spaced, regex=True)
        except Exception:
            print(f'Failed to parse emoji: {emo}')
    return cleaned_content

def preprocess(data: pd.DataFrame, lang: str="english") -> list:
    """
    Converts OLID-formatted file into a list of datapoints.

    Parameters
    ----------
    file : str
        tsv file containing data (OLID format)
    lang : str
        language of data. default is "english"

    Returns
    -------
    list of dicts: {'content': str, 'label': int}
        content is tweet
        label is 1 for offensive, 0 for not offensive

    """
    
    # cut out everything but tweet and label
    tweets = data[["tweet", "subtask_a"]]

    tweets_copy = tweets['tweet'].copy()
    # general preprocessing tasks
    if args.split_punctuation:
        tweets_copy = split_punctuation(tweets_copy)
    if args.remove_apostraphes:
        tweets_copy = remove_apostraphes(tweets_copy)
    if args.remove_hashtags:
        tweets_copy = remove_hashtags(tweets_copy)
    if args.split_emojis:
        tweets_copy = split_emojis(tweets_copy)

    # language specific preprocessing methods
    if lang == "english":
        if args.convert_negation:
            tweets_copy = convert_negation(tweets_copy)
        if args.convert_emojis: # should be the last method being applied
            tweets_copy = convert_emojis(tweets_copy)
    elif lang == "greek":
        if args.lemmatize:
            tweets_copy = lemmatiz_greek(tweets_copy)
        if args.convert_unicode:
            tweets_copy = convert_unicode(tweets_copy)
        if args.remove_diacritics:
            tweets_copy = remove_diacritics(tweets_copy)

    tweets['tweet'] = tweets_copy
    data_list = []
    
    for index, row in tweets.iterrows():
        if row["subtask_a"] == OFF:
            label = 1
        else:
            label = 0
        
        data_list.append({'content': row["tweet"], 'label': label})
        
    return(data_list)


def write_file(data: list, out_file: str) -> None:
    """
    Writes tweets and labels to tsv file

    """
    file = open(out_file, mode="w+", encoding="utf-8")
    # header
    file.write("content\tlabel\n")
    
    # write data
    for line in data:
        file.write(f"{line['content']}\t{line['label']}\n")
        
    file.close()

def split_data(data: list, train_ids_file: str, 
               val_ids_file: str) -> [list, list]:
    """
    Splits canonical training data into train and dev

    Parameters
    ----------
    data : list
        OLID data, must include 'id' field
    train_ids_file : str
        line-separated file containing ids corresponding to training set
    val_ids_file : str
        line-separated file containing ids corresponding to dev set

    Returns
    -------
    train_set : list
    test_set : list

    """
    train_ids = [int(idx.strip()) for idx in open(train_ids_file).readlines()]
    val_ids = [int(idx.strip()) for idx in open(val_ids_file).readlines()]
    
    train_set = [data[i] for i in train_ids]
    val_set = [data[i] for i in val_ids]
    
    return(train_set, val_set)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Pre-process OLID dataset for Task A.")
    parser.add_argument(
        "--file", type=str, required=True, help="path to data (OLID format)")
    parser.add_argument(
        "--language", type=str, default="english", 
        choices=["english", "arabic", "danish","greek", "turkish"],
        help="language of data")
    parser.add_argument(
        "--train_ids", default=None, 
        help="path to file containing IDs for training set (line-separated)")
    parser.add_argument(
        "--val_ids", default=None, 
        help="path to file containing IDs for development set (line-separated)")

    parser.add_argument(
        "--all_train", action="store_true",
        help="whether the input file is the entire training data"
    )
    parser.add_argument(
        "--all_test", action="store_true",
        help="whether the input file is the entire test data"
    )
    parser.add_argument(
        "--split_punctuation", action="store_true",
        help="whether to split punctuation from the end of tokens"
    )
    parser.add_argument(
        "--remove_apostraphes", action="store_true",
        help="whether to remove apostraphe from contractions"
    )
    parser.add_argument(
        "--remove_hashtags", action="store_true",
        help="whether to remove the # symbol from hashtags"
    )
    parser.add_argument(
        "--split_emojis", action="store_true",
        help="whether to split sequence of emojis by whitespace"
    )
    parser.add_argument(
        "--convert_negation", action="store_true",
        help="whether to convert negated sentences into non-negated forms"
    )
    parser.add_argument(
        "--convert_emojis", action="store_true",
        help="whether to convert emojis into corresponding text descriptions"
    )
    parser.add_argument(
        "--convert_unicode", action="store_true",
        help="whether to convert unicode data into ascii characters"
    )
    parser.add_argument(
        "--remove_diacritics", action="store_true",
        help="whether to remove diacritics"
    )
    parser.add_argument(
        "--lemmatize", action="store_true",
        help="whether to lemmatize tweets"
    )

    args = parser.parse_args(argv[1:])
    lang = args.language
    
    file = args.file
    file_ending = re.search(r"([^//]*)$", args.file).groups()[0]
    
    data = pd.read_csv(file, sep="\t", header=0, encoding="utf8")
    
    # preprocess the entire data
    preprocessed_data = preprocess(data, lang)

    # split data if specified
    if args.train_ids != None:
        train_ids = args.train_ids
        val_ids = args.val_ids

        train_preprocessed, val_preprocessed = split_data(preprocessed_data, train_ids, val_ids)

        train_csv_path = f'data/clean_train_{args.language}.tsv'
        val_csv_path = f'data/clean_val_{args.language}.tsv'

        print(f"Writing preprocessed training data to: {train_csv_path}")
        print(f"Writing preprocessed validation data to: {val_csv_path}")
        write_file(train_preprocessed, train_csv_path)
        write_file(val_preprocessed, val_csv_path)
        
    else:
        if args.all_train:
            write_file(preprocessed_data, f"data/clean_all_train_{args.language}.tsv")
        elif args.all_test:
            write_file(preprocessed_data, f"data/clean_all_test_{args.language}.tsv")
            
    print(f'Preprocessing Time Elapsed: {time.time() - start_time} seconds')
