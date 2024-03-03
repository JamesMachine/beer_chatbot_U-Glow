import pandas as pd
import numpy as np
import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

file_path = "0303_imperial_ipa_rate.csv"

def dataframe_prep(file_path):

    df = pd.read_csv(file_path)
    df = df[df["comment"].notnull()]
    df = df.reset_index()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    df["sen_vec"] = df["comment"].map(lambda x: sen2vec_func(x))

    stop_words = set(stopwords.words('english')) # import a list of stopwords from nltk
    lemmatizer = WordNetLemmatizer() 

    df["normalized_comment"] = df["comment"].map(lambda x: clean_comment(x))

    tfidf_pipe = tfidf_func(df["normalized_comment"])
    df['tf_idf'] = df["normalized_comment"].map(lambda x: tfidf_pipe.transform([x]))

    nltk.download('averaged_perceptron_tagger')
    df["adj"] = df['normalized_comment'].map(lambda x: pos(x))


    return df



def sen2vec_func(beer_comment):
    return model.encode(beer_comment)



def clean_comment(doc):

    # lowercase + remove punctuation
    # string.punctuation: !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    # str.maketrans("OldStrig", "NewString", "DeleteString")
    ######### OldString & NewString have to have the same length
    doc = doc.translate(str.maketrans('', '', string.punctuation)).lower()
    doc = doc.replace("’", "")
    
    # remove digits
    # .isdigit() is checking if a string value is a number
    doc = ''.join([i for i in doc if not i.isdigit()])

    # tokenize + remove stopwords
    tokens = [word for word in doc.split() if word not in stop_words]

    # lemmatize tokens
    tokens = [lemmatizer.lemmatize(word, 'v') for word in tokens]
    tokens = [lemmatizer.lemmatize(word, 'a') for word in tokens]

    # convert lemmatized tokens back to string (one sentence)
    doc = ' '.join(tokens)

    return doc



def tfidf_func(normalized_comment):

    corpus = list(normalized_comment)
    vocabulary = set([token for sentence in corpus for token in sentence.split()])

    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                    ('tfid', TfidfTransformer())]).fit(corpus)

    return pipe
    


def pos(normalized_comment):

    tokens = nltk.word_tokenize(normalized_comment)
    pos_tags = nltk.pos_tag(tokens, tagset='universal')
    
    return [token[0] for token in pos_tags if token[1]=="ADJ"]


