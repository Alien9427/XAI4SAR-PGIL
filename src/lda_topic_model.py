from gensim.models import LdaModel
from gensim.test.utils import common_corpus
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import MiniBatchKMeans, KMeans
import joblib
from sklearn.preprocessing import OneHotEncoder
import cv2
import time
import pandas as pd
from collections import Counter

SCAT_CLASS = 9

def my_hist_fun(data):
    global SCAT_CLASS
    hist = []
    N = data.size
    for i in range(SCAT_CLASS):
        hist.append(len(np.where(data == i)[0]) / N)

    return np.array(hist)

def gen_scat_word(scat_patch, win, stride):
    """
    generate the visual words from scattering labels
    :param scat_result_root:
    :param win:
    :return:
    """
    size_x, size_y = scat_patch.shape
    hist_list = []
    for i in range(0, size_x-win+1, stride):
        for j in range(0, size_y-win+1, stride):
            cropped = scat_patch[i:i+win, j:j+win].flatten()
            hist = my_hist_fun(cropped)
            hist_list.append(hist)

    return np.array(hist_list)

def gen_corpus(docs, kmeans, win, stride):
    """
        from docs and vocab, generate the corpus
    :param docs:
    :param kmeans:
    :param win: word_win
    :param stride:
    :return:
    """

    vocab_size = kmeans.cluster_centers_.shape[0] # shape: center_num * dimension
    num, size_x, size_y = docs.shape

    corpus = []
    for k in range(num):
        doc = docs[k,]
        hist = []
        for i in range(0, size_x-win+1, stride):
            for j in range(0, size_y-win+1, stride):
                cropped = doc[i:i+win, j:j+win].flatten()
                hist.append(my_hist_fun(cropped))
        hist_np = np.array(hist)
        del hist
        word_labels = kmeans.predict(hist_np)
        word_scores = np.exp(-kmeans.transform(hist_np))
        temp = OneHotEncoder(sparse=False, handle_unknown='ignore', categories=np.arange(vocab_size).reshape([1,vocab_size]))\
                   .fit_transform(word_labels.reshape([-1,1])) * word_scores
        corpus.append(np.sum(temp, axis=0))

    return np.array(corpus)

def transform_gensim_corpus(corpus):
    num_docs, num_words = corpus.shape
    gensim_corpus = []
    for i in range(num_docs):
        item = corpus[i,]
        gensim_corpus.append([(j, item[j]) for j in range(num_words)])
    return gensim_corpus


def lda_test_doc(doc_bow, lda):
    topic = lda.get_document_topics(doc_bow, minimum_probability=0.1)

    return topic


if __name__ == '__main__':
    """
        LDA for L(x)
        
    input: Dataset "ICE_dataset.txt"
    output: LDA model "ICE_lda_175.pkl"
            Vocabulary "ICE_kmeans.pkl"
    
    param instructions:
    1. doc,             shape (N_docs * size_x * size_y)
    2. kmeans.cluster_centers_
        (vocab),        shape (N_words * Length_words)
    3. corpus,          shape (N_docs * N_words)
    4. gensim_corpus,   shape ()
    
    """

    timer = time.perf_counter()
    print('=== load data ===\n')

    data_root = '../data/SeaIceData/'
    data_txt = '../data/ICE_dataset.txt'
    word_win = 8  # word patch

    data = pd.read_csv(data_txt)

    word_hist_all = np.array([])
    docs = np.array([])
    for idx in range(len(data)):
        scat_patch = np.load(data_root + data.loc[idx]['path'] + '_scat.npy') # ICE
        hist_idx = gen_scat_word(scat_patch, win=word_win, stride=word_win)
        if idx == 0:
            word_hist_all = hist_idx
            docs = scat_patch.reshape([1, scat_patch.shape[0], scat_patch.shape[1]])
        else:
            word_hist_all = np.concatenate((word_hist_all, hist_idx), axis=0)
            if scat_patch.shape == docs.shape[1:]:
                docs = np.concatenate((docs, scat_patch.reshape([1, scat_patch.shape[0], scat_patch.shape[1]])), axis=0)

    np.save('../data/ICE_docs.npy', docs)
    docs = np.load('../data/ICE_docs.npy')

    print(str(time.perf_counter() - timer))
    timer = time.perf_counter()


    print('=== generate vocab with kmeans === \n')
    kmeans = KMeans(n_clusters=500, n_init=20)
    kmeans = kmeans.fit(word_hist_all)

    """ save kmeans model
    """
    joblib.dump(kmeans, '../result/ICE_kmeans.pkl')
    kmeans = joblib.load('../result/ICE_kmeans.pkl')

    print(str(time.perf_counter() - timer))
    timer = time.perf_counter()

    """ from docs and vocab, generate corpus
    """
    print('=== generate corpus === \n')
    corpus = gen_corpus(docs, kmeans, word_win, stride=word_win)
    np.save('../data/ICE_corpus.npy', corpus)

    print(str(time.perf_counter() - timer))
    timer = time.perf_counter()

    """ LDA
    """
    print('=== LDA training === \n')

    corpus = np.load('../data/ICE_corpus.npy')
    gensim_corpus = transform_gensim_corpus(corpus)
    lda = LdaModel(corpus=gensim_corpus, num_topics=175)
    lda.save('../result/ICE_lda_175.pkl')
