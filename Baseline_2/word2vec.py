# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/4/30 16:10
@author: LiFan Chen
@Filename: word2vec.py
@Software: PyCharm
"""
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from tqdm import tqdm


def seq_to_kmers(seq, k=3):
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
        k (int), default 3
    Returns:
        List containing a list of kmers.
    """
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]


class Corpus(object):
    """ An iteratable for training seq2vec models. """

    def __init__(self, dir, ngram):
        self.df = pd.read_csv(dir)
        self.ngram = ngram

    def __iter__(self):
        for sentence in self.df.Seq.values:
            yield seq_to_kmers(sentence, self.ngram)


def get_protein_embedding(model,protein):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((len(protein), 100))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i = 1 + i
    return vec

def Vec_sub(dataset):
    filepath = f'./data/{dataset}.txt'
    print(f"DATASET: {dataset}")
    print('read file...')
    with open(filepath,'r') as f:
        lines = f.readlines()
    newlines = []
    print('processing...')
    for line in tqdm(lines):
        linelist = line.replace('\n', '').replace('\r', '').strip().split()
        newline =str(linelist[1]) + '\n'
        newlines.append(newline)
    return newlines
    

def Vec():
    newlines = []
    for dataset in ['Davis', 'KIBA', 'DrugBank']:
        newlines.extend(Vec_sub(dataset))
    df = pd.DataFrame({'Seq':newlines})
    df.to_csv(f'dataset/train_vec.csv')

if __name__ == "__main__":

    Vec()
    sent_corpus = Corpus(f"dataset/train_vec.csv",3)
    model = Word2Vec(vector_size=100, window=5, min_count=1, workers=12)
    model.build_vocab(sent_corpus)
    model.train(sent_corpus,epochs=30,total_examples=model.corpus_count)
    model.save("word2vec_30_mod.model")

    """
    model = Word2Vec.load("word2vec_30.model")
    vector = get_protein_embedding(model,seq_to_kmers("MSPLNQSAEGLPQEASNRSLNATETSEAWDPRTLQALKISLAVVLSVITLATVLSNAFVLTTILLTRKLHTPANYLIGSLATTDLLVSILVMPISIAYTITHTWNFGQILCDIWLSSDITCCTASILHLCVIALDRYWAITDALEYSKRRTAGHAATMIAIVWAISICISIPPLFWRQAKAQEEMSDCLVNTSQISYTIYSTCGAFYIPSVLLIILYGRIYRAARNRILNPPSLYGKRFTTAHLITGSAGSSLCSLNSSLHEGHSHSAGSPLFFNHVKIKLADSALERKRISAARERKATKILGIILGAFIICWLPFFVVSLVLPICRDSCWIHPALFDFFTWLGYLNSLINPIIYTVFNEEFRQAFQKIVPFRKAS"))
    print(vector.shape)
    """