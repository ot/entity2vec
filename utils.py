import os
import sys

from ext.gensim.models import Word2Vec

def load_word2vec_model(filename, mmap=None):
    if filename.endswith('.bin'):
        model = Word2Vec.load_word2vec_format(filename, binary=True)
    else:
        model = Word2Vec.load(filename, mmap=mmap)

    return model

