import logging
from collections import defaultdict, Counter
from multiprocessing.pool import Pool
from functools import partial

import numpy as np
from scipy.optimize import lbfgsb
import theano
import theano.tensor as T

from ext.gensim.utils import SaveLoad

from array_mmap_proxy import ArrayMMapProxy


def make_logistic_likelihood():
    global log_l

    lbda = T.dscalar('lbda')
    w = T.dvector('w')
    data = T.fmatrix('data')
    pos_counts = T.vector('pos_counts', dtype='uint32')
    neg_counts = T.vector('neg_counts', dtype='uint32')

    fw = T.cast(w, 'float32') # lbfgsb only works with double

    prods = T.dot(fw[:-1], data.T) + fw[-1]
    ll = T.sum(pos_counts * T.log(1 + T.exp(prods))) + \
         T.sum(neg_counts * T.log(1 + T.exp(-prods)))
    ll_reg = ll + lbda * T.sum(fw[:-1]**2) / (fw.shape[0] - 1)

    ll_reg_grad = T.cast(T.grad(ll_reg, wrt=w), 'float64') # again

    log_l = theano.function([w, data, pos_counts, neg_counts, lbda],
                            (ll_reg, ll_reg_grad),
                            mode='FAST_RUN')


class EntityModel(SaveLoad):
    def __init__(self):
        self.vectors = None
        self.entities = None

    def _train_base(self, compute_vector, entity_word_seqs):
        pool = Pool()

        entities = {}
        vectors = []

        def idx_seqs():
            for idx, (entity, seq) in enumerate(entity_word_seqs):
                entities[entity] = idx
                yield seq

        for vec in pool.imap(compute_vector, idx_seqs()):
            vectors.append(vec)

            if len(vectors) % 1000 == 0:
                logging.info('Computed %d vectors', len(vectors))

        self.entities = entities
        self.vectors = np.asarray(vectors)


    def _count_word_vectors(self, model, word_idxs):
        counts = Counter(word_idxs)
        word_vecs = model.syn0[counts.keys()]
        word_counts = counts.values()

        return word_vecs, word_counts


def sample(bins, size):
    points = np.random.randint(bins[-1], size=size)
    return np.searchsorted(bins, points)


def compute_vector_lr(syn0_proxy, bins_proxy,
                      neg_words_mult, lbda, word_idxs):
    syn0 = syn0_proxy.get()
    bins = bins_proxy.get()

    counts = defaultdict(lambda: np.zeros(2).astype(np.uint32))

    for widx in word_idxs:
        counts[widx][0] += 1

    neg_words_idxs = sample(bins, int(neg_words_mult * len(word_idxs)))
    for neg_widx in neg_words_idxs:
        counts[neg_widx][1] += 1

    vectors = syn0[counts.keys()]
    count_pairs = np.vstack(counts.values())

    f = lambda w, params=(vectors, count_pairs[:, 0], count_pairs[:, 1], lbda): log_l(w, *params)

    x0 = np.zeros(syn0.shape[1] + 1)
    opt = lbfgsb.fmin_l_bfgs_b(f, x0)

    if opt[2]['warnflag']:
        logging.debug('Error in optimization: %s', opt[2])

    lr_vec = opt[0].astype(np.float32)
    if not np.all(np.isfinite(lr_vec)):
        logging.info('Error computing lr vector')
        lr_vec[:] = 0

    return lr_vec


class EntityModelLR(EntityModel):

    def __init__(self, neg_cdf, neg_words_mult, lbda):
        super(EntityModel, self).__init__()

        self.neg_cdf = neg_cdf
        self.neg_words_mult = neg_words_mult
        self.lbda = lbda


    def train(self, model, entity_word_seqs):
        make_logistic_likelihood()
        self._train_base(partial(compute_vector_lr,
                                 ArrayMMapProxy(model.syn0),
                                 ArrayMMapProxy.fromarray(self.neg_cdf),
                                 self.neg_words_mult,
                                 self.lbda),
                         entity_word_seqs)


    def score(self, model, entity_vecs, words_idxs):
        word_vecs, word_counts = self._count_word_vectors(model, words_idxs)

        weights = entity_vecs[:, :-1]
        bias = entity_vecs[:, -1:]
        dotprods = np.dot(weights, word_vecs.T)
        log_dot_prods = np.log(1 + np.exp(dotprods + bias))
        return -np.dot(word_counts, log_dot_prods.T)


def compute_vector_centroid(syn0_proxy, word_idxs):
    syn0 = syn0_proxy.get()

    counts = Counter(word_idxs)

    centroid_vec = np.dot(counts.values(), syn0[counts.keys()])
    centroid_vec /= np.sqrt(np.sum(centroid_vec**2))

    if not np.all(np.isfinite(centroid_vec)):
        logging.info('Error computing centroid vector')
        centroid_vec[:] = 0

    return centroid_vec


class EntityModelCentroid(EntityModel):

    def __init__(self):
        super(EntityModel, self).__init__()


    def train(self, model, entity_word_seqs):
        self._train_base(partial(compute_vector_centroid,
                                 ArrayMMapProxy(model.syn0)),
                         entity_word_seqs)


    def score(self, model, entity_vecs, words_idxs):
        word_vecs, word_counts = self._count_word_vectors(model, words_idxs)

        centroid = np.dot(word_counts, word_vecs)
        centroid /= np.sqrt(np.sum(centroid**2))
        return np.dot(entity_vecs, centroid)


