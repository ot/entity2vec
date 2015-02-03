#!/usr/bin/env python

import os
import sys
import logging
from collections import Counter
import json
from collections import OrderedDict
from subprocess import check_output
import math
from copy import copy

import numpy as np
import numexpr as ne
from scipy.linalg import eigh

from ext import baker

from utils import load_word2vec_model
from analogy_benchmark import fast_accuracy
from entity_models import EntityModel, EntityModelCentroid, EntityModelLR


def golomb_bits(v):
    nat_v = ne.evaluate('where(v >= 0, 2 * v, -(2 * v + 1))')
    abs_sum = ne.evaluate('sum(nat_v + 1L)') # 1L to force conversion to int64
    if len(v) == abs_sum:
        return 0
    f = float(len(v)) / abs_sum
    m = int(math.ceil(math.log(2 - f) / -math.log(1 - f)))
    assert m > 0
    b = int(math.ceil(math.log(m, 2)))
    return float(ne.evaluate('sum(nat_v / m + 1 + where(nat_v < 2**b - m, b - 1, b))'))


def quantize_array(A, q):
    quant_A = (np.abs(A) * q).astype(np.int32)
    quant_A *= np.sign(A)

    dequant_A = np.array(quant_A, dtype=np.float32)
    dequant_A += 0.5 * np.sign(quant_A)
    dequant_A /= q

    return quant_A, dequant_A


def quantize_array_lbda(A, lbda):
    best_cost = None
    best = None

    for logq in xrange(12, 24):
        q = int(1.2**logq)
        quant_A, dequant_A = quantize_array(A, q)
        bits = golomb_bits(quant_A)
        err = ne.evaluate('sum((A - dequant_A)**2)')
        cost = err + lbda * bits
        if best_cost is None or best_cost > cost:
            best_cost = cost
            best = q, bits, quant_A, dequant_A

    return best


def quantize_array_target(A, target_err):
    low = 1
    high = 128
    A_norms = np.sqrt(np.sum(A**2, axis=-1))

    while high - low > 1:
        mid = (high + low) / 2
        quant_A, dequant_A = quantize_array(A, mid)
        mean_err = np.mean(np.sqrt(np.sum((dequant_A - A)**2, axis=-1)) / A_norms)
        logging.info("Binary search: q=%d, err=%.3f", mid, mean_err)

        if mean_err > target_err:
            low = mid
        else:
            high = mid

    return mid, mean_err, quant_A, dequant_A


def quantize(model, target_err, transform=False, inv_transform=False):
    model.init_sims()

    A = model.syn0norm
    if transform:
        ATA = A.T.dot(A)
        _, V = eigh(ATA)
        A = V.T.dot(A.T).T

    q, mean_err, quant_syn0, dequant_syn0 = quantize_array_target(A, target_err)
    bits = [golomb_bits(col) for col in quant_syn0.T]
    zeros = np.sum(quant_syn0 == 0)

    if transform and inv_transform:
        dequant_syn0 = V.dot(dequant_syn0.T).T

    dequant_model = copy(model)
    dequant_model.syn1 = None
    dequant_model.syn0 = dequant_syn0
    dequant_model.syn0norm = dequant_syn0

    return q, sum(bits), zeros, mean_err, quant_syn0, dequant_model


def entropy(a, round_bits=False):
    freqs = np.array(Counter(a).values(), dtype=np.float32)
    total = np.sum(freqs)
    avg_bit_lengths = np.log2(total / freqs)
    if round_bits:
        avg_bit_lengths = np.ceil(avg_bit_lengths)
    e = np.sum(freqs * avg_bit_lengths)
    return e


def save_vectors(f, vocab, quant_syn0, q):
    print >> f, '%s\t%s\t%s\t' % (len(vocab), quant_syn0.shape[1], q)

    for w in vocab:
        print >> f, w.encode('utf8')

    np.savetxt(f, quant_syn0, fmt='%d')


def quantize_entities(entity_model, target_err):
    A = entity_model.vectors
    q, mean_err, quant_vecs, dequant_vecs = quantize_array_target(A, target_err)
    bits = [golomb_bits(col) for col in quant_vecs.T]
    zeros = np.sum(quant_vecs == 0)

    dequant_model = copy(entity_model)
    dequant_model.entities = entity_model.entities
    dequant_model.vectors = dequant_vecs

    return q, sum(bits), zeros, mean_err, quant_vecs, dequant_model


@baker.command
def quant(input_file, output_template=None, target_err=0.1, transform=True, test_accuracy=None):
    model = load_word2vec_model(input_file, mmap='r')

    q, pred_bits, zeros, avg_err, quant_syn0, dequant_model = quantize(model, target_err, transform)
    pred_bps = float(pred_bits) / quant_syn0.size
    avg_zeros = float(zeros) / quant_syn0.size

    if output_template is not None:
        output_filename = '%s.e%.3f.%s' % (output_template, target_err, 'tr' if transform else 'nt')
        with open(output_filename + '.txt', 'w') as fout:
            save_vectors(fout, model.index2word, quant_syn0, q)

        dequant_model.save(output_filename + '.model')

    acc = None
    if test_accuracy is not None:
        acc = fast_accuracy(dequant_model.vocab, dequant_model.syn0,
                            test_accuracy, restrict=100000)

    print json.dumps(OrderedDict([
        ('q', q),
        ('transform', transform),
        ('pred_bps', float(pred_bps)),
        ('avg_zeros', float(avg_zeros)),
        ('avg_err', float(avg_err)),
        ('accuracy', acc),
    ]))


@baker.command
def quant_entities(input_file, output_template=None, target_err=0.1):
    entity_model = EntityModel.load(input_file, mmap='r')

    q, pred_bits, zeros, avg_err, quant_vecs, dequant_model = quantize_entities(entity_model, target_err)
    pred_bps = float(pred_bits) / quant_vecs.size
    avg_zeros = float(zeros) / quant_vecs.size

    if output_template is not None:
        output_filename = '%s.e%.3f' % (output_template, target_err)
        with open(output_filename + '.txt', 'w') as fout:
            index2entity = [None] * len(entity_model.entities)
            for entity, idx in entity_model.entities.iteritems():
                index2entity[idx] = entity
            save_vectors(fout, index2entity, quant_vecs, q)

        dequant_model.save(output_filename + '.model')

    print json.dumps(OrderedDict([
        ('q', q),
        ('pred_bps', float(pred_bps)),
        ('avg_zeros', float(avg_zeros)),
        ('avg_err', float(avg_err)),
    ]))


def load_quant_data(json_filename):
    import pandas as pd

    with open(json_filename) as fin:
        data = []
        decoder = json.JSONDecoder(object_pairs_hook=OrderedDict)
        for line in fin:
            row = decoder.decode(line)
            accuracy = row['accuracy'][-1]
            assert accuracy['section'] == 'total' # XXX
            acc_percentage = float(accuracy['correct']) / (accuracy['correct'] + accuracy['incorrect'])
            row['accuracy'] = acc_percentage

            data.append(row)

        return pd.DataFrame(data)



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    baker.run()
