#!/usr/bin/env python

import os
import sys
import json
import logging
from heapq import nlargest
from operator import itemgetter
from itertools import imap, groupby

import numpy as np

from ext import baker
from utils import load_word2vec_model

def fast_accuracy(vocab, syn0, questions_file, restrict=100000, logger=logging):
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool()

    ok_vocab = nlargest(restrict, vocab.iteritems(),
                        key=lambda (_, item): item.count)
    ok_vocab.sort(key=lambda (_, item): item.index)

    ok_proj_vocab = dict((word, proj_idx)
                         for proj_idx, (word, _) in enumerate(ok_vocab))
    ok_syn0 = syn0[[item.index for _, item in ok_vocab]]

    # normalize
    for i in xrange(ok_syn0.shape[0]):
        ok_syn0[i] /= np.sqrt(np.sum(ok_syn0[i]**2))

    questions = []

    with open(questions_file) as fin:
        cur_section = None
        for line_no, line in enumerate(fin):
            if line.startswith(': '):
                cur_section = line.lstrip(': ').strip()
            else:
                if cur_section is None:
                    raise ValueError('Missing section header')

                try:
                    # TODO assumes vocabulary preprocessing uses lowercase, too...
                    wa, wb, wc, wexpected = [word.lower() for word in line.split()]
                except:
                    logger.info("skipping invalid line #%i in %s" % (line_no, questions))
                    continue

                try:
                    a = ok_proj_vocab[wa]
                    b = ok_proj_vocab[wb]
                    c = ok_proj_vocab[wc]
                    expected = ok_proj_vocab[wexpected]
                except KeyError:
                    logger.debug("skipping line #%i with OOV words: %s" % (line_no, line))
                    continue

                questions.append((cur_section, a, b, c, expected))

    def check(question):
        section, a, b, c, expected = question
        ignore = set([a, b, c])

        mean = np.zeros_like(syn0[0])
        for weight, idx in [(-1, a), (1, b), (1, c)]:
            mean += weight * ok_syn0[idx]
        mean /= np.sqrt(np.sum(mean**2))

        dists = np.dot(ok_syn0, mean)
        correct = False
        for proj_idx in np.argsort(dists)[::-1]:
            if proj_idx not in ignore:
                if proj_idx == expected:
                    correct = True
                break

        return section, correct

    def log_section((section, correct, all_qs)):
        logger.info("%s: %.1f%% (%i/%i)",
                    section, 100. * correct / all_qs,
                    correct, all_qs)

    summary = []
    for section, answers in groupby(pool.imap(check, questions),
                                    key=itemgetter(0)):
        answers = list(answers)
        correct = sum(answer for _, answer in answers)
        all_qs = len(answers)
        summary.append((section, correct, all_qs))
        log_section(summary[-1])

    total_correct = sum(t[1] for t in summary)
    total_all_qs = sum(t[2] for t in summary)
    summary.append(('total', total_correct, total_all_qs))
    log_section(summary[-1])

    return summary

@baker.command
def accuracy(input_file, questions_file, restrict=100000):
    model = load_word2vec_model(input_file, mmap='r')
    acc = fast_accuracy(model.vocab, model.syn0, questions_file, restrict)
    print json.dumps(acc)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    baker.run()
