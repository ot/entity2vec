#!/usr/bin/env python

import os
import sys
import logging
import time
import re
from collections import defaultdict
from heapq import nlargest
import cPickle as pickle
from itertools import imap, izip, islice
import random
import urllib
import unicodedata
from multiprocessing.pool import ThreadPool

from ext import baker

import numpy as np

from utils import load_word2vec_model
from entity_models import EntityModel, EntityModelLR, EntityModelCentroid


WORDS_RE = re.compile('[a-zA-Z0-9]+')
CAMEL1_RE = re.compile(r'([A-Z0-9])([A-Z0-9])([a-z])')
CAMEL2_RE = re.compile(r'([a-z])([A-Z0-9])')

def extract_words(line, with_title_words=True):
    fields = line.rstrip('\n').split('\t')
    title = fields[0]
    text = ''
    if len(fields) > 1:
        text = fields[1]

    words = text.lower().split()
    if with_title_words:
        # extract words from the title itself
        decoded_title = urllib.unquote(title).lower()
        decoded_title = CAMEL1_RE.sub(r'\1 \2\3', decoded_title)
        decoded_title = CAMEL2_RE.sub(r'\1 \2', decoded_title)
        words += WORDS_RE.findall(decoded_title)

    return title, words


def read_entity_word_seqs(filename, model, min_words=1):
    with open(filename) as fin:
        for line_idx, line in enumerate(fin):
            if line_idx % 10000 == 0:
                logging.info('Read %d lines', line_idx)

            try:
                entity, words = extract_words(line)
            except Exception, e:
                logging.info('Error reading line %r: %s', line, e)
                continue

            word_idxs = []
            for w in words:
                wd = model.vocab.get(w)
                if wd is not None:
                    word_idxs.append(wd.index)

            if len(word_idxs) < min_words:
                continue

            yield entity, word_idxs


@baker.command
def train(mode, model_file, descriptions_file, output_file=None,
          neg_words_mult=2., lbda=50, min_words=1):

    model = load_word2vec_model(model_file, mmap='r')
    if mode == 'centroid':
        entity_model = EntityModelCentroid()
    elif mode == 'lr':
        bins = np.cumsum([model.vocab[word].count
                      for word in model.index2word])
        entity_model = EntityModelLR(bins, neg_words_mult, lbda)
    else:
        raise Exception('unsupported mode %s' % mode)

    entity_model.train(model,
                       read_entity_word_seqs(descriptions_file, model, min_words))

    if output_file is not None:
        entity_model.save(output_file)


@baker.command
def train_eval(mode, model_file, descriptions_file,
               neg_words_mult=2., lbda=50, min_words=50,
               eval_lines=5000, eval_words=10):

    model = load_word2vec_model(model_file, mmap='r')
    if mode == 'centroid':
        entity_model = EntityModelCentroid()
    elif mode == 'lr':
        bins = np.cumsum([model.vocab[word].count
                      for word in model.index2word])
        entity_model = EntityModelLR(bins, neg_words_mult, lbda)
    else:
        raise Exception('unsupported mode %s' % mode)

    rng = random.Random(1729)

    eval_items = []
    def sampled_word_seqs():
        for i, (entity, t, word_idxs) in \
            enumerate(read_entity_word_seqs(descriptions_file, model, min_words)):
            rng.shuffle(word_idxs)
            if i < eval_lines:
                eval_items.append((entity, word_idxs[:eval_words], len(word_idxs)))
            yield entity, t, word_idxs[eval_words:]

    entity_model.train(model, sampled_word_seqs())

    evaluate_retrieval(model, entity_model, eval_items)


def evaluate_retrieval(model, entity_model, eval_items):
    pool = ThreadPool()

    def entity_rank((entity, word_idxs, doc_len)):
        if word_idxs:
            entity_id = entity_model.entities[entity]
            scores = entity_model.score(model,
                                        entity_model.vectors,
                                        word_idxs)
            rank = np.sum(scores >= scores[entity_id])
        else:
            rank = entity_vecs.shape[0]

        return int(np.log2(doc_len)), np.log2(rank)

    ranks = defaultdict(list)
    for size, rank in pool.imap(entity_rank, eval_items):
        ranks[size].append(rank)

    sorted_ranks = sorted(ranks.iteritems())
    logging.info('%s overall score: %.3f by size: %s',
                 type(entity_model).__name__,
                 np.mean(np.hstack(ranks.values())),
                 ' '.join('%d: %.3f' % (k, np.mean(v)) for k, v in sorted_ranks))


def parse_query(norm_entities, context):
    words = []
    to_match = []
    for word in context.lower().split():
        if word.startswith('+'):
            word = word[1:]
            to_match.append(word)
        words.append(word)

    matches = [entity for norm_entity, entity in norm_entities
               if all(word in norm_entity for word in to_match)]
    logging.info('%d matching entities', len(matches))

    return words, matches

def top_entities(model, entity_model, entities, words, k=30):
    word_idxs = [wd.index for wd in (model.vocab.get(word) for word in words)
                 if wd is not None]

    if not word_idxs or not entities:
        return []

    entities_idxs = [entity_model.entities[e] for e in entities]

    tick = time.time()
    scores = entity_model.score(model,
                                entity_model.vectors[entities_idxs],
                                word_idxs)
    top = nlargest(k, izip(scores, entities))
    logging.info('%s scoring time: %.1f ms',
                 type(entity_model).__name__,
                 (time.time() - tick) * 1000)

    return top

@baker.command
def eval(model_file, lr_entity_file, centroid_entity_file):
    import readline
    readline.parse_and_bind('set editing-mode emacs')

    model = load_word2vec_model(model_file, mmap='r')
    lr_entity_model = EntityModel.load(lr_entity_file, mmap='r')
    centroid_entity_model = EntityModel.load(centroid_entity_file, mmap='r')

    norm_entities = [(entity.lower(), entity) for entity in lr_entity_model.entities]

    while True:
        try:
            line = raw_input('> ').strip()
        except EOFError:
            break

        words, entities = parse_query(norm_entities, line)
        lr_top = top_entities(model, lr_entity_model, entities, words)
        centroid_top = top_entities(model, centroid_entity_model, entities, words)

        for (lr_score, lr_ent), (centroid_score, centroid_ent) in zip(lr_top, centroid_top):
            print '%-50s%10.3f | %-50s%10.3f' % (lr_ent, lr_score, centroid_ent, centroid_score)


if __name__ == '__main__':
    np.random.seed(1729)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    baker.run()
