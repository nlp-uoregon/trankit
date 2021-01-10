'''
Adapted from https://github.com/stanfordnlp/stanza/blob/master/stanza/models/ner/utils.py
Date: 2021/01/06
'''
from collections import Counter


def decode_from_bioes(tags):
    res = []
    ent_idxs = []
    cur_type = None

    def flush():
        if len(ent_idxs) > 0:
            res.append({
                'start': ent_idxs[0],
                'end': ent_idxs[-1],
                'type': cur_type})

    for idx, tag in enumerate(tags):
        if tag is None:
            tag = 'O'
        if tag == 'O':
            flush()
            ent_idxs = []
        elif tag.startswith('B-'):
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
        elif tag.startswith('I-'):
            ent_idxs.append(idx)
            cur_type = tag[2:]
        elif tag.startswith('E-'):
            ent_idxs.append(idx)
            cur_type = tag[2:]
            flush()
            ent_idxs = []
        elif tag.startswith('S-'):
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
            flush()
            ent_idxs = []

    flush()
    return res


def score_by_entity(pred_bioes_tag_sequences, gold_bioes_tag_sequences, logger):
    assert (len(gold_bioes_tag_sequences) == len(pred_bioes_tag_sequences))

    def decode_all(tag_sequences):
        ents = []
        for sent_id, tags in enumerate(tag_sequences):
            for ent in decode_from_bioes(tags):
                ent['sent_id'] = sent_id
                ents += [ent]
        return ents

    gold_ents = decode_all(gold_bioes_tag_sequences)
    pred_ents = decode_all(pred_bioes_tag_sequences)

    correct_by_type = Counter()
    guessed_by_type = Counter()
    gold_by_type = Counter()

    for p in pred_ents:
        guessed_by_type[p['type']] += 1
        if p in gold_ents:
            correct_by_type[p['type']] += 1
    for g in gold_ents:
        gold_by_type[g['type']] += 1

    prec_micro = 0.0
    if sum(guessed_by_type.values()) > 0:
        prec_micro = sum(correct_by_type.values()) * 1.0 / sum(guessed_by_type.values())
    rec_micro = 0.0
    if sum(gold_by_type.values()) > 0:
        rec_micro = sum(correct_by_type.values()) * 1.0 / sum(gold_by_type.values())
    f_micro = 0.0
    if prec_micro + rec_micro > 0:
        f_micro = 2.0 * prec_micro * rec_micro / (prec_micro + rec_micro)

    return {
        'p': prec_micro * 100,
        'r': rec_micro * 100,
        'f1': f_micro * 100
    }
