'''
Adapted from https://github.com/stanfordnlp/stanza/blob/master/stanza/models/ner/utils.py
Date: 2021/01/06
'''
import os, json
from collections import Counter

MIN_NUM_FIELD = 2
MAX_NUM_FIELD = 5

DOC_START_TOKEN = '-DOCSTART-'


def bio2_to_bioes(tags):
    """
    Convert the BIO2 tag sequence into a BIOES sequence.
    Args:
        tags: a list of tags in BIO2 format
    Returns:
        new_tags: a list of tags in BIOES format
    """
    new_tags = []
    for i, tag in enumerate(tags):
        tag = tag.replace('B--', 'B-').replace('I--', 'I-')
        if tag == 'O':
            new_tags.append(tag)
        else:
            if len(tag) < 2:
                raise Exception(f"Invalid BIO2 tag found: {tag}")
            else:
                if tag[:2] == 'I-':  # convert to E- if next tag is not I-
                    if i + 1 < len(tags) and tags[i + 1][:2] == 'I-':
                        new_tags.append(tag)
                    else:
                        new_tags.append('E-' + tag[2:])
                elif tag[:2] == 'B-':  # convert to S- if next tag is not I-
                    if i + 1 < len(tags) and tags[i + 1][:2] == 'I-':
                        new_tags.append(tag)
                    else:
                        new_tags.append('S-' + tag[2:])
                else:
                    print(f"Invalid IOB tag found: {tag}, set this to 'O'.")
                    new_tags.append('O')
    return new_tags


def process_cache(cached_lines):
    tokens = []
    ner_tags = []
    for line in cached_lines:
        array = line.split()
        assert len(array) >= MIN_NUM_FIELD and len(array) <= MAX_NUM_FIELD
        tokens.append(array[0])
        ner_tags.append(array[-1])
    return (tokens, ner_tags)


def read_ner_data(config, data_fpath, evaluate):
    conllu_examples = load_conllu(data_fpath)
    output_examples = [{'words': x[0], 'entity-labels': bio2_to_bioes(x[1])} for x in conllu_examples]
    if not evaluate:  # if this is training data
        tagset = set()
        for ex in output_examples:
            tagset.update(set(ex['entity-labels']))
        taglist = list(tagset)
        vocab = {'O': 0}
        taglist = [t for t in taglist if t != 'O']
        taglist.sort()
        for t in taglist:
            vocab[t] = vocab.get(t, len(vocab))

        with open(config.vocab_fpath,
                  'w') as f:
            json.dump(vocab, f)

    return output_examples


def load_conllu(filename, skip_doc_start=True):
    cached_lines = []
    examples = []
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            if skip_doc_start and DOC_START_TOKEN in line:
                continue
            if len(line) > 0:
                array = line.split()
                if len(array) < MIN_NUM_FIELD:
                    continue
                else:
                    cached_lines.append(line)
            elif len(cached_lines) > 0:
                example = process_cache(cached_lines)
                examples.append(example)
                cached_lines = []
        if len(cached_lines) > 0:
            examples.append(process_cache(cached_lines))
    return examples


def decode_from_bioes(tags):
    """
    Decode from a sequence of BIOES tags, assuming default tag is 'O'.
    Args:
        tags: a list of BIOES tags
    Returns:
        A list of dict with start_idx, end_idx, and type values.
    """
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
        elif tag.startswith('B-'):  # start of new ent
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
        elif tag.startswith('I-'):  # continue last ent
            ent_idxs.append(idx)
            cur_type = tag[2:]
        elif tag.startswith('E-'):  # end last ent
            ent_idxs.append(idx)
            cur_type = tag[2:]
            flush()
            ent_idxs = []
        elif tag.startswith('S-'):  # start single word ent
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
            flush()
            ent_idxs = []
    # flush after whole sentence
    flush()
    return res


def score_by_entity(pred_bioes_tag_sequences, gold_bioes_tag_sequences, logger):
    """ Score predicted tags at the entity level.
    Args:
        pred_tags_sequences: a list of list of predicted tags for each word
        gold_tags_sequences: a list of list of gold tags for each word
        verbose: print log with results
    Returns:
        Precision, recall and F1 scores.
    """
    assert (len(gold_bioes_tag_sequences) == len(pred_bioes_tag_sequences)), \
        "Number of predicted tag sequences does not match gold sequences."

    def decode_all(tag_sequences):
        # decode from all sequences, each sequence with a unique id
        ents = []
        for sent_id, tags in enumerate(tag_sequences):
            for ent in decode_from_bioes(tags):
                ent['sent_id'] = sent_id
                ents += [ent]
        return ents

    gold_ents = decode_all(gold_bioes_tag_sequences)
    pred_ents = decode_all(pred_bioes_tag_sequences)

    # scoring
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

    logger.info("Prec.\tRec.\tF1")
    logger.info("{:.2f}\t{:.2f}\t{:.2f}".format(prec_micro * 100, rec_micro * 100, f_micro * 100))

    return {
        'p': prec_micro * 100,
        'r': rec_micro * 100,
        'f1': f_micro * 100
    }
