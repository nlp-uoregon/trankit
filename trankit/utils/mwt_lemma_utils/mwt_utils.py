import re
from trankit.utils.conll import *

multi_word_token_misc = re.compile(r".*MWT=Yes.*")


def get_mwt_expansions(sentences, evaluation, training_mode):
    if not training_mode:
        expansions = []
        for sentence in sentences:
            for token in sentence[TOKENS]:
                m = type(token[ID]) == tuple
                n = multi_word_token_misc.match(token[MISC]) if MISC in token and token[MISC] is not None else None
                if m or n:
                    src = token[TEXT]
                    expansions.append(src)
        return expansions
    elif evaluation:
        expansions = []
        for sentence in sentences:
            for token in sentence:
                m = (len(token[ID]) > 1)
                n = multi_word_token_misc.match(token[MISC]) if MISC in token and token[MISC] is not None else None
                if m or n:
                    src = token[TEXT]
                    expansions.append(src)
        return expansions
    else:
        expansions = []
        for sentence in sentences:
            mwt_start = 0
            mwt_end = -1
            src = None
            expanded = []
            for token in sentence:
                m = (len(token[ID]) > 1)
                n = multi_word_token_misc.match(token[MISC]) if MISC in token and token[MISC] is not None else None
                if m or n:
                    mwt_start = token[ID][0]
                    mwt_end = token[ID][1]
                    src = token[TEXT]
                    expanded = []
                elif mwt_start <= token[ID][0] < mwt_end:
                    expanded += [token[TEXT]]
                elif token[ID][0] == mwt_end:
                    expanded += [token[TEXT]]
                    dst = ' '.join(expanded)
                    expansions += [(src, dst)]
                    mwt_start = 0
                    mwt_end = -1
                    src = None

        if evaluation: expansions = [e[0] for e in expansions]
        return expansions


def set_mwt_expansions(sentences, expansions, training_mode=False):
    """ Extend the multi-word tokens annotated by tokenizer. A list of list of expansions
    will be expected for each multi-word token.
    """
    if not training_mode:
        idx_e = 0
        new_sentences = []
        for sentence in sentences:
            idx_w = 0
            for token in sentence[TOKENS]:
                idx_w += 1
                m = type(token[ID]) == tuple
                n = multi_word_token_misc.match(token[MISC]) if MISC in token and token[MISC] is not None else None
                if m or n:
                    expanded = [x for x in expansions[idx_e].split(' ') if len(x) > 0]
                    idx_e += 1
                    idx_w_end = idx_w + len(expanded) - 1
                    token[MISC] = None if token[MISC] == 'MWT=Yes' else '|'.join(
                        [x for x in token[MISC].split('|') if x != 'MWT=Yes'])
                    token[ID] = (idx_w, idx_w_end)
                    token['words'] = []
                    for i, e_word in enumerate(expanded):
                        token['words'].append({ID: idx_w + i, TEXT: e_word})
                    idx_w = idx_w_end
            new_sentence = {}
            for k, v in sentence.items():
                if k != TOKENS:
                    new_sentence[k] = v
            new_sentence[TOKENS] = []
            tid = 1
            for token in sentence[TOKENS]:
                if type(token[ID]) == int or len(token[ID]) == 1:
                    token[ID] = tid
                    new_sentence[TOKENS].append(token)
                    tid += 1
                else:
                    offset = tid - token[ID][0]
                    new_sentence[TOKENS].append({
                        ID: (offset + token[ID][0], offset + token[ID][1]),
                        TEXT: token[TEXT],
                        EXPANDED: []
                    })
                    if SSPAN in token:
                        new_sentence[TOKENS][-1][SSPAN] = token[SSPAN]
                    if DSPAN in token:
                        new_sentence[TOKENS][-1][DSPAN] = token[DSPAN]

                    for word in token['words']:
                        new_sentence[TOKENS][-1][EXPANDED].append({ID: offset + word[ID], TEXT: word[TEXT]})
                    tid = offset + token[ID][1] + 1

            new_sentences.append(new_sentence)
        return new_sentences
    else:
        idx_e = 0
        new_sentences = []
        for sentence in sentences:
            new_sent = []
            idx_w = 0
            for token in sentence:
                idx_w += 1
                m = (len(token[ID]) > 1)
                n = multi_word_token_misc.match(token.get(MISC, None)) if token.get(MISC, None) is not None else None
                if not m and not n:
                    new_sent.append({ID: idx_w, TEXT: token[TEXT]})
                else:
                    expanded = [x for x in expansions[idx_e].split(' ') if len(x) > 0]
                    idx_e += 1
                    idx_w_end = idx_w + len(expanded) - 1
                    token[MISC] = None if token.get(MISC, '_') == 'MWT=Yes' else '|'.join(
                        [x for x in token.get(MISC, '_').split('|') if x != 'MWT=Yes'])
                    token[ID] = (idx_w, idx_w_end)
                    new_sent.append({ID: token[ID], TEXT: token[TEXT], MISC: token[MISC]})

                    for i, e_word in enumerate(expanded):
                        new_sent.append({ID: idx_w + i, TEXT: e_word})
                    idx_w = idx_w_end
            new_sentences.append(new_sent)
        assert idx_e == len(expansions), "{} {}".format(idx_e, len(expansions))
        return new_sentences
