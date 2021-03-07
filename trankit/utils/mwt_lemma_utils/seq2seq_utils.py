'''
Adapted from https://github.com/stanfordnlp/stanza/blob/master/stanza/models/common/seq2seq_utils.py
Date: 2021/01/06
'''
"""
Utility functions.
"""
import os
from collections import Counter
import random
import json
import unicodedata
import torch
import torch.nn as nn
import numpy as np

PAD = '<PAD>'
PAD_ID = 0
UNK = '<UNK>'
UNK_ID = 1
SOS = '<SOS>'
SOS_ID = 2
EOS = '<EOS>'
EOS_ID = 3

VOCAB_PREFIX = [PAD, UNK, SOS, EOS]

EMB_INIT_RANGE = 1.0
INFINITY_NUMBER = 65504

EDIT_TO_ID = {'none': 0, 'identity': 1, 'lower': 2}


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else UNK_ID for t in tokens]
    return ids


def get_long_tensor(tokens_list, batch_size, pad_id=PAD_ID):
    """ Convert (list of )+ tokens to a padded LongTensor. """
    sizes = []
    x = tokens_list
    while isinstance(x[0], list):
        sizes.append(max(len(y) for y in x))
        x = [z for y in x for z in y]
    tokens = torch.LongTensor(batch_size, *sizes).fill_(pad_id)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def get_float_tensor(features_list, batch_size):
    if features_list is None or features_list[0] is None:
        return None
    seq_len = max(len(x) for x in features_list)
    feature_len = len(features_list[0][0])
    features = torch.FloatTensor(batch_size, seq_len, feature_len).zero_()
    for i, f in enumerate(features_list):
        features[i, :len(f), :] = torch.FloatTensor(f)
    return features


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def get_edit_type(word, lemma):
    """ Calculate edit types. """
    if lemma == word:
        return 'identity'
    elif lemma == word.lower():
        return 'lower'
    return 'none'


def edit_word(word, pred, edit_id):
    """
    Edit a word, given edit and seq2seq predictions.
    """
    if edit_id == 1:
        return word
    elif edit_id == 2:
        return word.lower()
    elif edit_id == 0:
        return pred
    else:
        raise Exception("Unrecognized edit ID: {}".format(edit_id))


def unpack_mwt_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:4]]
    else:
        inputs = [b if b is not None else None for b in batch[:4]]
    orig_idx = batch[4]
    return inputs, orig_idx


def unpack_lemma_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:6]]
    else:
        inputs = [b if b is not None else None for b in batch[:6]]
    orig_idx = batch[6]
    return inputs, orig_idx


def SequenceLoss(vocab_size):
    weight = torch.ones(vocab_size)
    weight[PAD_ID] = 0
    crit = nn.NLLLoss(weight)
    return crit


def weighted_cross_entropy_loss(labels, log_dampened=False):
    """
    Either return a loss function which reweights all examples so the
    classes have the same effective weight, or dampened reweighting
    using log() so that the biggest class has some priority
    """
    if isinstance(labels, list):
        all_labels = np.array(labels)
    _, weights = np.unique(labels, return_counts=True)
    weights = weights / float(np.sum(weights))
    weights = np.sum(weights) / weights
    if log_dampened:
        weights = 1 + np.log(weights)
    loss = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).type('torch.FloatTensor')
    )
    return loss


class MixLoss(nn.Module):
    """
    A mixture of SequenceLoss and CrossEntropyLoss.
    Loss = SequenceLoss + alpha * CELoss
    """

    def __init__(self, vocab_size, alpha):
        super().__init__()
        self.seq_loss = SequenceLoss(vocab_size)
        self.ce_loss = nn.CrossEntropyLoss()
        assert alpha >= 0
        self.alpha = alpha

    def forward(self, seq_inputs, seq_targets, class_inputs, class_targets):
        sl = self.seq_loss(seq_inputs, seq_targets)
        cel = self.ce_loss(class_inputs, class_targets)
        loss = sl + self.alpha * cel
        return loss


class MaxEntropySequenceLoss(nn.Module):
    """
    A max entropy loss that encourage the model to have large entropy,
    therefore giving more diverse outputs.

    Loss = NLLLoss + alpha * EntropyLoss
    """

    def __init__(self, vocab_size, alpha):
        super().__init__()
        weight = torch.ones(vocab_size)
        weight[PAD_ID] = 0
        self.nll = nn.NLLLoss(weight)
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        inputs: [N, C]
        targets: [N]
        """
        assert inputs.size(0) == targets.size(0)
        nll_loss = self.nll(inputs, targets)
        # entropy loss
        mask = targets.eq(PAD_ID).unsqueeze(1).expand_as(inputs)
        masked_inputs = inputs.clone().masked_fill_(mask, 0.0)
        p = torch.exp(masked_inputs)
        ent_loss = p.mul(masked_inputs).sum() / inputs.size(0)  # average over minibatch
        loss = nll_loss + self.alpha * ent_loss
        return loss


# filenames
def get_wordvec_file(w2v_name, wordvec_dir, shorthand, wordvec_type=None):
    """ Lookup the name of the word vectors file, given a directory and the language shorthand.
    """
    lcode, tcode = shorthand.split('_', 1)
    # locate language folder
    word2vec_dir = os.path.join('../..', wordvec_dir, 'word2vec', w2v_name)
    fasttext_dir = os.path.join('../..', wordvec_dir, 'fasttext', w2v_name)

    lang_dir = None
    if wordvec_type is not None:
        lang_dir = os.path.join(wordvec_dir, wordvec_type, w2v_name)
        if not os.path.exists(lang_dir):
            raise FileNotFoundError(
                "Word vector type {} was specified, but directory {} does not exist".format(wordvec_type, lang_dir))
    elif os.path.exists(word2vec_dir):  # first try word2vec
        lang_dir = word2vec_dir
    elif os.path.exists(fasttext_dir):  # otherwise try fasttext
        lang_dir = fasttext_dir
    else:
        raise FileNotFoundError(
            "Cannot locate word vector directory for language: {}  Looked in {} and {}".format(w2v_name, word2vec_dir,
                                                                                               fasttext_dir))
    # look for wordvec filename in {lang_dir}
    filename = os.path.join(lang_dir, '{}.vectors'.format(lcode))
    if os.path.exists(filename + ".xz"):
        filename = filename + ".xz"
    elif os.path.exists(filename + ".txt"):
        filename = filename + ".txt"
    return filename


# training schedule
def get_adaptive_eval_interval(cur_dev_size, thres_dev_size, base_interval):
    """ Adjust the evaluation interval adaptively.
    If cur_dev_size <= thres_dev_size, return base_interval;
    else, linearly increase the interval (round to integer times of base interval).
    """
    if cur_dev_size <= thres_dev_size:
        return base_interval
    else:
        alpha = round(cur_dev_size / thres_dev_size)
        return base_interval * alpha


# ud utils

def harmonic_mean(a, weights=None):
    if any(x == 0 for x in a):
        return 0
    else:
        assert weights is None or len(weights) == len(
            a), 'Weights has length {} which is different from that of the array ({}).'.format(len(weights), len(a))
        if weights is None:
            return len(a) / sum([1 / x for x in a])
        else:
            return sum(weights) / sum(w / x for x, w in zip(a, weights))


# torch utils
def get_optimizer(name, parameters, lr, betas=(0.9, 0.999), eps=1e-8, momentum=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters)  # use default lr
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var


def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad


# other utils
def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)


def save_config(config, path, verbose=True):
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config


def load_config(path, verbose=True):
    with open(path) as f:
        config = json.load(f)
    if verbose:
        print("Config loaded from file {}".format(path))
    return config


def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def unmap_with_copy(indices, src_tokens, vocab):
    """
    Unmap a list of list of indices, by optionally copying from src_tokens.
    """
    result = []
    for ind, tokens in zip(indices, src_tokens):
        words = []
        for idx in ind:
            if idx >= 0:
                words.append(vocab.id2word[idx])
            else:
                idx = -idx - 1  # flip and minus 1
                words.append(tokens[idx])
        result += [words]
    return result


def prune_decoded_seqs(seqs):
    """
    Prune decoded sequences after EOS token.
    """
    out = []
    for s in seqs:
        if EOS in s:
            idx = s.index(EOS)
            out += [s[:idx]]
        else:
            out += [s]
    return out


def prune_hyp(hyp):
    """
    Prune a decoded hypothesis
    """
    if EOS_ID in hyp:
        idx = hyp.index(EOS_ID)
        return hyp[:idx]
    else:
        return hyp


def prune(data_list, lens):
    assert len(data_list) == len(lens)
    nl = []
    for d, l in zip(data_list, lens):
        nl.append(d[:l])
    return nl


def sort(packed, ref, reverse=True):
    """
    Sort a series of packed list, according to a ref list.
    Also return the original index before the sort.
    """
    assert (isinstance(packed, tuple) or isinstance(packed, list)) and isinstance(ref, list)
    packed = [ref] + [range(len(ref))] + list(packed)
    sorted_packed = [list(t) for t in zip(*sorted(zip(*packed), reverse=reverse))]
    return tuple(sorted_packed[1:])


def unsort(sorted_list, oidx):
    """
    Unsort a sorted list, based on the original idx.
    """
    assert len(sorted_list) == len(oidx), "Number of list elements must match with original indices."
    _, unsorted = [list(t) for t in zip(*sorted(zip(oidx, sorted_list)))]
    return unsorted


def tensor_unsort(sorted_tensor, oidx):
    """
    Unsort a sorted tensor on its 0-th dimension, based on the original idx.
    """
    assert sorted_tensor.size(0) == len(oidx), "Number of list elements must match with original indices."
    backidx = [x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]
    return sorted_tensor[backidx]


def set_random_seed(seed, cuda):
    """
    Set a random seed on all of the things which might need it.
    torch, np, python random, and torch.cuda
    """
    if seed is None:
        seed = random.randint(0, 1000000000)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    return seed
