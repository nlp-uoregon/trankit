'''
Adapted from https://github.com/stanfordnlp/stanza/blob/master/stanza/models/mwt/data.py
Date: 2021/01/06
'''
from . import *
from ..utils.mwt_lemma_utils import seq2seq_utils


class MWTVocab(BaseVocab):
    def build_vocab(self):
        pairs = self.data
        allchars = "".join([src + tgt for src, tgt in pairs])
        counter = Counter(allchars)

        self._id2unit = seq2seq_utils.VOCAB_PREFIX + list(
            sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}


class MWTDataLoader:
    def __init__(self, doc, batch_size, args, vocab=None, evaluation=False, training_mode=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.doc = doc

        data = self.load_doc(self.doc, evaluation=self.eval, training_mode=training_mode)

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab

        data = self.preprocess(data, self.vocab, args)
        # shuffle for training
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def init_vocab(self, data):
        assert self.eval == False  # for eval vocab must exist
        vocab = MWTVocab(data, self.args['shorthand'])
        return vocab

    def preprocess(self, data, vocab, args):
        processed = []
        for d in data:
            src = list(d[0])
            src = [SOS] + src + [EOS]
            src = vocab.map(src)
            if self.eval:
                tgt = src  # as a placeholder
            else:
                tgt = list(d[1])
            tgt_in = vocab.map([SOS] + tgt)
            tgt_out = vocab.map(tgt + [EOS])
            processed += [[src, tgt_in, tgt_out]]
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 3

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, PAD_ID)
        tgt_in = get_long_tensor(batch[1], batch_size)
        tgt_out = get_long_tensor(batch[2], batch_size)
        assert tgt_in.size(1) == tgt_out.size(1), \
            "Target input and output sequence sizes do not match."
        return (src, src_mask, tgt_in, tgt_out, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_doc(self, doc, evaluation=False, training_mode=False):
        data = get_mwt_expansions(doc, evaluation, training_mode)
        if evaluation: data = [[e] for e in data]
        return data
