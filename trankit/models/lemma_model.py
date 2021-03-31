'''
Adapted from https://github.com/stanfordnlp/stanza/blob/master/stanza/models/lemmatizer.py
Date: 2021/01/06
'''
import sys
from trankit.layers.seq2seq import Seq2SeqModel
from ..iterators.lemmatizer_iterators import LemmaDataLoader
from ..utils.base_utils import *


def set_lemma(doc, preds, obmit_tag=None, training_mode=False):
    if not training_mode:
        wid = 0
        for sentence in doc:
            for token in sentence[TOKENS]:
                if type(token[ID]) == int or len(token[ID]) == 1:
                    token[LEMMA] = preds[wid]
                    wid += 1

                    if obmit_tag:
                        if UPOS in token:
                            del token[UPOS]
                        if XPOS in token:
                            del token[XPOS]
                        if FEATS in token:
                            del token[FEATS]
                        if HEAD in token:
                            del token[HEAD]
                        if DEPREL in token:
                            del token[DEPREL]
                else:
                    for word in token[EXPANDED]:
                        word[LEMMA] = preds[wid]
                        wid += 1

                        if obmit_tag:
                            if UPOS in word:
                                del word[UPOS]
                            if XPOS in word:
                                del word[XPOS]
                            if FEATS in token:
                                del token[FEATS]
                            if HEAD in word:
                                del word[HEAD]
                            if DEPREL in word:
                                del word[DEPREL]
        return doc
    else:
        wid = 0
        for sentence in doc:
            for token in sentence:
                if type(token[ID]) == tuple and len(token[ID]) == 2:
                    continue
                else:
                    token[LEMMA] = preds[wid]
                    wid += 1
        return doc


class Trainer:
    """ A trainer for training models. """

    def __init__(self, args=None, vocab=None, emb_matrix=None, model_file=None, use_cuda=False, training_mode=False):
        self.use_cuda = use_cuda
        self.training_mode = training_mode
        if model_file is not None:
            # load everything from file
            self.load(model_file, use_cuda)
        else:
            # build model from scratch
            self.args = args
            self.model = None if args['dict_only'] else Seq2SeqModel(args, emb_matrix=emb_matrix, use_cuda=use_cuda,
                                                                     training_mode=training_mode)
            self.vocab = vocab
            # dict-based components
            self.word_dict = dict()
            self.composite_dict = dict()
        if not self.args['dict_only']:
            if self.args.get('edit', False):
                self.crit = MixLoss(self.vocab['char'].size, self.args['alpha'])
            else:
                self.crit = SequenceLoss(self.vocab['char'].size)
            self.parameters = [p for p in self.model.parameters() if p.requires_grad]
            if use_cuda:
                self.model.cuda()
                self.crit.cuda()
            else:
                self.model.cpu()
                self.crit.cpu()
            self.optimizer = get_optimizer(self.args['optim'], self.parameters, self.args['lr'])

    def update(self, batch, eval=False):
        inputs, orig_idx = unpack_lemma_batch(batch, self.use_cuda)
        src, src_mask, tgt_in, tgt_out, pos, edits = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs, edit_logits = self.model(src, src_mask, tgt_in, pos)
        if self.args.get('edit', False):
            assert edit_logits is not None
            loss = self.crit(log_probs.view(-1, self.vocab['char'].size), tgt_out.view(-1), \
                             edit_logits, edits)
        else:
            loss = self.crit(log_probs.view(-1, self.vocab['char'].size), tgt_out.view(-1))
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, beam_size=1):
        inputs, orig_idx = unpack_lemma_batch(batch, self.use_cuda)
        src, src_mask, tgt, tgt_mask, pos, edits = inputs

        self.model.eval()
        batch_size = src.size(0)
        preds, edit_logits = self.model.predict(src, src_mask, pos=pos, beam_size=beam_size)
        pred_seqs = [self.vocab['char'].unmap(ids) for ids in preds]  # unmap to tokens
        pred_seqs = prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs]  # join chars to be tokens
        pred_tokens = unsort(pred_tokens, orig_idx)
        if self.args.get('edit', False):
            assert edit_logits is not None
            edits = np.argmax(edit_logits.data.cpu().numpy(), axis=1).reshape([batch_size]).tolist()
            edits = unsort(edits, orig_idx)
        else:
            edits = None
        return pred_tokens, edits

    def postprocess(self, words, preds, edits=None):
        """ Postprocess, mainly for handing edits. """
        assert len(words) == len(preds), "Lemma predictions must have same length as words."
        edited = []
        if self.args.get('edit', False):
            assert edits is not None and len(words) == len(edits)
            for w, p, e in zip(words, preds, edits):
                lem = edit_word(w, p, e)
                edited += [lem]
        else:
            edited = preds  # do not edit
        # final sanity check
        assert len(edited) == len(words)
        final = []
        for lem, w in zip(edited, words):
            if len(lem) == 0 or UNK in lem:
                final += [w]  # invalid prediction, fall back to word
            else:
                final += [lem]
        return final

    def update_lr(self, new_lr):
        change_lr(self.optimizer, new_lr)

    def train_dict(self, triples):
        """ Train a dict lemmatizer given training (word, pos, lemma) triples. """
        # accumulate counter
        ctr = Counter()
        ctr.update([(p[0], p[1], p[2]) for p in triples])
        # find the most frequent mappings
        for p, _ in ctr.most_common():
            w, pos, l = p
            if (w, pos) not in self.composite_dict:
                self.composite_dict[(w, pos)] = l
            if w not in self.word_dict:
                self.word_dict[w] = l

    def predict_dict(self, pairs):
        """ Predict a list of lemmas using the dict model given (word, pos) pairs. """
        lemmas = []
        for p in pairs:
            w, pos = p
            if (w, pos) in self.composite_dict:
                lemmas += [self.composite_dict[(w, pos)]]
            elif w in self.word_dict:
                lemmas += [self.word_dict[w]]
            else:
                lemmas += [w]
        return lemmas

    def skip_seq2seq(self, pairs):
        """ Determine if we can skip the seq2seq module when ensembling with the frequency lexicon. """

        skip = []
        for p in pairs:
            w, pos = p
            if (w, pos) in self.composite_dict:
                skip.append(True)
            elif w in self.word_dict:
                skip.append(True)
            else:
                skip.append(False)
        return skip

    def ensemble(self, pairs, other_preds):
        """ Ensemble the dict with statistical model predictions. """
        lemmas = []
        assert len(pairs) == len(other_preds)
        for p, pred in zip(pairs, other_preds):
            w, pos = p
            if (w, pos) in self.composite_dict:
                lemma = self.composite_dict[(w, pos)]
            elif w in self.word_dict:
                lemma = self.word_dict[w]
            else:
                lemma = pred
            if lemma is None:
                lemma = w
            lemmas.append(lemma)
        return lemmas

    def save(self, filename):
        params = {
            'model': self.model.state_dict() if self.model is not None else None,
            'dicts': (self.word_dict, self.composite_dict),
            'vocab': self.vocab.state_dict(),
            'config': self.args
        }
        try:
            torch.save(params, filename)
        except BaseException:
            raise

    def load(self, filename, use_cuda=False):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            raise
        self.args = checkpoint['config']
        self.word_dict, self.composite_dict = checkpoint['dicts']
        if not self.args['dict_only']:
            self.model = Seq2SeqModel(self.args, use_cuda=use_cuda)
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model = None
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])


def get_identity_lemma_model():
    args = {
        'data_dir': '',
        'train_file': '',
        'eval_file': '',
        'output_file': '',
        'gold_file': '',
        'mode': 'predict',
        'lang': '',
        'batch_size': 5000,
        'seed': 1234
    }

    return args


def get_args():
    args = {
        'train_file': '',
        'eval_file': '',
        'output_file': '',
        'gold_file': '',
        'mode': 'predict',
        'lang': '',
        'ensemble_dict': True,
        'dict_only': False,
        'hidden_dim': 200,
        'emb_dim': 50,
        'num_layers': 1,
        'emb_dropout': 0.5,
        'dropout': 0.5,
        'max_dec_len': 50,
        'beam_size': 1,
        'attn_type': 'soft',
        'pos_dim': 50,
        'pos_dropout': 0.5,
        'edit': True,
        'num_edit': len(EDIT_TO_ID),
        'alpha': 1.0,
        'pos': True,
        'sample_train': 1.0,
        'optim': 'adam',
        'lr': 1e-3,
        'lr_decay': 0.9,
        'decay_epoch': 30,
        'num_epoch': 50,
        'batch_size': 5000,
        'max_grad_norm': 5.0,
        'log_step': 20,
        'seed': 1234
    }
    return args


def get_lemma_model(cache_dir, language, use_gpu):
    args = get_args()
    # load model
    model_file = os.path.join(cache_dir, '{}/{}_lemmatizer.pt'.format(language, language))
    args['data_dir'] = os.path.join(cache_dir, language)
    args['model_dir'] = os.path.join(cache_dir, language)
    trainer = Trainer(model_file=model_file, use_cuda=use_gpu)
    if use_gpu:
        trainer.model.half()
    loaded_args, vocab = trainer.args, trainer.vocab

    for k in args:
        if k.endswith('_dir') or k.endswith('_file'):
            loaded_args[k] = args[k]

    return trainer, args, loaded_args, vocab


class LemmaWrapper:
    # adapted from stanza
    def __init__(self, config, treebank_name, use_gpu, evaluate=True):
        self.config = config
        self.treebank_name = treebank_name
        if evaluate:
            if self.treebank_name in ['UD_Old_French-SRCMF', 'UD_Vietnamese-VTB', 'UD_Vietnamese-VLSP']:
                self.args = get_identity_lemma_model()
            else:
                self.model, self.args, self.loaded_args, self.vocab = get_lemma_model(os.path.join(self.config._cache_dir, self.config.embedding_name),
                                                                                      treebank2lang[treebank_name],
                                                                                      use_gpu)
            print('Loading lemmatizer for {}'.format(treebank2lang[treebank_name]))
        else:
            self.get_lemma_trainer(treebank2lang[treebank_name], use_gpu)

    def get_lemma_trainer(self, language, use_gpu):
        args = get_args()
        args['mode'] = 'train'
        args['batch_size'] = self.config.batch_size
        args['lang'] = language
        args['cuda'] = use_gpu
        args['model_dir'] = self.config._save_dir
        args['num_epoch'] = self.config.max_epoch

        self.train_file = self.config.train_conllu_fpath
        # pred and gold path
        self.system_pred_file = os.path.join(self.config._save_dir, 'preds', 'lemmatizer.dev.conllu')
        self.gold_file = self.config.dev_conllu_fpath

        in_dev_file = os.path.join(self.config._save_dir, 'preds', 'tagger.dev.conllu')
        if not os.path.exists(in_dev_file):
            in_dev_file = self.config.dev_conllu_fpath

        train_doc = CoNLL.conll2dict(input_file=self.train_file)
        self.train_batch = LemmaDataLoader(train_doc, args['batch_size'], args, evaluation=False, training_mode=True)
        vocab = self.train_batch.vocab
        args['vocab_size'] = vocab['char'].size
        args['pos_vocab_size'] = vocab['pos'].size

        dev_doc = CoNLL.conll2dict(input_file=in_dev_file)
        self.dev_batch = LemmaDataLoader(dev_doc, args['batch_size'], args, vocab=vocab, evaluation=True,
                                         training_mode=True)

        self.model_file = os.path.join(self.config._save_dir, '{}_lemmatizer.pt'.format(language))

        # skip training if the language does not have training or dev data
        if len(self.train_batch) == 0 or len(self.dev_batch) == 0:
            print("This language does not require multi-word token expansion")
            self.config.logger.info("This language does not require multi-word token expansion")
            sys.exit(0)

        # initialize a trainer
        self.trainer = Trainer(args=args, vocab=vocab, use_cuda=args['cuda'], training_mode=True)

        self.args = args
        self.loaded_args, self.vocab = self.trainer.args, self.trainer.vocab
        print('Initialized lemmatizer trainer')
        self.config.logger.info('Initialized lemmatizer trainer!')

    def train(self):
        if self.treebank_name not in ['UD_Old_French-SRCMF', 'UD_Vietnamese-VTB', 'UD_Vietnamese-VLSP']:
            print("Training dictionary-based lemmatizer")
            self.config.logger.info("Training dictionary-based lemmatizer")
            self.trainer.train_dict(
                [[token[TEXT], token[UPOS], token[LEMMA]] for sentence in self.train_batch.doc for token in sentence if
                 not (
                         type(token[ID]) == tuple and len(token[ID]) == 2)])
            dev_preds = self.trainer.predict_dict(
                [[token[TEXT], token[UPOS]] for sentence in self.dev_batch.doc for token in sentence if
                 not (type(token[ID]) == tuple and len(token[ID]) == 2)])
            self.dev_batch.doc = set_lemma(self.dev_batch.doc, dev_preds, training_mode=True)
            CoNLL.dict2conll(self.dev_batch.doc, self.system_pred_file)
            dev_f = get_ud_score(self.system_pred_file, self.gold_file)['Lemmas'].f1
            print("Dev F1 = {:.2f}".format(dev_f * 100))
            self.config.logger.info("Dev F1 = {:.2f}".format(dev_f * 100))

            # train a seq2seq model
            print("Training seq2seq-based lemmatizer")
            self.config.logger.info("Training seq2seq-based lemmatizer")
            global_step = 0
            max_steps = len(self.train_batch) * self.args['num_epoch']
            dev_score_history = []
            best_dev_preds = []
            current_lr = self.args['lr']
            global_start_time = time.time()
            format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

            # start training
            best_dev_score = {}
            for epoch in range(1, self.args['num_epoch'] + 1):
                train_loss = 0
                progress = tqdm(total=len(self.train_batch), ncols=75,
                                desc='Train {}'.format(epoch))
                for i, batch in enumerate(self.train_batch):
                    progress.update(1)
                    start_time = time.time()
                    global_step += 1
                    loss = self.trainer.update(batch, eval=False)  # update step
                    train_loss += loss
                    if global_step % self.args['log_step'] == 0:
                        duration = time.time() - start_time
                        self.config.logger.info(
                            format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step, \
                                              max_steps, epoch, self.args['num_epoch'], loss, duration, current_lr))
                progress.close()
                # eval on dev
                dev_preds = []
                dev_edits = []
                for i, batch in enumerate(self.dev_batch):
                    preds, edits = self.trainer.predict(batch, self.args['beam_size'])
                    dev_preds += preds
                    if edits is not None:
                        dev_edits += edits
                dev_preds = self.trainer.postprocess(
                    [token[TEXT] for sentence in self.dev_batch.doc for token in sentence if
                     not (type(token[ID]) == tuple and len(token[ID]) == 2)], dev_preds, edits=dev_edits)

                # try ensembling with dict if necessary
                print("Ensembling dict with seq2seq model")
                self.config.logger.info("Ensembling dict with seq2seq model")
                dev_preds = self.trainer.ensemble(
                    [[token[TEXT], token[UPOS]] for sentence in self.dev_batch.doc for token in sentence if
                     not (type(token[ID]) == tuple and len(token[ID]) == 2)], dev_preds)

                self.dev_batch.doc = set_lemma(self.dev_batch.doc, dev_preds, training_mode=True)
                CoNLL.dict2conll(self.dev_batch.doc, self.system_pred_file)
                dev_score = get_ud_score(self.system_pred_file, self.gold_file)

                train_loss = train_loss / self.train_batch.num_examples * self.args['batch_size']  # avg loss per batch

                # save best model
                if epoch == 1 or dev_score['Lemmas'].f1 > max(dev_score_history):
                    self.trainer.save(self.model_file)
                    print("Saving new best model to ... {}".format(self.model_file))
                    best_dev_score = dev_score
                print(get_ud_performance_table(best_dev_score))
                self.config.logger.info(get_ud_performance_table(best_dev_score))

                # lr schedule
                if epoch > self.args['decay_epoch'] and dev_score['Lemmas'].f1 <= dev_score_history[-1] and \
                        self.args['optim'] in ['sgd', 'adagrad']:
                    current_lr *= self.args['lr_decay']
                    self.trainer.update_lr(current_lr)

                dev_score_history += [dev_score['Lemmas'].f1]

            print("Training done")
            self.config.logger.info("Training done")

        else:
            print('This language does not require lemmatization.')
            self.config.logger.info('This language does not require lemmatization.')

    def predict(self, tagged_doc, obmit_tag):
        if self.treebank_name not in ['UD_Old_French-SRCMF', 'UD_Vietnamese-VTB', 'UD_Vietnamese-VLSP']:
            vocab = self.vocab
            # load data
            batch = LemmaDataLoader(tagged_doc, self.args['batch_size'], self.loaded_args, vocab=vocab,
                                    evaluation=True)

            # skip eval if dev data does not exist
            if len(batch) == 0:
                print("No dev data available...")
                sys.exit(0)
            predict_dict_input = []
            for sentence in batch.doc:
                for t in sentence[TOKENS]:
                    if type(t[ID]) == int or len(t[ID]) == 1:
                        predict_dict_input.append([t[TEXT], t[UPOS] if UPOS in t else None])
                    else:
                        for w in t[EXPANDED]:
                            predict_dict_input.append([w[TEXT], w[UPOS] if UPOS in w else None])

            preds = []
            edits = []
            for i, b in enumerate(batch):
                ps, es = self.model.predict(b, self.args['beam_size'])
                preds += ps
                if es is not None:
                    edits += es

            postprocess_input = [w[0] for w in predict_dict_input]
            preds = self.model.postprocess(
                postprocess_input,
                preds,
                edits=edits)

            preds = self.model.ensemble(
                predict_dict_input, preds)

            # write to file and score
            lemmatized_doc = set_lemma(batch.doc, preds, obmit_tag)
        else:
            # use identity mapping for prediction
            preds = [t[TEXT] for sentence in tagged_doc for t in sentence[TOKENS] if
                     type(t[ID]) == int or len(t[ID]) == 1]

            # write to file and score
            lemmatized_doc = set_lemma(tagged_doc, preds, obmit_tag)
        return lemmatized_doc
