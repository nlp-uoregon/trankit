'''
Adapted from https://github.com/stanfordnlp/stanza/blob/master/stanza/models/mwt_expander.py
Date: 2021/01/06
'''
import sys
from copy import deepcopy
from trankit.layers.seq2seq import Seq2SeqModel
from ..iterators.mwt_iterators import MWTDataLoader
from trankit.utils.mwt_lemma_utils.mwt_utils import get_mwt_expansions, set_mwt_expansions
from ..utils.base_utils import *


class Trainer:
    """ A trainer for training models. """

    def __init__(self, args=None, vocab=None, emb_matrix=None, model_file=None, use_cuda=False, training_mode=False):
        self.use_cuda = use_cuda
        self.training_mode = training_mode
        if model_file is not None:
            # load from file
            self.load(model_file, use_cuda)
        else:
            self.args = args
            self.model = None if args['dict_only'] else Seq2SeqModel(args, emb_matrix=emb_matrix, use_cuda=use_cuda, training_mode=training_mode)
            self.vocab = vocab
            self.expansion_dict = dict()
        if not self.args['dict_only']:
            self.crit = SequenceLoss(self.vocab.size)
            self.parameters = [p for p in self.model.parameters() if p.requires_grad]
            if use_cuda:
                self.model.cuda()
                self.crit.cuda()
            else:
                self.model.cpu()
                self.crit.cpu()
            self.optimizer = get_optimizer(self.args['optim'], self.parameters, self.args['lr'])

    def update(self, batch, eval=False):
        inputs, orig_idx = unpack_mwt_batch(batch, self.use_cuda)
        src, src_mask, tgt_in, tgt_out = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs, _ = self.model(src, src_mask, tgt_in)
        loss = self.crit(log_probs.view(-1, self.vocab.size), tgt_out.view(-1))
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch):
        inputs, orig_idx = unpack_mwt_batch(batch, self.use_cuda)
        src, src_mask, tgt, tgt_mask = inputs

        self.model.eval()
        batch_size = src.size(0)
        preds, _ = self.model.predict(src, src_mask, self.args['beam_size'])
        pred_seqs = [self.vocab.unmap(ids) for ids in preds]  # unmap to tokens
        pred_seqs = prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs]  # join chars to be tokens
        pred_tokens = unsort(pred_tokens, orig_idx)
        return pred_tokens

    def train_dict(self, pairs):
        """ Train a MWT expander given training word-expansion pairs. """
        # accumulate counter
        ctr = Counter()
        ctr.update(
            [(p[0], p[1]) for p in pairs])  # p[0]: source token, p[1]: expanded form that consists of multiple words
        seen = set()
        # find the most frequent mappings
        for p, _ in ctr.most_common():
            w, l = p  # w: src token, l: expanded form that consists of multiple words
            if w not in seen and w != l:
                self.expansion_dict[w] = l
            seen.add(w)

    def predict_dict(self, words):
        """ Predict a list of expansions given words. """
        expansions = []
        for w in words:
            if w in self.expansion_dict:
                expansions += [self.expansion_dict[w]]
            elif w.lower() in self.expansion_dict:
                expansions += [self.expansion_dict[w.lower()]]
            else:
                expansions += [w]
        return expansions

    def ensemble(self, cands, other_preds):
        """ Ensemble the dict with statistical model predictions. """
        expansions = []
        assert len(cands) == len(other_preds)
        for c, pred in zip(cands, other_preds):
            if c in self.expansion_dict:
                expansions += [self.expansion_dict[c]]
            elif c.lower() in self.expansion_dict:
                expansions += [self.expansion_dict[c.lower()]]
            else:
                expansions += [pred]
        return expansions

    def save(self, filename):
        params = {
            'model': self.model.state_dict() if self.model is not None else None,
            'dict': self.expansion_dict,
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
        self.expansion_dict = checkpoint['dict']
        if not self.args['dict_only']:
            self.model = Seq2SeqModel(self.args, use_cuda=use_cuda)
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model = None
        self.vocab = Vocab.load_state_dict(checkpoint['vocab'])


def get_args():
    args = {
        'train_file': '',
        'eval_file': '',
        'output_file': '',
        'gold_file': '',
        'mode': 'predict',
        'lang': '',
        'ensemble_dict': True,
        'ensemble_early_stop': False,
        'dict_only': False,
        'hidden_dim': 100,
        'emb_dim': 50,
        'num_layers': 1,
        'emb_dropout': 0.5,
        'dropout': 0.5,
        'max_dec_len': 50,
        'beam_size': 1,
        'attn_type': 'soft',
        'sample_train': 1.0,
        'optim': 'adam',
        'lr': 1e-3,
        'lr_decay': 0.9,
        'decay_epoch': 30,
        'num_epoch': 50,
        'batch_size': 5000,
        'max_grad_norm': 5.0,
        'log_step': 20,
        'save_name': '',
        'seed': 1234
    }
    return args


def get_mwt_model(cache_dir, language, use_gpu):
    args = get_args()
    # ############## load model #############
    # file paths
    model_file = os.path.join(cache_dir, '{}/{}_mwt_expander.pt'.format(language, language))
    args['data_dir'] = os.path.join(cache_dir, language)
    args['save_dir'] = os.path.join(cache_dir, language)
    # load model
    trainer = Trainer(model_file=model_file, use_cuda=use_gpu)
    if use_gpu:
        trainer.model.half()
    loaded_args, vocab = trainer.args, trainer.vocab

    for k in args:
        if k.endswith('_dir') or k.endswith('_file'):
            loaded_args[k] = args[k]

    return trainer, args, loaded_args, vocab


class MWTWrapper:
    # adapted from stanza
    def __init__(self, config, treebank_name, use_gpu, evaluate=True):
        self.config = config
        if evaluate:
            self.model, self.args, self.loaded_args, self.vocab = get_mwt_model(os.path.join(config._cache_dir, config.embedding_name),
                                                                                language=treebank2lang[treebank_name],
                                                                                use_gpu=use_gpu)
            print('Loading multi-word expander for {}'.format(treebank2lang[treebank_name]))
        else:
            self.get_mwt_trainer(treebank2lang[treebank_name], use_gpu)

    def get_mwt_trainer(self, language, use_gpu):
        args = get_args()
        args['mode'] = 'train'
        args['batch_size'] = self.config.batch_size
        args['lang'] = language
        args['shorthand'] = language
        args['cuda'] = use_gpu
        args['model_dir'] = self.config._save_dir
        args['num_epoch'] = self.config.max_epoch

        self.train_file = self.config.train_conllu_fpath
        # pred and gold path
        self.system_pred_file = os.path.join(self.config._save_dir, 'preds', 'mwt.dev.conllu')
        self.gold_file = self.config.dev_conllu_fpath

        self.in_dev_file = os.path.join(self.config._save_dir, 'preds', 'tokenizer.dev.conllu')
        if not os.path.exists(self.in_dev_file):
            self.in_dev_file = self.config.dev_conllu_fpath

        # load data
        self.train_batch = MWTDataLoader(CoNLL.conll2dict(self.train_file), args['batch_size'], args, evaluation=False,
                                         training_mode=True)
        vocab = self.train_batch.vocab
        args['vocab_size'] = vocab.size

        self.dev_batch = MWTDataLoader(CoNLL.conll2dict(self.in_dev_file), args['batch_size'], args, vocab=vocab,
                                       evaluation=True, training_mode=True)

        self.model_file = os.path.join(self.config._save_dir, '{}_mwt_expander.pt'.format(language))

        # skip training if the language does not have training or dev data
        if len(self.train_batch) == 0 or len(self.dev_batch) == 0:
            print("No training data available...")
            sys.exit(0)

        # train a dictionary-based MWT expander
        self.trainer = Trainer(args=args, vocab=vocab, use_cuda=args['cuda'], training_mode=True)
        self.args = args
        self.vocab = self.trainer.vocab

        print('Initiliazed MWT trainer')
        self.config.logger.info('Initiliazed MWT trainer')

    def train(self):
        print("Training dictionary-based MWT expander...")
        self.config.logger.info("Training dictionary-based MWT expander...")
        self.trainer.train_dict(get_mwt_expansions(self.train_batch.doc, evaluation=False, training_mode=True))

        dev_preds = self.trainer.predict_dict(
            get_mwt_expansions(self.dev_batch.doc, evaluation=True, training_mode=True))
        doc = deepcopy(self.dev_batch.doc)

        doc = set_mwt_expansions(doc, dev_preds, training_mode=True)
        CoNLL.dict2conll(doc, self.system_pred_file)
        dev_f = get_ud_score(self.system_pred_file, self.gold_file)['Words'].f1
        print("Dev F1 = {:.2f}".format(dev_f * 100))

        # train a seq2seq model
        print("Training seq2seq-based MWT expander...")
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
                start_time = time.time()
                progress.update(1)
                global_step += 1
                loss = self.trainer.update(batch, eval=False)  # update step
                train_loss += loss
                if global_step % self.args['log_step'] == 0:
                    duration = time.time() - start_time
                    self.config.logger.info(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step, \
                                                              max_steps, epoch, self.args['num_epoch'], loss, duration,
                                                              current_lr))

            # eval on dev
            progress.close()
            dev_preds = []
            for i, batch in enumerate(self.dev_batch):
                preds = self.trainer.predict(batch)
                dev_preds += preds

            doc = deepcopy(self.dev_batch.doc)
            doc = set_mwt_expansions(doc, dev_preds, training_mode=True)
            CoNLL.dict2conll(doc, self.system_pred_file)
            dev_score = get_ud_score(self.system_pred_file, self.gold_file)

            train_loss = train_loss / self.train_batch.num_examples * self.args['batch_size']  # avg loss per batch
            print("epoch {}: train_loss = {:.6f}, dev_score = {:.4f}".format(epoch, train_loss, dev_score['Words'].f1))

            # save best model
            if epoch == 1 or dev_score['Words'].f1 > max(dev_score_history):
                self.trainer.save(self.model_file)
                print("new best model saved.")
                best_dev_preds = dev_preds

            dev_score_history += [dev_score['Words'].f1]

        print("Training done.")

        # try ensembling with dict if necessary
        if self.args.get('ensemble_dict', False):
            print("Ensembling dict with seq2seq model")
            dev_preds = self.trainer.ensemble(
                get_mwt_expansions(self.dev_batch.doc, evaluation=True, training_mode=True), best_dev_preds)
            doc = deepcopy(self.dev_batch.doc)
            doc = set_mwt_expansions(doc, dev_preds, training_mode=True)
            CoNLL.dict2conll(doc, self.system_pred_file)
            dev_score = get_ud_score(self.system_pred_file, self.gold_file)
            print(get_ud_performance_table(dev_score))

    def predict(self, tokenized_doc):
        args = self.args
        loaded_args = self.loaded_args
        vocab = self.vocab
        # load data
        doc = tokenized_doc
        batch = MWTDataLoader(doc, args['batch_size'], loaded_args, vocab=vocab, evaluation=True)

        if len(batch) > 0:
            preds = []
            for i, b in enumerate(batch):
                preds += self.model.predict(b)
            preds = self.model.ensemble(get_mwt_expansions(batch.doc, evaluation=True, training_mode=False), preds)
        else:
            preds = []

        doc = deepcopy(batch.doc)
        expanded_doc = set_mwt_expansions(doc, preds)
        return expanded_doc
