from .config import config as master_config
from .models.base_models import Multilingual_Embedding
from .models.classifiers import TokenizerClassifier, PosDepClassifier, NERClassifier
from .models.mwt_model import MWTWrapper
from .models.lemma_model import LemmaWrapper
from .iterators.tokenizer_iterators import TokenizeDataset
from .iterators.tagger_iterators import TaggerDataset
from .iterators.ner_iterators import NERDataset
from .utils.tokenizer_utils import *
from .utils.scorers.ner_scorer import score_by_entity
from collections import defaultdict
from .utils.conll import *
from .utils.tbinfo import tbname2training_id, lang2treebank
from .utils.chuliu_edmonds import *
from .adapter_transformers import XLMRobertaTokenizer
from tqdm import tqdm
from .adapter_transformers import AdamW, get_linear_schedule_with_warmup
import logging


class TPipeline:
    def __init__(self, training_config):
        super(TPipeline, self).__init__()
        # set up training config
        self._set_up_config(training_config)

        # prepare data and vocabs
        self._prepare_data_and_vocabs()

        # initialize model
        if self._task == 'tokenize':
            self._embedding_layers = Multilingual_Embedding(self._config, model_name='tokenizer')
            self._embedding_layers.to(self._config.device)

            # tokenizers
            self._tokenizer = TokenizerClassifier(self._config, treebank_name=lang2treebank[self._lang])
            self._tokenizer.to(self._config.device)

            self.model_parameters = [(n, p) for n, p in self._embedding_layers.named_parameters()] + \
                                    [(n, p) for n, p in self._tokenizer.named_parameters()]

        elif self._task == 'posdep':
            self._embedding_layers = Multilingual_Embedding(self._config, model_name='tagger')
            self._embedding_layers.to(self._config.device)

            # taggers
            self._tagger = PosDepClassifier(self._config, treebank_name=lang2treebank[self._lang])
            self._tagger.to(self._config.device)

            self.model_parameters = [(n, p) for n, p in self._embedding_layers.named_parameters()] + \
                                    [(n, p) for n, p in self._tagger.named_parameters()]
        elif self._task == 'mwt':
            # mwt
            self._mwt_model = MWTWrapper(self._config, treebank_name=self._config.treebank_name,
                                         use_gpu=self._use_gpu, evaluate=False)
        elif self._task == 'lemmatize':
            # lemma
            self._lemma_model = LemmaWrapper(self._config, treebank_name=self._config.treebank_name,
                                             use_gpu=self._use_gpu, evaluate=False)
        elif self._task == 'ner':
            self._embedding_layers = Multilingual_Embedding(self._config, model_name='ner')
            self._embedding_layers.to(self._config.device)

            self._ner_model = NERClassifier(self._config, self._lang)
            self._ner_model.to(self._config.device)

            self.model_parameters = [(n, p) for n, p in self._embedding_layers.named_parameters()] + \
                                    [(n, p) for n, p in self._ner_model.named_parameters()]

        # optimizer
        if self._task in ['tokenize', 'posdep', 'ner']:
            param_groups = [
                {
                    'params': [p for n, p in self.model_parameters if 'task_adapters' in n if
                               p.requires_grad],
                    'lr': self._config.adapter_learning_rate, 'weight_decay': self._config.adapter_weight_decay
                },
                {
                    'params': [p for n, p in self.model_parameters if 'task_adapters' not in n if
                               p.requires_grad],
                    'lr': self._config.learning_rate, 'weight_decay': self._config.weight_decay
                }
            ]
            self.optimizer = AdamW(params=param_groups)

            self.schedule = get_linear_schedule_with_warmup(self.optimizer,
                                                            num_warmup_steps=self.batch_num * 5,
                                                            num_training_steps=self.batch_num * self._config.max_epoch)

    def _detect_split_by_space_lang(self, train_txt_fpath):
        if train_txt_fpath is None:
            return True
        else:
            with open(train_txt_fpath) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            line_ids = list(range(len(lines)))
            random.shuffle(line_ids)
            _100_random_lines = [lines[lid] for lid in line_ids[:100]]
            split_by_space = 0.
            for line in _100_random_lines:
                if len(line[:100]) > len(line[:100].replace(' ', '')):
                    split_by_space += 1

            if split_by_space / len(_100_random_lines) > 0.8:
                return True
            else:
                return False

    def _set_up_config(self, training_config):
        print('Setting up training config...')
        # set random seed
        os.environ['PYTHONHASHSEED'] = str(1234)
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # empty cache
        torch.cuda.empty_cache()

        # set embedding name
        master_config.embedding_name = 'xlm-roberta-base' if 'embedding' not in training_config else training_config['embedding']
        assert master_config.embedding_name in supported_embeddings, '{} has not been supported.\nSupported embeddings: {}'.format(
            master_config.embedding_name, supported_embeddings)

        # lang and data
        self._lang = training_config['category'] if 'category' in training_config else 'customized'
        self._task = training_config['task']
        assert self._task in ['tokenize', 'mwt', 'posdep', 'lemmatize', 'ner']

        # the following variables are used for UD training
        self._train_txt_fpath = training_config['train_txt_fpath'] if 'train_txt_fpath' in training_config else None
        self._train_conllu_fpath = training_config[
            'train_conllu_fpath'] if 'train_conllu_fpath' in training_config else None
        self._dev_txt_fpath = training_config['dev_txt_fpath'] if 'dev_txt_fpath' in training_config else None
        self._dev_conllu_fpath = training_config['dev_conllu_fpath'] if 'dev_conllu_fpath' in training_config else None
        # the following variables are used for NER training
        self._train_bio_fpath = training_config['train_bio_fpath'] if 'train_bio_fpath' in training_config else None
        self._dev_bio_fpath = training_config['dev_bio_fpath'] if 'dev_bio_fpath' in training_config else None

        master_config.train_conllu_fpath = self._train_conllu_fpath
        master_config.dev_conllu_fpath = self._dev_conllu_fpath

        if self._task == 'tokenize':
            assert self._train_txt_fpath and self._train_conllu_fpath and self._dev_txt_fpath and self._dev_conllu_fpath, 'Missing one of these files: (i) train/dev txt file containing raw text (ii) train/dev conllu file containing annotated labels'
        elif self._task in ['posdep', 'mwt', 'lemmatize']:
            assert self._train_conllu_fpath and self._dev_conllu_fpath, 'Missing one of these files: train/dev conllu file containing annotated labels'
        elif self._task == 'ner':
            assert self._train_bio_fpath and self._dev_bio_fpath, 'Missing one of these files: train/dev BIO file containing annotated NER labels'
        # detect if text in this language is split by spaces or not
        self._text_split_by_space = self._detect_split_by_space_lang(self._train_txt_fpath)
        if not self._text_split_by_space:
            treebank_name = 'UD_Japanese-like'  # use this special name to note that text is not split by spaces, similar to Japanese language.
        else:
            treebank_name = lang2treebank.get(self._lang, 'UD_{}-New'.format(self._lang))
        lang2treebank[self._lang] = treebank_name
        treebank2lang[treebank_name] = self._lang

        # device and save dir
        self._save_dir = training_config['save_dir'] if 'save_dir' in training_config else './cache/'
        self._save_dir = os.path.join(self._save_dir, self._lang)
        self._cache_dir = self._save_dir
        self._gpu = training_config['gpu'] if 'gpu' in training_config else True
        self._use_gpu = training_config['gpu'] if 'gpu' in training_config else True
        self._ud_eval = True
        if self._gpu and torch.cuda.is_available():
            self._use_gpu = True
            master_config.device = torch.device('cuda')
        else:
            self._use_gpu = False
            master_config.device = torch.device('cpu')

        master_config._save_dir = self._save_dir
        master_config._cache_dir = self._save_dir
        ensure_dir(self._save_dir)
        self._config = master_config
        self._config.training = True
        self._config.lang = self._lang
        self._config.treebank_name = treebank_name

        # training hyper-parameters
        self._config.max_input_length = training_config[
            'max_input_length'] if 'max_input_length' in training_config else 512  # this is for tokenizer only

        if 'batch_size' in training_config:
            self._config.batch_size = training_config['batch_size']
        elif training_config['task'] == 'tokenize':
            self._config.batch_size = 4
        elif training_config['task'] in ['posdep', 'ner']:
            self._config.batch_size = 16
        elif training_config['task'] in ['mwt', 'lemmatize']:
            self._config.batch_size = 50

        self._config.max_epoch = training_config['max_epoch'] if 'max_epoch' in training_config else 100

        # logging
        log_dir = os.path.join(self._save_dir, 'logs')
        ensure_dir(log_dir)
        for name in logging.root.manager.loggerDict:
            if 'transformers' in name:
                logging.getLogger(name).setLevel(logging.CRITICAL)

        logging.basicConfig(format='%(message)s', level=logging.INFO,
                            filename=os.path.join(log_dir, '{}.training'.format(self._task)),
                            filemode='w')
        self.logger = logging.getLogger(__name__)
        self._config.logger = self.logger

        # wordpiece splitter
        if self._task not in ['mwt', 'lemmatize']:
            master_config.wordpiece_splitter = XLMRobertaTokenizer.from_pretrained(master_config.embedding_name,
                                                                                   cache_dir=os.path.join(
                                                                                       master_config._save_dir,
                                                                                       master_config.embedding_name))

    def _prepare_tokenize(self):
        self.train_set = TokenizeDataset(
            self._config,
            txt_fpath=self._train_txt_fpath,
            conllu_fpath=self._train_conllu_fpath,
            evaluate=False
        )
        self.train_set.numberize()
        self.batch_num = len(self.train_set) // self._config.batch_size

        self.dev_set = TokenizeDataset(
            self._config,
            txt_fpath=self._dev_txt_fpath,
            conllu_fpath=self._dev_conllu_fpath,
            evaluate=True
        )
        self.dev_set.numberize()
        self.dev_batch_num = len(self.dev_set) // self._config.batch_size + \
                             (len(self.dev_set) % self._config.batch_size != 0)

    def _printlog(self, message, printout=True):
        if printout:
            print(message)
        self.logger.info(message)

    def _prepare_mwt(self):
        return None

    def _prepare_posdep(self):
        in_conllu = {
            'dev': os.path.join(self._config._save_dir, 'preds', 'mwt.dev.conllu')
        }
        if not os.path.exists(in_conllu['dev']):
            in_conllu = {
                'dev': os.path.join(self._config._save_dir, 'preds', 'tokenizer.dev.conllu')
            }
            if not os.path.exists(in_conllu['dev']):
                in_conllu = {
                    'dev': self._dev_conllu_fpath
                }

        self.train_set = TaggerDataset(
            self._config,
            input_conllu=self._train_conllu_fpath,
            gold_conllu=self._train_conllu_fpath,
            evaluate=False
        )
        self.train_set.numberize()
        self.batch_num = len(self.train_set) // self._config.batch_size

        # load vocabs
        self._config.vocabs = {
            self._config.treebank_name: self.train_set.vocabs
        }
        self._config.itos = {}
        self._config.itos[UPOS] = {v: k for k, v in self.train_set.vocabs[UPOS].items()}
        self._config.itos[XPOS] = {v: k for k, v in self.train_set.vocabs[XPOS].items()}
        self._config.itos[FEATS] = {v: k for k, v in self.train_set.vocabs[FEATS].items()}
        self._config.itos[DEPREL] = {v: k for k, v in self.train_set.vocabs[DEPREL].items()}

        self.dev_set = TaggerDataset(
            self._config,
            input_conllu=in_conllu['dev'],
            gold_conllu=self._dev_conllu_fpath,
            evaluate=True
        )
        self.dev_set.numberize()
        self.dev_batch_num = len(self.dev_set) // self._config.batch_size + \
                             (len(self.dev_set) % self._config.batch_size != 0)

    def _prepare_lemma(self):
        return None

    def _prepare_ner(self):
        self.train_set = NERDataset(
            config=self._config,
            bio_fpath=self._train_bio_fpath,
            evaluate=False
        )
        self.train_set.numberize()
        self.batch_num = len(self.train_set) // self._config.batch_size

        self.dev_set = NERDataset(
            config=self._config,
            bio_fpath=self._dev_bio_fpath,
            evaluate=True
        )
        self.dev_set.numberize()
        self.dev_batch_num = len(self.dev_set) // self._config.batch_size + \
                             (len(self.dev_set) % self._config.batch_size != 0)

        # load vocab and itos
        self._config.ner_vocabs = {}
        self._config.ner_vocabs[self._config.lang] = self.train_set.vocabs
        self.tag_itos = {v: k for k, v in self._config.ner_vocabs[self._config.lang].items()}

    def _prepare_data_and_vocabs(self):
        if self._task == 'tokenize':
            self._prepare_tokenize()
        elif self._task == 'mwt':
            self._prepare_mwt()
        elif self._task == 'posdep':
            self._prepare_posdep()
        elif self._task == 'lemmatize':
            self._prepare_lemma()
        elif self._task == 'ner':
            self._prepare_ner()

    def _train_tokenize(self):
        ensure_dir(os.path.join(self._config._save_dir, 'preds'))
        best_dev = {'average': 0}
        best_epoch = 0
        for epoch in range(self._config.max_epoch):
            self._printlog('*' * 30)
            print('Tokenizer: Epoch: {}'.format(epoch))
            # training set
            progress = tqdm(total=self.batch_num, ncols=75,
                            desc='Train {}'.format(epoch))
            self._embedding_layers.train()
            self._tokenizer.train()
            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(DataLoader(
                    self.train_set, batch_size=self._config.batch_size,
                    shuffle=True, collate_fn=self.train_set.collate_fn)):
                progress.update(1)
                wordpiece_reprs = self._embedding_layers.get_tokenizer_inputs(batch)
                loss = self._tokenizer(wordpiece_reprs, batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_([p for n, p in self.model_parameters], self._config.grad_clipping)
                self.optimizer.step()
                self.schedule.step()
                self.optimizer.zero_grad()
                self._printlog(
                    'tokenizer: step: {}/{}, loss: {}'.format(batch_idx + 1, self.batch_num, loss.item()),
                    printout=False
                )
            progress.close()
            dev_score, pred_conllu_fpath = self._eval_tokenize(data_set=self.dev_set, batch_num=self.dev_batch_num,
                                                               name='dev', epoch=epoch)

            if epoch <= 30 or dev_score['average'] > best_dev['average']:
                self._save_model(ckpt_fpath=os.path.join(self._config._save_dir,
                                                         '{}.tokenizer.mdl'.format(self._lang)),
                                 epoch=epoch)
                best_dev = dev_score
                best_epoch = epoch

                os.rename(
                    pred_conllu_fpath,
                    os.path.join(self._config._save_dir, 'preds', 'tokenizer.dev.conllu')
                )

            remove_with_path(pred_conllu_fpath)
            self._printlog('-' * 30 + ' Best dev CoNLLu score: epoch {}'.format(best_epoch) + '-' * 30)
            self._printlog(get_ud_performance_table(dev_score))

    def _eval_tokenize(self, data_set, batch_num, name, epoch):
        self._embedding_layers.eval()
        self._tokenizer.eval()
        # evaluate
        progress = tqdm(total=batch_num, ncols=75,
                        desc='{} {}'.format(name, epoch))
        wordpiece_pred_labels, wordpiece_ends, paragraph_indexes = [], [], []
        for batch in DataLoader(data_set, batch_size=self._config.batch_size,
                                shuffle=False, collate_fn=data_set.collate_fn):
            progress.update(1)
            wordpiece_reprs = self._embedding_layers.get_tokenizer_inputs(batch)
            predictions = self._tokenizer.predict(batch, wordpiece_reprs)
            wp_pred_labels, wp_ends, para_ids = predictions[0], predictions[1], predictions[2]
            wp_pred_labels = wp_pred_labels.data.cpu().numpy().tolist()

            for i in range(len(wp_pred_labels)):
                wordpiece_pred_labels.append(wp_pred_labels[i][: len(wp_ends[i])])

            wordpiece_ends.extend(wp_ends)
            paragraph_indexes.extend(para_ids)
        progress.close()
        # mapping
        para_id_to_wp_pred_labels = defaultdict(list)

        for wp_pred_ls, wp_es, p_index in zip(wordpiece_pred_labels, wordpiece_ends,
                                              paragraph_indexes):
            para_id_to_wp_pred_labels[p_index].extend([(pred, char_position) for pred, char_position in
                                                       zip(wp_pred_ls, wp_es)])
        # compute scores
        with open(data_set.plaintext_file, 'r') as f:
            corpus_text = ''.join(f.readlines())

        paragraphs = [pt.rstrip() for pt in
                      NEWLINE_WHITESPACE_RE.split(corpus_text) if
                      len(pt.rstrip()) > 0]
        all_wp_preds = []
        all_raw = []
        ##############
        for para_index, para_text in enumerate(paragraphs):
            para_wp_preds = [0 for _ in para_text]
            for wp_l, end_position in para_id_to_wp_pred_labels[para_index]:
                para_wp_preds[end_position] = wp_l

            all_wp_preds.append(para_wp_preds)
            all_raw.append(para_text)
        ###########################3
        offset = 0
        doc = []
        for j in range(len(paragraphs)):
            raw = all_raw[j]
            wp_pred = all_wp_preds[j]

            current_tok = ''
            current_sent = []

            for t, wp_p in zip(raw, wp_pred):
                offset += 1
                current_tok += t
                if wp_p >= 1:
                    tok = normalize_token(data_set.treebank_name, current_tok)
                    assert '\t' not in tok, tok
                    if len(tok) <= 0:
                        current_tok = ''
                        continue
                    additional_info = dict()
                    current_sent += [(tok, wp_p, additional_info)]
                    current_tok = ''
                    if (wp_p == 2 or wp_p == 4):
                        doc.append(tget_output_sentence(current_sent))
                        current_sent = []

            if len(current_tok):
                tok = normalize_token(data_set.treebank_name, current_tok)
                assert '\t' not in tok, tok
                if len(tok) > 0:
                    additional_info = dict()
                    current_sent += [(tok, 2, additional_info)]

            if len(current_sent):
                doc.append(tget_output_sentence(current_sent))

        pred_conllu_fpath = os.path.join(self._config._save_dir, 'preds',
                                         'tokenizer.{}.conllu'.format(name) + '.epoch-{}'.format(epoch))
        gold_conllu_fpath = data_set.conllu_file

        CoNLL.dict2conll(doc, pred_conllu_fpath)

        score = get_ud_score(pred_conllu_fpath, gold_conllu_fpath)
        score['epoch'] = epoch
        return score, pred_conllu_fpath

    def _train_mwt(self):
        self._mwt_model.train()

    def _train_posdep(self):
        ensure_dir(os.path.join(self._config._save_dir, 'preds'))
        best_dev = {'average': 0}
        best_epoch = 0
        for epoch in range(self._config.max_epoch):
            self._printlog('*' * 30)
            print('Posdep tagger: Epoch: {}'.format(epoch))
            # training set
            progress = tqdm(total=self.batch_num, ncols=75,
                            desc='Train {}'.format(epoch))
            self._embedding_layers.train()
            self._tagger.train()
            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(DataLoader(
                    self.train_set, batch_size=self._config.batch_size,
                    shuffle=True, collate_fn=self.train_set.collate_fn)):
                progress.update(1)
                word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
                loss = self._tagger(batch, word_reprs, cls_reprs)
                loss.backward()

                torch.nn.utils.clip_grad_norm_([p for n, p in self.model_parameters], self._config.grad_clipping)
                self.optimizer.step()
                self.schedule.step()
                self.optimizer.zero_grad()
                self._printlog(
                    'posdep tagger: step: {}/{}, loss: {}'.format(batch_idx + 1, self.batch_num, loss.item()),
                    printout=False
                )
            progress.close()
            dev_score, pred_conllu_fpath = self._eval_posdep(data_set=self.dev_set, batch_num=self.dev_batch_num,
                                                             name='dev', epoch=epoch)

            if epoch <= 30 or dev_score['average'] > best_dev['average']:
                self._save_model(ckpt_fpath=os.path.join(self._config._save_dir,
                                                         '{}.tagger.mdl'.format(self._lang)),
                                 epoch=epoch)
                best_dev = dev_score
                best_epoch = epoch
                os.rename(
                    pred_conllu_fpath,
                    os.path.join(self._config._save_dir, 'preds', 'tagger.dev.conllu')
                )
            remove_with_path(pred_conllu_fpath)
            self._printlog('-' * 30 + ' Best dev CoNLLu score: epoch {}'.format(best_epoch) + '-' * 30)
            self._printlog(get_ud_performance_table(dev_score))

    def _eval_posdep(self, data_set, batch_num, name, epoch):
        self._embedding_layers.eval()
        self._tagger.eval()
        # evaluate
        progress = tqdm(total=batch_num, ncols=75,
                        desc='{} {}'.format(name, epoch))

        for batch in DataLoader(data_set, batch_size=self._config.batch_size,
                                shuffle=False, collate_fn=data_set.collate_fn):
            batch_size = len(batch.word_num)

            progress.update(1)
            word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
            predictions = self._tagger.predict(batch, word_reprs, cls_reprs)
            predicted_upos = predictions[0]
            predicted_xpos = predictions[1]
            predicted_feats = predictions[2]

            predicted_upos = predicted_upos.data.cpu().numpy().tolist()
            predicted_xpos = predicted_xpos.data.cpu().numpy().tolist()
            predicted_feats = predicted_feats.data.cpu().numpy().tolist()

            # head, deprel
            predicted_dep = predictions[3]
            sentlens = [l + 1 for l in batch.word_num]
            head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in
                         zip(predicted_dep[0], sentlens)]
            deprel_seqs = [[self._config.itos[DEPREL][predicted_dep[1][i][j + 1][h]] for j, h in
                            enumerate(hs)] for
                           i, hs
                           in
                           enumerate(head_seqs)]

            pred_tokens = [[[head_seqs[i][j], deprel_seqs[i][j]] for j in range(sentlens[i] - 1)] for i in
                           range(batch_size)]

            for bid in range(batch_size):
                for i in range(batch.word_num[bid]):
                    sentid = batch.sent_index[bid]
                    wordid = batch.word_ids[bid][i]

                    # upos
                    pred_upos_id = predicted_upos[bid][i]
                    upos_name = self._config.itos[UPOS][pred_upos_id]
                    data_set.conllu_doc[sentid][wordid][UPOS] = upos_name
                    # xpos
                    pred_xpos_id = predicted_xpos[bid][i]
                    xpos_name = self._config.itos[XPOS][pred_xpos_id]
                    data_set.conllu_doc[sentid][wordid][XPOS] = xpos_name
                    # feats
                    pred_feats_id = predicted_feats[bid][i]
                    feats_name = self._config.itos[FEATS][pred_feats_id]
                    data_set.conllu_doc[sentid][wordid][FEATS] = feats_name

                    # head
                    data_set.conllu_doc[sentid][wordid][HEAD] = int(pred_tokens[bid][i][0])
                    # deprel
                    data_set.conllu_doc[sentid][wordid][DEPREL] = pred_tokens[bid][i][1]

        progress.close()
        pred_conllu_fpath = os.path.join(self._config._save_dir, 'preds',
                                         'tagger.{}.conllu'.format(name) + '.epoch-{}'.format(epoch))
        doc = tget_output_doc(conllu_doc=data_set.conllu_doc)
        CoNLL.dict2conll(doc, pred_conllu_fpath)
        score = get_ud_score(pred_conllu_fpath, data_set.gold_conllu)
        score['epoch'] = epoch
        return score, pred_conllu_fpath

    def _train_lemma(self):
        self._lemma_model.train()

    def _train_ner(self):
        best_dev = {'p': 0, 'r': 0, 'f1': 0}
        best_epoch = 0
        for epoch in range(self._config.max_epoch):
            self._printlog('*' * 30)
            self._printlog('NER: Epoch: {}'.format(epoch))
            # training set
            progress = tqdm(total=self.batch_num, ncols=75,
                            desc='Train {}'.format(epoch))
            self._embedding_layers.train()
            self._ner_model.train()
            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(DataLoader(
                    self.train_set, batch_size=self._config.batch_size,
                    shuffle=True, collate_fn=self.train_set.collate_fn)):
                progress.update(1)
                word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
                loss = self._ner_model(batch, word_reprs)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self._ner_model.parameters(), self._config.grad_clipping)
                self.optimizer.step()
                self.schedule.step()
                self.optimizer.zero_grad()
                self._printlog('NER: step: {}/{}, loss: {}'.format(batch_idx + 1, self.batch_num, loss.item()),
                               printout=False)
            progress.close()
            dev_score = self._eval_ner(data_set=self.dev_set, batch_num=self.dev_batch_num,
                                       name='dev', epoch=epoch)

            if dev_score['f1'] > best_dev['f1']:
                self._save_model(ckpt_fpath=os.path.join(self._config._save_dir,
                                                         '{}.ner.mdl'.format(self._lang)),
                                 epoch=epoch)

                best_dev = dev_score
                best_epoch = epoch

            # printout current best dev
            self._printlog('-' * 30)
            self._printlog('Best dev F1 score: epoch {}, F1: {:.2f}'.format(best_epoch, best_dev['f1']))
        print('Training done!')

    def _eval_ner(self, data_set, batch_num, name, epoch):
        self._ner_model.eval()
        # evaluate
        progress = tqdm(total=batch_num, ncols=75,
                        desc='{} {}'.format(name, epoch))
        predictions = []
        golds = []
        for batch in DataLoader(data_set, batch_size=self._config.batch_size,
                                shuffle=False, collate_fn=data_set.collate_fn):
            progress.update(1)
            word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
            pred_entity_labels = self._ner_model.predict(batch, word_reprs)
            predictions += pred_entity_labels
            batch_entity_labels = batch.entity_label_idxs.data.cpu().numpy().tolist()
            golds += [[self.tag_itos[l] for l in seq[:batch.word_num[i]]] for i, seq in enumerate(batch_entity_labels)]
        progress.close()
        score = score_by_entity(predictions, golds, self.logger)
        return score

    def _save_model(self, ckpt_fpath, epoch):
        trainable_weight_names = [n for n, p in self.model_parameters if p.requires_grad]
        state = {
            'adapters': {},
            'epoch': epoch
        }
        for k, v in self._embedding_layers.state_dict().items():
            if k in trainable_weight_names:
                state['adapters'][k] = v
        if self._task == 'tokenize':
            for k, v in self._tokenizer.state_dict().items():
                if k in trainable_weight_names:
                    state['adapters'][k] = v
        elif self._task == 'posdep':
            for k, v in self._tagger.state_dict().items():
                if k in trainable_weight_names:
                    state['adapters'][k] = v
        elif self._task == 'ner':
            for k, v in self._ner_model.state_dict().items():
                if k in trainable_weight_names:
                    state['adapters'][k] = v

        torch.save(state, ckpt_fpath)
        print('Saving adapter weights to ... {} ({:.2f} MB)'.format(ckpt_fpath,
                                                                    os.path.getsize(ckpt_fpath) * 1. / (1024 * 1024)))

    def train(self):
        if self._task == 'tokenize':
            self._train_tokenize()
        elif self._task == 'mwt':
            self._train_mwt()
        elif self._task == 'posdep':
            self._train_posdep()
        elif self._task == 'lemmatize':
            self._train_lemma()
        elif self._task == 'ner':
            self._train_ner()
