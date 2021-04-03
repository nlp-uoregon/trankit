from .config import config as master_config
from .models.base_models import Multilingual_Embedding
from .models.classifiers import TokenizerClassifier, PosDepClassifier, NERClassifier
from .models.mwt_model import MWTWrapper
from .models.lemma_model import LemmaWrapper
from .iterators.tokenizer_iterators import TokenizeDatasetLive
from .iterators.tagger_iterators import TaggerDatasetLive
from .iterators.ner_iterators import NERDatasetLive
from .utils.tokenizer_utils import *
from collections import defaultdict
from .utils.conll import *
from .utils.tbinfo import tbname2training_id, lang2treebank
from .utils.chuliu_edmonds import *
from .adapter_transformers import XLMRobertaTokenizer
from datetime import datetime
import langid


def is_string(input):
    if type(input) == str and len(input.strip()) > 0:
        return True
    return False


def is_list_strings(input):
    if type(input) == list and len(input) > 0:
        for element in input:
            if not (type(element) == str and not element.isspace()):
                return False
        return True
    return False


def is_list_list_strings(input):
    if type(input) == list and len(input) > 0 and type(input[0]) == list and len(input[0]) > 0:
        for element in input[0]:
            if not (type(element) == str and not element.isspace()):
                return False
        return True
    return False


class Pipeline:
    def __init__(self, lang, cache_dir=None, gpu=True, embedding='xlm-roberta-base'):
        super(Pipeline, self).__init__()
        # auto detection of lang
        if lang == 'auto':
            lang = list(code2lang.values())[0]
            self.auto_mode = True
        else:
            self.auto_mode = False

        # set the embedding type
        assert embedding in supported_embeddings, '{} has not been supported.\nSupported embeddings: {}'.format(
            embedding, supported_embeddings)
        master_config.embedding_name = embedding

        self._cache_dir = cache_dir
        self._gpu = gpu
        self._use_gpu = gpu
        self._ud_eval = False
        self._setup_config(lang)
        self._config.training = False
        self.added_langs = [lang]
        for lang in self.added_langs:
            assert lang in lang2treebank, '{} has not been supported. Currently supported languages: {}'.format(lang,
                                                                                                                list(
                                                                                                                    lang2treebank.keys()))
        # download saved model for initial language
        download(
            cache_dir=self._config._cache_dir,
            language=lang,
            saved_model_version='v1.0.0',  # manually set this to avoid duplicated storage
            embedding_name=master_config.embedding_name
        )

        # load ALL vocabs
        self._load_vocabs()

        # shared multilingual embeddings
        print('Loading pretrained XLM-Roberta, this may take a while...')
        self._embedding_layers = Multilingual_Embedding(self._config)
        self._embedding_layers.to(self._config.device)
        if self._use_gpu:
            self._embedding_layers.half()
        self._embedding_layers.eval()
        # tokenizers
        self._tokenizer = {}
        for lang in self.added_langs:
            self._tokenizer[lang] = TokenizerClassifier(self._config, treebank_name=lang2treebank[lang])
            self._tokenizer[lang].to(self._config.device)
            if self._use_gpu:
                self._tokenizer[lang].half()
            self._tokenizer[lang].eval()
        # taggers
        self._tagger = {}
        for lang in self.added_langs:
            self._tagger[lang] = PosDepClassifier(self._config, treebank_name=lang2treebank[lang])
            self._tagger[lang].to(self._config.device)
            if self._use_gpu:
                self._tagger[lang].half()
            self._tagger[lang].eval()

        #     - mwt and lemma:
        self._mwt_model = {}
        for lang in self.added_langs:
            treebank_name = lang2treebank[lang]
            if tbname2training_id[treebank_name] % 2 == 1:
                self._mwt_model[lang] = MWTWrapper(self._config, treebank_name=treebank_name, use_gpu=self._use_gpu)
        self._lemma_model = {}
        for lang in self.added_langs:
            treebank_name = lang2treebank[lang]
            self._lemma_model[lang] = LemmaWrapper(self._config, treebank_name=treebank_name, use_gpu=self._use_gpu)
        # ner if possible
        self._ner_model = {}
        for lang in self.added_langs:
            if lang in langwithner:
                self._ner_model[lang] = NERClassifier(self._config, lang)
                self._ner_model[lang].to(self._config.device)
                if self._use_gpu:
                    self._ner_model[lang].half()
                self._ner_model[lang].eval()

        # load and hold the pretrained weights
        self._embedding_weights = self._embedding_layers.state_dict()

        if self.auto_mode:
            for l in code2lang.values():
                if l not in self.added_langs:
                    self.add(l)
            # constrain the language set for auto mode
            langid.set_languages([lang2code[l] for l in self.added_langs])
            self.code2lang = code2lang
            print('=' * 50)
            print('Trankit is in auto mode!\nAvailable languages: {}'.format(self.added_langs))
            print('=' * 50)
        else:
            self.set_active(lang)

    def _setup_config(self, lang):
        torch.cuda.empty_cache()
        # decide whether to run on GPU or CPU
        if self._gpu and torch.cuda.is_available():
            self._use_gpu = True
            master_config.device = torch.device('cuda')
            self._tokbatchsize = 6
            self._tagbatchsize = 24
        else:
            self._use_gpu = False
            master_config.device = torch.device('cpu')
            self._tokbatchsize = 2
            self._tagbatchsize = 12

        if self._cache_dir is None:
            master_config._cache_dir = 'cache/trankit'
        else:
            master_config._cache_dir = self._cache_dir

        if not os.path.exists(master_config._cache_dir):
            os.makedirs(master_config._cache_dir, exist_ok=True)

        master_config.wordpiece_splitter = XLMRobertaTokenizer.from_pretrained(master_config.embedding_name,
                                                                               cache_dir=os.path.join(
                                                                                   master_config._cache_dir,
                                                                                   master_config.embedding_name))
        self._config = master_config
        self._config.max_input_length = tbname2max_input_length.get(lang2treebank[lang],
                                                                    400)  # this is for tokenizer only

    def set_auto(self, state):
        assert type(state) == bool
        if state is True:
            print('Turning on auto mode for {} ...'.format(self.added_langs))
            self.auto_mode = True

            cls_codes = []
            self.code2lang = {}
            for l in self.added_langs:
                if l in extra_lang2code:
                    cls_codes.append(extra_lang2code[l])
                    self.code2lang[extra_lang2code[l]] = l

            langid.set_languages(cls_codes)
            print('=' * 50)
            print('Trankit is in auto mode!')
            print('=' * 50)
        else:
            self.auto_mode = False
            lang = self.added_langs[0]
            self._config.active_lang = lang
            self.active_lang = lang
            self._config.treebank_name = lang2treebank[lang]
            self._config.max_input_length = tbname2max_input_length.get(lang2treebank[lang],
                                                                        400)  # this is for tokenizer only
            print('=' * 50)
            print('Trankit is in normal mode!')
            print('=' * 50)
            print('Active language: {}'.format(self._config.active_lang))
            print('Available languages: {}'.format(self.added_langs))
            print('=' * 50)

    def set_active(self, lang):
        assert not self.auto_mode, 'Cannot set a particular language as active in auto mode.\nPlease consider using Trankit in the normal mode to use this function.'
        assert is_string(
            lang) and lang in self.added_langs, 'Specified language must be added before being activated.\nCurrent added languages: {}'.format(
            self.added_langs)
        self._config.active_lang = lang
        self.active_lang = lang
        self._config.treebank_name = lang2treebank[lang]
        self._config.max_input_length = tbname2max_input_length.get(lang2treebank[lang],
                                                                    400)  # this is for tokenizer only
        print('=' * 50)
        print('Active language: {}'.format(self._config.active_lang))
        print('=' * 50)

    def add(self, lang):
        assert is_string(
            lang) and lang in supported_langs, 'Specified language must be one of the supported languages: {}'.format(
            supported_langs)
        # download saved models
        download(
            cache_dir=self._config._cache_dir,
            language=lang,
            saved_model_version='v1.0.0',  # manually set this to avoid duplicated storage
            embedding_name=master_config.embedding_name
        )
        # update vocabs
        treebank_name = lang2treebank[lang]
        with open(os.path.join(self._config._cache_dir, master_config.embedding_name,
                               '{}/{}.vocabs.json'.format(treebank2lang[treebank_name],
                                                          treebank2lang[treebank_name]))) as f:
            vocabs = json.load(f)
            self._config.vocabs[treebank_name] = vocabs
        if lang in langwithner:
            with open(os.path.join(self._config._cache_dir, master_config.embedding_name,
                                   '{}/{}.ner-vocab.json'.format(lang, lang))) as f:
                self._config.ner_vocabs[lang] = json.load(f)
        self._config.itos[lang][UPOS] = {v: k for k, v in vocabs[UPOS].items()}
        self._config.itos[lang][XPOS] = {v: k for k, v in vocabs[XPOS].items()}
        self._config.itos[lang][FEATS] = {v: k for k, v in vocabs[FEATS].items()}
        self._config.itos[lang][DEPREL] = {v: k for k, v in vocabs[DEPREL].items()}
        # add tokenizer
        self._tokenizer[lang] = TokenizerClassifier(self._config, treebank_name=lang2treebank[lang])
        self._tokenizer[lang].to(self._config.device)
        if self._use_gpu:
            self._tokenizer[lang].half()
        self._tokenizer[lang].eval()
        # add tagger
        self._tagger[lang] = PosDepClassifier(self._config, treebank_name=lang2treebank[lang])
        self._tagger[lang].to(self._config.device)
        if self._use_gpu:
            self._tagger[lang].half()
        self._tagger[lang].eval()
        # mwt if needed
        treebank_name = lang2treebank[lang]
        if tbname2training_id[treebank_name] % 2 == 1:
            self._mwt_model[lang] = MWTWrapper(self._config, treebank_name=treebank_name, use_gpu=self._use_gpu)
        # lemma
        self._lemma_model[lang] = LemmaWrapper(self._config, treebank_name=treebank_name, use_gpu=self._use_gpu)
        # ner if possible
        if lang in langwithner:
            self._ner_model[lang] = NERClassifier(self._config, lang)
            self._ner_model[lang].to(self._config.device)
            if self._use_gpu:
                self._ner_model[lang].half()
            self._ner_model[lang].eval()
        self.added_langs.append(lang)

    def _load_vocabs(self):
        self._config.vocabs = {}
        self._config.ner_vocabs = {}
        self._config.itos = defaultdict(dict)
        for lang in self.added_langs:
            treebank_name = lang2treebank[lang]
            with open(os.path.join(self._config._cache_dir, master_config.embedding_name,
                                   '{}/{}.vocabs.json'.format(lang, lang))) as f:
                vocabs = json.load(f)
                self._config.vocabs[treebank_name] = vocabs
            self._config.itos[lang][UPOS] = {v: k for k, v in vocabs[UPOS].items()}
            self._config.itos[lang][XPOS] = {v: k for k, v in vocabs[XPOS].items()}
            self._config.itos[lang][FEATS] = {v: k for k, v in vocabs[FEATS].items()}
            self._config.itos[lang][DEPREL] = {v: k for k, v in vocabs[DEPREL].items()}
            # ner vocabs
            if lang in langwithner:
                with open(os.path.join(self._config._cache_dir, master_config.embedding_name,
                                       '{}/{}.ner-vocab.json'.format(lang, lang))) as f:
                    self._config.ner_vocabs[lang] = json.load(f)

    def _load_adapter_weights(self, model_name):
        assert model_name in ['tokenizer', 'tagger', 'ner']
        if model_name == 'tokenizer':
            pretrained_weights = self._tokenizer[self._config.active_lang].pretrained_tokenizer_weights
        elif model_name == 'tagger':
            pretrained_weights = self._tagger[self._config.active_lang].pretrained_tagger_weights
        else:
            assert model_name == 'ner'
            pretrained_weights = self._ner_model[self._config.active_lang].pretrained_ner_weights

        for name, value in pretrained_weights.items():
            if 'adapters.{}.adapter'.format(model_name) in name:
                target_name = name.replace('adapters.{}.adapter'.format(model_name), 'adapters.embedding.adapter')
                self._embedding_weights[target_name] = value
        self._embedding_layers.load_state_dict(self._embedding_weights)

    def _detect_lang_and_switch(self, text):
        detected_code = langid.classify(text)[0]
        assert detected_code in self.code2lang, 'Detected code "{}" must be in {}'.format(detected_code,
                                                                                          self.code2lang.keys())
        lang = self.code2lang[detected_code]

        assert is_string(
            lang) and lang in self.added_langs, 'Specified language must be added before being activated.\nCurrent added languages: {}'.format(
            self.added_langs)
        self._config.active_lang = lang
        self.active_lang = lang
        self._config.treebank_name = lang2treebank[lang]
        self._config.max_input_length = tbname2max_input_length.get(lang2treebank[lang],
                                                                    400)  # this is for tokenizer only
        # print('=' * 50)
        # print('Switching to {}'.format(lang))
        # print('=' * 50)

    def ssplit(self, in_doc):  # assuming input is a document
        assert is_string(in_doc), 'Input must be a non-empty string.'
        # switch to detected lang if auto mode is on
        if self.auto_mode:
            self._detect_lang_and_switch(text=in_doc)

        eval_batch_size = tbname2tokbatchsize.get(lang2treebank[self.active_lang], self._tokbatchsize)
        # load input text
        config = self._config
        test_set = TokenizeDatasetLive(config, in_doc, max_input_length=tbname2max_input_length.get(
            lang2treebank[self.active_lang], 400))
        test_set.numberize(config.wordpiece_splitter)

        # load weights of tokenizer into the combined model
        self._load_adapter_weights(model_name='tokenizer')

        # make predictions
        wordpiece_pred_labels, wordpiece_ends, paragraph_indexes = [], [], []
        for batch in DataLoader(test_set, batch_size=eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn):
            wordpiece_reprs = self._embedding_layers.get_tokenizer_inputs(batch)
            predictions = self._tokenizer[self._config.active_lang].predict(batch, wordpiece_reprs)
            wp_pred_labels, wp_ends, para_ids = predictions[0], predictions[1], predictions[2]
            wp_pred_labels = wp_pred_labels.data.cpu().numpy().tolist()

            for i in range(len(wp_pred_labels)):
                wordpiece_pred_labels.append(wp_pred_labels[i][: len(wp_ends[i])])

            wordpiece_ends.extend(wp_ends)
            paragraph_indexes.extend(para_ids)
            torch.cuda.empty_cache()
        # mapping
        para_id_to_wp_pred_labels = defaultdict(list)

        for wp_pred_ls, wp_es, p_index in zip(wordpiece_pred_labels, wordpiece_ends,
                                              paragraph_indexes):
            para_id_to_wp_pred_labels[p_index].extend([(pred, char_position) for pred, char_position in
                                                       zip(wp_pred_ls, wp_es)])
        # get predictions
        corpus_text = in_doc

        paragraphs = [pt.rstrip() for pt in
                      NEWLINE_WHITESPACE_RE.split(corpus_text) if
                      len(pt.rstrip()) > 0]
        all_wp_preds = []
        all_para_texts = []
        all_para_starts = []
        ##############
        cloned_raw_text = deepcopy(in_doc)
        global_offset = 0
        for para_index, para_text in enumerate(paragraphs):
            cloned_raw_text, start_char_idx = get_start_char_idx(para_text, cloned_raw_text)
            start_char_idx += global_offset
            global_offset = start_char_idx + len(para_text)
            all_para_starts.append(start_char_idx)

            para_wp_preds = [0 for _ in para_text]
            for wp_l, end_position in para_id_to_wp_pred_labels[para_index]:
                para_wp_preds[end_position] = wp_l

            all_wp_preds.append(para_wp_preds)
            all_para_texts.append(para_text)

        ###########################
        sentences = []
        for j in range(len(paragraphs)):
            para_text = all_para_texts[j]
            wp_pred = all_wp_preds[j]
            para_start = all_para_starts[j]

            current_tok = ''
            current_sent = []
            local_position = 0
            for t, wp_p in zip(para_text, wp_pred):
                local_position += 1
                current_tok += t
                if wp_p >= 1:
                    tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                    assert '\t' not in tok, tok
                    if len(tok) <= 0:
                        current_tok = ''
                        continue
                    additional_info = {DSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent += [(tok, wp_p, additional_info)]
                    current_tok = ''
                    if (wp_p == 2 or wp_p == 4):
                        sent_span = (current_sent[0][2][DSPAN][0], current_sent[-1][2][DSPAN][1])
                        sentences.append(
                            {ID: len(sentences) + 1, TEXT: in_doc[sent_span[0]: sent_span[1]],
                             DSPAN: (sent_span[0], sent_span[1])})
                        current_sent = []

            if len(current_tok):
                tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                assert '\t' not in tok, tok
                if len(tok) > 0:
                    additional_info = {DSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent += [(tok, 2, additional_info)]

            if len(current_sent):
                sent_span = (current_sent[0][2][DSPAN][0], current_sent[-1][2][DSPAN][1])
                sentences.append(
                    {ID: len(sentences) + 1, TEXT: in_doc[sent_span[0]: sent_span[1]],
                     DSPAN: (sent_span[0], sent_span[1])})

        torch.cuda.empty_cache()
        return {TEXT: in_doc, SENTENCES: sentences, LANG: self.active_lang}

    def tokenize(self, input, is_sent=False):
        assert is_string(input), 'Input must be a non-empty string.'
        # switch to detected lang if auto mode is on
        if self.auto_mode:
            self._detect_lang_and_switch(text=input)

        if type(input) == str and input.isspace():
            return []
        ori_text = deepcopy(input)
        if is_sent:
            return {TEXT: ori_text, TOKENS: self._tokenize_sent(in_sent=input), LANG: self.active_lang}
        else:
            return {TEXT: ori_text, SENTENCES: self._tokenize_doc(in_doc=input), LANG: self.active_lang}

    def _tokenize_sent(self, in_sent):  # assuming input is a sentence
        eval_batch_size = tbname2tokbatchsize.get(lang2treebank[self.active_lang], self._tokbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 2)

        # load input text
        config = self._config
        test_set = TokenizeDatasetLive(config, in_sent, max_input_length=tbname2max_input_length.get(
            lang2treebank[self.active_lang], 400))
        test_set.numberize(config.wordpiece_splitter)

        # load weights of tokenizer into the combined model
        self._load_adapter_weights(model_name='tokenizer')

        # make predictions
        wordpiece_pred_labels, wordpiece_ends, paragraph_indexes = [], [], []
        for batch in DataLoader(test_set, batch_size=eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn):
            wordpiece_reprs = self._embedding_layers.get_tokenizer_inputs(batch)
            predictions = self._tokenizer[self._config.active_lang].predict(batch, wordpiece_reprs)
            wp_pred_labels, wp_ends, para_ids = predictions[0], predictions[1], predictions[2]
            wp_pred_labels = wp_pred_labels.data.cpu().numpy().tolist()

            for i in range(len(wp_pred_labels)):
                wordpiece_pred_labels.append(wp_pred_labels[i][: len(wp_ends[i])])

            wordpiece_ends.extend(wp_ends)
            paragraph_indexes.extend(para_ids)
        # mapping
        para_id_to_wp_pred_labels = defaultdict(list)

        for wp_pred_ls, wp_es, p_index in zip(wordpiece_pred_labels, wordpiece_ends,
                                              paragraph_indexes):
            para_id_to_wp_pred_labels[p_index].extend([(pred, char_position) for pred, char_position in
                                                       zip(wp_pred_ls, wp_es)])
        # get predictions
        corpus_text = in_sent

        paragraphs = [pt.rstrip() for pt in
                      NEWLINE_WHITESPACE_RE.split(corpus_text) if
                      len(pt.rstrip()) > 0]
        all_wp_preds = []
        all_para_texts = []
        all_para_starts = []
        ##############
        cloned_raw_text = deepcopy(in_sent)
        global_offset = 0
        for para_index, para_text in enumerate(paragraphs):
            cloned_raw_text, start_char_idx = get_start_char_idx(para_text, cloned_raw_text)
            start_char_idx += global_offset
            global_offset = start_char_idx + len(para_text)
            all_para_starts.append(start_char_idx)

            para_wp_preds = [0 for _ in para_text]
            for wp_l, end_position in para_id_to_wp_pred_labels[para_index]:
                para_wp_preds[end_position] = wp_l

            all_wp_preds.append(para_wp_preds)
            all_para_texts.append(para_text)

        ###########################
        tokens = []
        for j in range(len(paragraphs)):
            para_text = all_para_texts[j]
            wp_pred = all_wp_preds[j]
            para_start = all_para_starts[j]

            current_tok = ''
            current_sent = []
            local_position = 0
            for t, wp_p in zip(para_text, wp_pred):
                local_position += 1
                current_tok += t
                if wp_p >= 1:
                    tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                    assert '\t' not in tok, tok
                    if len(tok) <= 0:
                        current_tok = ''
                        continue
                    additional_info = {'current_len': len(tokens),
                                       SSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent += [(tok, wp_p, additional_info)]
                    current_tok = ''
                    if (wp_p == 2 or wp_p == 4):
                        tokens += get_output_sentence(current_sent)
                        current_sent = []

            if len(current_tok):
                tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                assert '\t' not in tok, tok
                if len(tok) > 0:
                    additional_info = {'current_len': len(tokens),
                                       SSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent += [(tok, 2, additional_info)]

            if len(current_sent):
                tokens += get_output_sentence(current_sent)

        # multi-word expansion if required
        if tbname2training_id[self._config.treebank_name] % 2 == 1:
            tokens = self._mwt_expand([{TOKENS: tokens}])[0][TOKENS]
        torch.cuda.empty_cache()
        return tokens

    def _tokenize_doc(self, in_doc):  # assuming input is a document
        eval_batch_size = tbname2tokbatchsize.get(lang2treebank[self.active_lang], self._tokbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 2)
        # load input text
        config = self._config
        test_set = TokenizeDatasetLive(config, in_doc, max_input_length=tbname2max_input_length.get(
            lang2treebank[self.active_lang], 400))
        test_set.numberize(config.wordpiece_splitter)

        # load weights of tokenizer into the combined model
        self._load_adapter_weights(model_name='tokenizer')

        # make predictions
        wordpiece_pred_labels, wordpiece_ends, paragraph_indexes = [], [], []
        for batch in DataLoader(test_set, batch_size=eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn):
            wordpiece_reprs = self._embedding_layers.get_tokenizer_inputs(batch)
            predictions = self._tokenizer[self._config.active_lang].predict(batch, wordpiece_reprs)
            wp_pred_labels, wp_ends, para_ids = predictions[0], predictions[1], predictions[2]
            wp_pred_labels = wp_pred_labels.data.cpu().numpy().tolist()

            for i in range(len(wp_pred_labels)):
                wordpiece_pred_labels.append(wp_pred_labels[i][: len(wp_ends[i])])

            wordpiece_ends.extend(wp_ends)
            paragraph_indexes.extend(para_ids)
        # mapping
        para_id_to_wp_pred_labels = defaultdict(list)

        for wp_pred_ls, wp_es, p_index in zip(wordpiece_pred_labels, wordpiece_ends,
                                              paragraph_indexes):
            para_id_to_wp_pred_labels[p_index].extend([(pred, char_position) for pred, char_position in
                                                       zip(wp_pred_ls, wp_es)])
        # get predictions
        corpus_text = in_doc

        paragraphs = [pt.rstrip() for pt in
                      NEWLINE_WHITESPACE_RE.split(corpus_text) if
                      len(pt.rstrip()) > 0]
        all_wp_preds = []
        all_para_texts = []
        all_para_starts = []
        ##############
        cloned_raw_text = deepcopy(in_doc)
        global_offset = 0
        for para_index, para_text in enumerate(paragraphs):
            cloned_raw_text, start_char_idx = get_start_char_idx(para_text, cloned_raw_text)
            start_char_idx += global_offset
            global_offset = start_char_idx + len(para_text)
            all_para_starts.append(start_char_idx)

            para_wp_preds = [0 for _ in para_text]
            for wp_l, end_position in para_id_to_wp_pred_labels[para_index]:
                para_wp_preds[end_position] = wp_l

            all_wp_preds.append(para_wp_preds)
            all_para_texts.append(para_text)
        ###########################
        doc = []
        for j in range(len(paragraphs)):
            para_text = all_para_texts[j]
            wp_pred = all_wp_preds[j]
            para_start = all_para_starts[j]

            current_tok = ''
            current_sent = []
            local_position = 0
            for t, wp_p in zip(para_text, wp_pred):
                local_position += 1
                current_tok += t
                if wp_p >= 1:
                    tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                    assert '\t' not in tok, tok
                    if len(tok) <= 0:
                        current_tok = ''
                        continue
                    additional_info = {DSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent += [(tok, wp_p, additional_info)]
                    current_tok = ''
                    if (wp_p == 2 or wp_p == 4):
                        processed_sent = get_output_sentence(current_sent)
                        doc.append({
                            ID: len(doc) + 1,
                            TEXT: in_doc[processed_sent[0][DSPAN][0]: processed_sent[-1][DSPAN][
                                1]],
                            TOKENS: processed_sent,
                            DSPAN: (processed_sent[0][DSPAN][0], processed_sent[-1][DSPAN][1])
                        })
                        current_sent = []

            if len(current_tok):
                tok = normalize_token(test_set.treebank_name, current_tok, ud_eval=self._ud_eval)
                assert '\t' not in tok, tok
                if len(tok) > 0:
                    additional_info = {DSPAN: (para_start + local_position - len(tok),
                                               para_start + local_position)}
                    current_sent += [(tok, 2, additional_info)]

            if len(current_sent):
                processed_sent = get_output_sentence(current_sent)
                doc.append({
                    ID: len(doc) + 1,
                    TEXT: in_doc[
                          processed_sent[0][DSPAN][0]: processed_sent[-1][DSPAN][1]],
                    TOKENS: processed_sent,
                    DSPAN: (processed_sent[0][DSPAN][0], processed_sent[-1][DSPAN][1])
                })

        # multi-word expansion if required
        if tbname2training_id[self._config.treebank_name] % 2 == 1:
            doc = self._mwt_expand(doc)
        torch.cuda.empty_cache()
        return doc

    def posdep(self, input, is_sent=False):
        if is_sent:
            assert is_string(input) or is_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of non-empty strings.'

            if is_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=' '.join(input))

                input = [{ID: k + 1, TEXT: w} for k, w in enumerate(input)]
                return {TOKENS: self._posdep_sent(in_sent=input), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)
                ori_text = deepcopy(input)
                return {TEXT: ori_text, TOKENS: self._posdep_sent(in_sent=input), LANG: self.active_lang}

        else:
            assert is_string(input) or is_list_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of lists of non-empty strings.'

            if is_list_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text='\n'.join([' '.join(sent) for sent in input]))

                input = [{ID: sid + 1, TOKENS: [{ID: tid + 1, TEXT: w} for tid, w in enumerate(sent)]} for sid, sent in
                         enumerate(input)]
                return {SENTENCES: self._posdep_doc(in_doc=input), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                ori_text = deepcopy(input)
                return {TEXT: ori_text, SENTENCES: self._posdep_doc(in_doc=input), LANG: self.active_lang}

    def _posdep_sent(self, in_sent):  # assuming input is a sentence
        if type(in_sent) == str:  # input sentence is an untokenized string in this case
            in_sent = self._tokenize_sent(in_sent)
        posdep_sent = deepcopy(in_sent)
        posdep_sent = [{ID: 1, TOKENS: posdep_sent}]
        # load outputs of tokenizer
        config = self._config
        test_set = TaggerDatasetLive(
            tokenized_doc=posdep_sent,
            wordpiece_splitter=config.wordpiece_splitter,
            config=config
        )
        test_set.numberize()

        # load weights of tagger into the combined model
        self._load_adapter_weights(model_name='tagger')

        # make predictions
        eval_batch_size = tbname2tagbatchsize.get(self._config.treebank_name, self._tagbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 3)

        for batch in DataLoader(test_set,
                                batch_size=eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn):
            batch_size = len(batch.word_num)

            word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
            predictions = self._tagger[self._config.active_lang].predict(batch, word_reprs, cls_reprs)
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
                         zip(predicted_dep[0], sentlens)]  # remove attachment for the root
            deprel_seqs = [
                [self._config.itos[self._config.active_lang][DEPREL][predicted_dep[1][i][j + 1][h]] for j, h in
                 enumerate(hs)] for
                i, hs
                in
                enumerate(head_seqs)]

            pred_tokens = [[[head_seqs[i][j], deprel_seqs[i][j]] for j in range(sentlens[i] - 1)] for i in
                           range(batch_size)]

            for bid in range(batch_size):
                sentid = batch.sent_index[bid]
                for i in range(batch.word_num[bid]):
                    wordid = batch.word_ids[bid][i]

                    # upos
                    pred_upos_id = predicted_upos[bid][i]
                    upos_name = self._config.itos[self._config.active_lang][UPOS][pred_upos_id]
                    test_set.conllu_doc[sentid][wordid][UPOS] = upos_name
                    # xpos
                    pred_xpos_id = predicted_xpos[bid][i]
                    xpos_name = self._config.itos[self._config.active_lang][XPOS][pred_xpos_id]
                    test_set.conllu_doc[sentid][wordid][XPOS] = xpos_name
                    # feats
                    pred_feats_id = predicted_feats[bid][i]
                    feats_name = self._config.itos[self._config.active_lang][FEATS][pred_feats_id]
                    test_set.conllu_doc[sentid][wordid][FEATS] = feats_name

                    # head
                    test_set.conllu_doc[sentid][wordid][HEAD] = int(pred_tokens[bid][i][0])
                    # deprel
                    test_set.conllu_doc[sentid][wordid][DEPREL] = pred_tokens[bid][i][1]
        tagged_doc = get_output_doc(posdep_sent, test_set.conllu_doc)
        torch.cuda.empty_cache()
        return tagged_doc[0][TOKENS]

    def _posdep_doc(self, in_doc):  # assuming input is a document
        if type(in_doc) == str:  # in_doc is an untokenized string in this case
            in_doc = self._tokenize_doc(in_doc)
        dposdep_doc = deepcopy(in_doc)
        # load outputs of tokenizer
        config = self._config
        test_set = TaggerDatasetLive(
            tokenized_doc=dposdep_doc,
            wordpiece_splitter=config.wordpiece_splitter,
            config=config
        )
        test_set.numberize()

        # load weights of tagger into the combined model
        self._load_adapter_weights(model_name='tagger')

        # make predictions
        eval_batch_size = tbname2tagbatchsize.get(self._config.treebank_name, self._tagbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 3)

        for batch in DataLoader(test_set,
                                batch_size=eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn):
            batch_size = len(batch.word_num)

            word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
            predictions = self._tagger[self._config.active_lang].predict(batch, word_reprs, cls_reprs)
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
                         zip(predicted_dep[0], sentlens)]  # remove attachment for the root
            deprel_seqs = [
                [self._config.itos[self._config.active_lang][DEPREL][predicted_dep[1][i][j + 1][h]] for j, h in
                 enumerate(hs)] for
                i, hs
                in
                enumerate(head_seqs)]

            pred_tokens = [[[head_seqs[i][j], deprel_seqs[i][j]] for j in range(sentlens[i] - 1)] for i in
                           range(batch_size)]

            for bid in range(batch_size):
                sentid = batch.sent_index[bid]
                for i in range(batch.word_num[bid]):
                    wordid = batch.word_ids[bid][i]

                    # upos
                    pred_upos_id = predicted_upos[bid][i]
                    upos_name = self._config.itos[self._config.active_lang][UPOS][pred_upos_id]
                    test_set.conllu_doc[sentid][wordid][UPOS] = upos_name
                    # xpos
                    pred_xpos_id = predicted_xpos[bid][i]
                    xpos_name = self._config.itos[self._config.active_lang][XPOS][pred_xpos_id]
                    test_set.conllu_doc[sentid][wordid][XPOS] = xpos_name
                    # feats
                    pred_feats_id = predicted_feats[bid][i]
                    feats_name = self._config.itos[self._config.active_lang][FEATS][pred_feats_id]
                    test_set.conllu_doc[sentid][wordid][FEATS] = feats_name

                    # head
                    test_set.conllu_doc[sentid][wordid][HEAD] = int(pred_tokens[bid][i][0])
                    # deprel
                    test_set.conllu_doc[sentid][wordid][DEPREL] = pred_tokens[bid][i][1]
        tagged_doc = get_output_doc(dposdep_doc, test_set.conllu_doc)
        torch.cuda.empty_cache()
        return tagged_doc

    def lemmatize(self, input, is_sent=False):
        if is_sent:
            assert is_string(input) or is_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of non-empty strings.'

            if is_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=' '.join(input))

                input = [{ID: k + 1, TEXT: w} for k, w in enumerate(input)]
                return {TOKENS: self._lemmatize_sent(in_sent=input, obmit_tag=True), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                ori_text = deepcopy(input)
                return {TEXT: ori_text, TOKENS: self._lemmatize_sent(in_sent=input, obmit_tag=True), LANG: self.active_lang}

        else:
            assert is_string(input) or is_list_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of lists of non-empty strings.'

            if is_list_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text='\n'.join([' '.join(sent) for sent in input]))

                input = [{ID: sid + 1, TOKENS: [{ID: tid + 1, TEXT: w} for tid, w in enumerate(sent)]} for sid, sent in
                         enumerate(input)]
                return {SENTENCES: self._lemmatize_doc(in_doc=input, obmit_tag=True), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                ori_text = deepcopy(input)
                return {TEXT: ori_text, SENTENCES: self._lemmatize_doc(in_doc=input, obmit_tag=True), LANG: self.active_lang}

    def _lemmatize_sent(self, in_sent, obmit_tag=False):
        if type(in_sent) == str:
            in_sent = self._tokenize_sent(in_sent)
            in_sent = self._posdep_sent(in_sent)
        dlemmatize_sent = deepcopy(in_sent)
        lemmatized_sent = \
            self._lemma_model[self._config.active_lang].predict([{ID: 1, TOKENS: dlemmatize_sent}], obmit_tag)[0][
                TOKENS]
        return lemmatized_sent

    def _lemmatize_doc(self, in_doc, obmit_tag=False):  # assuming input is a document
        if type(in_doc) == str:  # in_doc is a raw string in this case
            in_doc = self._tokenize_doc(in_doc)
            in_doc = self._posdep_doc(in_doc)

        dlemmatize_doc = deepcopy(in_doc)
        lemmatized_doc = self._lemma_model[self._config.active_lang].predict(dlemmatize_doc, obmit_tag)
        return lemmatized_doc

    def _mwt_expand(self, tokenized_doc):
        expanded_doc = self._mwt_model[self._config.active_lang].predict(tokenized_doc)
        return expanded_doc

    def ner(self, input, is_sent=False):
        if is_sent:
            assert is_string(input) or is_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of non-empty strings.'

            if is_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=' '.join(input))

                assert self.active_lang in langwithner, 'NER module is not available for "{}"'.format(self.active_lang)

                input = [{ID: k + 1, TEXT: w} for k, w in enumerate(input)]
                return {TOKENS: self._ner_sent(in_sent=input), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                assert self.active_lang in langwithner, 'NER module is not available for "{}"'.format(self.active_lang)

                ori_text = deepcopy(input)
                return {TEXT: ori_text, TOKENS: self._ner_sent(in_sent=input), LANG: self.active_lang}

        else:
            assert is_string(input) or is_list_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of lists of non-empty strings.'

            if is_list_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text='\n'.join([' '.join(sent) for sent in input]))

                assert self.active_lang in langwithner, 'NER module is not available for "{}"'.format(self.active_lang)

                input = [{ID: sid + 1, TOKENS: [{ID: tid + 1, TEXT: w} for tid, w in enumerate(sent)]} for sid, sent in
                         enumerate(input)]
                return {SENTENCES: self._ner_doc(in_doc=input), LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                assert self.active_lang in langwithner, 'NER module is not available for "{}"'.format(self.active_lang)

                ori_text = deepcopy(input)
                return {TEXT: ori_text, SENTENCES: self._ner_doc(in_doc=input), LANG: self.active_lang}

    def _ner_sent(self, in_sent):  # assuming input is a document
        if type(in_sent) == str:
            in_sent = self._tokenize_sent(in_sent)

        dner_doc = [{ID: 1, TOKENS: deepcopy(in_sent)}]
        sentences = [[t[TEXT] for t in sentence[TOKENS]] for sentence in dner_doc]
        test_set = NERDatasetLive(
            config=self._config,
            tokenized_sentences=sentences
        )
        test_set.numberize()
        # load ner adapter weights
        self._load_adapter_weights(model_name='ner')
        eval_batch_size = tbname2tagbatchsize.get(self._config.treebank_name, self._tagbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 3)

        for batch in DataLoader(test_set,
                                batch_size=eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn):
            word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
            pred_entity_labels = self._ner_model[self._config.active_lang].predict(batch, word_reprs)

            batch_size = len(batch.word_num)

            for bid in range(batch_size):
                sentid = batch.sent_index[bid]
                for i in range(batch.word_num[bid]):
                    wordid = batch.word_ids[bid][i]

                    # NER tag
                    dner_doc[sentid][TOKENS][wordid][NER] = pred_entity_labels[bid][i]

        torch.cuda.empty_cache()
        return dner_doc[0][TOKENS]

    def _ner_doc(self, in_doc):  # assuming input is a document
        if type(in_doc) == str:
            in_doc = self._tokenize_doc(in_doc)
        dner_doc = deepcopy(in_doc)
        sentences = [[t[TEXT] for t in sentence[TOKENS]] for sentence in dner_doc]
        test_set = NERDatasetLive(
            config=self._config,
            tokenized_sentences=sentences
        )
        test_set.numberize()
        # load ner adapter weights
        self._load_adapter_weights(model_name='ner')
        eval_batch_size = tbname2tagbatchsize.get(self._config.treebank_name, self._tagbatchsize)
        if self._config.embedding_name == 'xlm-roberta-large':
            eval_batch_size = int(eval_batch_size / 3)

        for batch in DataLoader(test_set,
                                batch_size=eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn):
            word_reprs, cls_reprs = self._embedding_layers.get_tagger_inputs(batch)
            pred_entity_labels = self._ner_model[self._config.active_lang].predict(batch, word_reprs)

            batch_size = len(batch.word_num)

            for bid in range(batch_size):
                sentid = batch.sent_index[bid]
                for i in range(batch.word_num[bid]):
                    wordid = batch.word_ids[bid][i]

                    # NER tag
                    dner_doc[sentid][TOKENS][wordid][NER] = pred_entity_labels[bid][i]

        torch.cuda.empty_cache()
        return dner_doc

    def __call__(self, input, is_sent=False):
        if is_sent:
            assert is_string(input) or is_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of non-empty strings.'

            if is_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=' '.join(input))

                tokenized_sent = [{ID: k + 1, TEXT: w} for k, w in enumerate(input)]
                tagged_sent = self._posdep_sent(tokenized_sent)
                out = self._lemmatize_sent(tagged_sent)
                if self._config.active_lang in langwithner:  # ner if possible
                    out = self._ner_sent(out)
                final = {TOKENS: out, LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                ori_text = deepcopy(input)
                tagged_sent = self._posdep_sent(input)
                out = self._lemmatize_sent(tagged_sent)
                if self._config.active_lang in langwithner:  # ner if possible
                    out = self._ner_sent(out)
                final = {TEXT: ori_text, TOKENS: out, LANG: self.active_lang}
        else:
            assert is_string(input) or is_list_list_strings(
                input), 'Input must be one of the following:\n(i) A non-empty string.\n(ii) A list of lists of non-empty strings.'

            if is_list_list_strings(input):
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text='\n'.join([' '.join(sent) for sent in input]))

                input = [{ID: sid + 1, TOKENS: [{ID: tid + 1, TEXT: w} for tid, w in enumerate(sent)]} for sid, sent in
                         enumerate(input)]
                tagged_doc = self._posdep_doc(input)
                out = self._lemmatize_doc(tagged_doc)
                if self._config.active_lang in langwithner:  # ner if possible
                    out = self._ner_doc(out)
                final = {SENTENCES: out, LANG: self.active_lang}
            else:
                # switch to detected lang if auto mode is on
                if self.auto_mode:
                    self._detect_lang_and_switch(text=input)

                ori_text = deepcopy(input)
                tagged_doc = self._posdep_doc(in_doc=input)
                out = self._lemmatize_doc(tagged_doc)
                if self._config.active_lang in langwithner:  # ner if possible
                    out = self._ner_doc(out)
                final = {TEXT: ori_text, SENTENCES: out, LANG: self.active_lang}
        return final

    def _conllu_predict(self, text_fpath):
        print('Running the pipeline on device={}'.format(self._config.device))
        with open(text_fpath) as f:
            raw_text = f.read()
        print('Beginning tokenization')
        tokenized_doc = self._tokenize_doc(raw_text)
        print('Beginning pos tagging and dependency parsing')
        tagged_doc = self._posdep_doc(tokenized_doc)
        print('Beginning lemmatization')
        lemmatized_doc = self._lemmatize_doc(tagged_doc)
        conllu_doc = []
        for sentence in lemmatized_doc:
            conllu_sentence = []
            for token in sentence[TOKENS]:
                if type(token[ID]) == int or len(token[ID]) == 1:
                    conllu_sentence.append(token)
                else:
                    conllu_sentence.append(token)
                    for word in token[EXPANDED]:
                        conllu_sentence.append(word)
            conllu_doc.append(conllu_sentence)

        pred_lemma_fpath = text_fpath + '.pred'
        CoNLL.dict2conll(conllu_doc, pred_lemma_fpath)
        return pred_lemma_fpath
