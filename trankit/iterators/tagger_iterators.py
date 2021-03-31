from . import *

# for sents
instance_fields = [
    'sent_index',
    'words', 'word_num', 'word_ids', 'word_span_idxs',
    'piece_idxs', 'attention_masks', 'word_lens',
    'edit_type_idxs', 'upos_type_idxs', 'xpos_type_idxs', 'feats_type_idxs',
    'head_idxs', 'deprel_idxs', 'word_mask'
]

batch_fields = [
    'sent_index',
    'words', 'word_num', 'word_ids', 'word_span_idxs',
    'piece_idxs', 'attention_masks', 'word_lens',
    'edit_type_idxs', 'upos_type_idxs', 'xpos_type_idxs', 'feats_type_idxs',
    'upos_ids', 'xpos_ids', 'feats_ids',
    'head_idxs', 'deprel_idxs', 'word_mask'
]

Instance = namedtuple('Instance', field_names=instance_fields)

Batch = namedtuple('Batch', field_names=batch_fields)


class TaggerDatasetLive(Dataset):
    def __init__(self, tokenized_doc, wordpiece_splitter, config):
        self.wordpiece_splitter = wordpiece_splitter
        self.config = config
        self.max_input_length = 512

        self.treebank_name = config.treebank_name

        self.conllu_doc = []
        self.tokenized_doc = tokenized_doc

        language = treebank2lang[self.treebank_name]
        self.vocabs_fpath = os.path.join(self.config._cache_dir, self.config.embedding_name, language,
                                         '{}.vocabs.json'.format(language))

        self.vocabs = {}
        self.data = []
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        with open(self.vocabs_fpath) as f:
            self.vocabs = json.load(f)

        self.data, self.conllu_doc = get_examples_from_conllu(
            self.wordpiece_splitter,
            self.max_input_length,
            self.tokenized_doc
        )
        # split long sentences into 512-length chunks
        new_data = []
        for inst in self.data:
            words = inst['words']
            pieces = [[p for p in self.wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
            for ps in pieces:
                if len(ps) == 0:
                    ps += ['-']
            flat_pieces = [p for ps in pieces for p in ps]
            if len(flat_pieces) > self.max_input_length - 2:
                sub_insts = []
                cur_inst = deepcopy(inst)
                for key in ['words', 'word_ids', LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, 'flat_pieces']:
                    cur_inst[key] = []

                for i in range(len(inst['words'])):
                    for key in ['words', 'word_ids', LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL]:
                        cur_inst[key].append(inst[key][i])
                    cur_inst['flat_pieces'].extend(pieces[i])
                    if len(cur_inst['flat_pieces']) >= self.max_input_length - 10:
                        sub_insts.append(cur_inst)

                        cur_inst = deepcopy(inst)
                        for key in ['words', 'word_ids', LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, 'flat_pieces']:
                            cur_inst[key] = []

                if len(cur_inst['flat_pieces']) > 0:
                    sub_insts.append(cur_inst)

                # all sub instances share the same sent_index,
                # 'word_ids' is used for later filling predictions into the right place
                new_data.extend(sub_insts)
            else:
                new_data.append(inst)
        self.data = new_data
        
    def numberize(self):
        wordpiece_splitter = self.wordpiece_splitter
        data = []
        for inst in self.data:
            words = inst['words']
            pieces = [[p for p in wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
            for ps in pieces:
                if len(ps) == 0:
                    ps += ['-']
            word_lens = [len(x) for x in pieces]
            assert 0 not in word_lens
            flat_pieces = [p for ps in pieces for p in ps]
            assert len(flat_pieces) > 0

            word_span_idxs = []
            start = 1
            for l in word_lens:
                word_span_idxs.append([start, start + l])
                start += l

            # Pad word pieces with special tokens
            piece_idxs = wordpiece_splitter.encode(
                flat_pieces,
                add_special_tokens=True,
                max_length=self.max_input_length,
                truncation=True
            )
            assert len(piece_idxs) <= self.max_input_length

            attn_masks = [1] * len(piece_idxs)
            piece_idxs = piece_idxs
            assert len(piece_idxs) > 0

            edit_type_idxs = [self.vocabs[LEMMA][edit] for edit in inst[LEMMA]]
            upos_type_idxs = [self.vocabs[UPOS][upos] for upos in inst[UPOS]]
            xpos_type_idxs = [self.vocabs[XPOS][xpos] for xpos in inst[XPOS]]
            feats_type_idxs = [self.vocabs[FEATS][feats] for feats in inst[FEATS]]

            assert len(edit_type_idxs) == len(inst['words'])

            # head, deprel, word_mask
            head_idxs = [head for head in inst[HEAD]]
            deprel_idxs = [self.vocabs[DEPREL][deprel] for deprel in inst[DEPREL]]
            word_mask = [0] * (len(inst['words']) + 1)

            instance = Instance(
                sent_index=inst['sent_index'],
                word_ids=inst['word_ids'],
                words=inst['words'],
                word_num=len(inst['words']),
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                word_span_idxs=word_span_idxs,
                word_lens=word_lens,
                edit_type_idxs=edit_type_idxs,
                upos_type_idxs=upos_type_idxs,
                xpos_type_idxs=xpos_type_idxs,
                feats_type_idxs=feats_type_idxs,
                head_idxs=head_idxs,
                deprel_idxs=deprel_idxs,
                word_mask=word_mask
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_sent_index = [inst.sent_index for inst in batch]

        batch_words = [inst.words for inst in batch]
        batch_word_num = [inst.word_num for inst in batch]
        batch_word_ids = [inst.word_ids for inst in batch]

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_word_lens = []
        batch_word_span_idxs = []

        batch_edit_type_idxs = []
        batch_upos_type_idxs = []
        batch_xpos_type_idxs = []
        batch_feats_type_idxs = []

        batch_upos_ids, batch_xpos_ids, batch_feats_ids = [], [], []

        batch_head_ids, batch_deprel_ids, batch_word_mask = [], [], []

        max_word_num = max(batch_word_num)
        max_wordpiece_num = max([len(inst.piece_idxs) for inst in batch])

        for inst in batch:
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_masks + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_word_lens.append(inst.word_lens)
            batch_word_span_idxs.append(inst.word_span_idxs + [[1, 2]] * (max_word_num - inst.word_num))

            # lemmatization
            batch_edit_type_idxs.extend(inst.edit_type_idxs +
                                        [-100] * (max_word_num - inst.word_num))
            # upos, xpos, feats
            batch_upos_type_idxs.extend(inst.upos_type_idxs +
                                        [-100] * (max_word_num - inst.word_num))
            batch_xpos_type_idxs.extend(inst.xpos_type_idxs +
                                        [-100] * (max_word_num - inst.word_num))
            batch_feats_type_idxs.extend(inst.feats_type_idxs +
                                         [-100] * (max_word_num - inst.word_num))
            # head, deprel
            batch_head_ids.append(inst.head_idxs + [0] * (max_word_num - inst.word_num))
            batch_deprel_ids.append(inst.deprel_idxs + [0] * (max_word_num - inst.word_num))
            batch_word_mask.append(inst.word_mask + [1] * (max_word_num - inst.word_num))
            # ids for feature building
            batch_upos_ids.append(inst.upos_type_idxs + [0] * (max_word_num - inst.word_num))
            batch_xpos_ids.append(inst.xpos_type_idxs + [0] * (max_word_num - inst.word_num))
            batch_feats_ids.append(inst.feats_type_idxs + [0] * (max_word_num - inst.word_num))

        batch_piece_idxs = torch.tensor(batch_piece_idxs, dtype=torch.long, device=self.config.device)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.float16, device=self.config.device)
        batch_edit_type_idxs = torch.tensor(batch_edit_type_idxs, dtype=torch.long, device=self.config.device)
        batch_word_span_idxs = torch.tensor(batch_word_span_idxs, dtype=torch.long, device=self.config.device)

        batch_upos_type_idxs = torch.tensor(batch_upos_type_idxs, dtype=torch.long, device=self.config.device)
        batch_xpos_type_idxs = torch.tensor(batch_xpos_type_idxs, dtype=torch.long, device=self.config.device)
        batch_feats_type_idxs = torch.tensor(batch_feats_type_idxs, dtype=torch.long, device=self.config.device)

        batch_upos_ids = torch.tensor(batch_upos_ids, dtype=torch.long, device=self.config.device)
        batch_xpos_ids = torch.tensor(batch_xpos_ids, dtype=torch.long, device=self.config.device)

        batch_head_ids = torch.tensor(batch_head_ids, dtype=torch.long, device=self.config.device)
        batch_deprel_ids = torch.tensor(batch_deprel_ids, dtype=torch.long, device=self.config.device)
        batch_word_mask = torch.tensor(batch_word_mask, dtype=torch.bool, device=self.config.device)

        return Batch(
            sent_index=batch_sent_index,
            word_ids=batch_word_ids,
            words=batch_words,
            word_num=batch_word_num,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            word_lens=batch_word_lens,
            word_span_idxs=batch_word_span_idxs,
            edit_type_idxs=batch_edit_type_idxs,
            upos_type_idxs=batch_upos_type_idxs,
            xpos_type_idxs=batch_xpos_type_idxs,
            feats_type_idxs=batch_feats_type_idxs,
            upos_ids=batch_upos_ids,
            xpos_ids=batch_xpos_ids,
            feats_ids=batch_xpos_ids,
            head_idxs=batch_head_ids,
            deprel_idxs=batch_deprel_ids,
            word_mask=batch_word_mask
        )


class TaggerDataset(Dataset):
    def __init__(self, config, input_conllu, gold_conllu, evaluate=False):
        self.config = config
        self.input_conllu = input_conllu
        self.gold_conllu = gold_conllu
        self.evaluate = evaluate

        self.treebank_name = config.treebank_name

        self.conllu_doc = []

        self.vocabs_fpath = os.path.join(self.config._save_dir, '{}.vocabs.json'.format(self.config.lang))
        self.vocabs = {}
        self.data = []
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        if not self.evaluate:
            self.vocabs, self.data, self.conllu_doc = tget_examples_from_conllu(
                self.config.wordpiece_splitter,
                self.config.max_input_length,
                self.gold_conllu, get_vocab=True)
            with open(self.vocabs_fpath, 'w') as f:
                json.dump(self.vocabs, f)
        else:
            with open(self.vocabs_fpath) as f:
                self.vocabs = json.load(f)
            self.data, self.conllu_doc = tget_examples_from_conllu(
                self.config.wordpiece_splitter,
                self.config.max_input_length,
                self.input_conllu
            )

        print('Loaded {} entries from {}'.format(len(self), self.input_conllu))

    def numberize(self):
        wordpiece_splitter = self.config.wordpiece_splitter
        data = []
        for inst in self.data:
            words = inst['words']
            pieces = [[p for p in wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
            for ps in pieces:
                if len(ps) == 0:
                    ps += ['-']
            word_lens = [len(x) for x in pieces]
            assert 0 not in word_lens
            flat_pieces = [p for ps in pieces for p in ps]
            assert len(flat_pieces) > 0

            word_span_idxs = []
            start = 1
            for l in word_lens:
                word_span_idxs.append([start, start + l])
                start += l

            # Pad word pieces with special tokens
            piece_idxs = wordpiece_splitter.encode(
                flat_pieces,
                add_special_tokens=True,
                max_length=self.config.max_input_length,
                truncation=True
            )
            assert len(piece_idxs) <= self.config.max_input_length

            attn_masks = [1] * len(piece_idxs)
            piece_idxs = piece_idxs
            assert len(piece_idxs) > 0

            edit_type_idxs = [self.vocabs[LEMMA][edit] for edit in inst[LEMMA]]
            upos_type_idxs = [self.vocabs[UPOS][upos] for upos in inst[UPOS]]
            xpos_type_idxs = [self.vocabs[XPOS][xpos] for xpos in inst[XPOS]]
            feats_type_idxs = [self.vocabs[FEATS][feats] for feats in inst[FEATS]]

            assert len(edit_type_idxs) == len(inst['words'])

            # head, deprel, word_mask
            head_idxs = [head for head in inst[HEAD]]
            deprel_idxs = [self.vocabs[DEPREL][deprel] for deprel in inst[DEPREL]]
            word_mask = [0] * (len(inst['words']) + 1)

            instance = Instance(
                sent_index=inst['sent_index'],
                word_ids=inst['word_ids'],
                words=inst['words'],
                word_num=len(inst['words']),
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                word_span_idxs=word_span_idxs,
                word_lens=word_lens,
                edit_type_idxs=edit_type_idxs,
                upos_type_idxs=upos_type_idxs,
                xpos_type_idxs=xpos_type_idxs,
                feats_type_idxs=feats_type_idxs,
                head_idxs=head_idxs,
                deprel_idxs=deprel_idxs,
                word_mask=word_mask
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_sent_index = [inst.sent_index for inst in batch]

        batch_words = [inst.words for inst in batch]
        batch_word_num = [inst.word_num for inst in batch]
        batch_word_ids = [inst.word_ids for inst in batch]

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_word_lens = []
        batch_word_span_idxs = []

        batch_edit_type_idxs = []
        batch_upos_type_idxs = []
        batch_xpos_type_idxs = []
        batch_feats_type_idxs = []

        batch_upos_ids, batch_xpos_ids, batch_feats_ids = [], [], []

        batch_head_ids, batch_deprel_ids, batch_word_mask = [], [], []

        max_word_num = max(batch_word_num)
        max_wordpiece_num = max([len(inst.piece_idxs) for inst in batch])

        for inst in batch:
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_masks + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_word_lens.append(inst.word_lens)
            batch_word_span_idxs.append(inst.word_span_idxs + [[1, 2]] * (max_word_num - inst.word_num))

            # lemmatization
            batch_edit_type_idxs.extend(inst.edit_type_idxs +
                                        [-100] * (max_word_num - inst.word_num))
            # upos, xpos, feats
            batch_upos_type_idxs.extend(inst.upos_type_idxs +
                                        [-100] * (max_word_num - inst.word_num))
            batch_xpos_type_idxs.extend(inst.xpos_type_idxs +
                                        [-100] * (max_word_num - inst.word_num))
            batch_feats_type_idxs.extend(inst.feats_type_idxs +
                                         [-100] * (max_word_num - inst.word_num))
            # head, deprel
            batch_head_ids.append(inst.head_idxs + [0] * (max_word_num - inst.word_num))
            batch_deprel_ids.append(inst.deprel_idxs + [0] * (max_word_num - inst.word_num))
            batch_word_mask.append(inst.word_mask + [1] * (max_word_num - inst.word_num))
            # ids for feature building
            batch_upos_ids.append(inst.upos_type_idxs + [0] * (max_word_num - inst.word_num))
            batch_xpos_ids.append(inst.xpos_type_idxs + [0] * (max_word_num - inst.word_num))
            batch_feats_ids.append(inst.feats_type_idxs + [0] * (max_word_num - inst.word_num))

        batch_piece_idxs = torch.tensor(batch_piece_idxs, dtype=torch.long, device=self.config.device)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.float16, device=self.config.device)
        batch_edit_type_idxs = torch.tensor(batch_edit_type_idxs, dtype=torch.long, device=self.config.device)
        batch_word_span_idxs = torch.tensor(batch_word_span_idxs, dtype=torch.long, device=self.config.device)

        batch_upos_type_idxs = torch.tensor(batch_upos_type_idxs, dtype=torch.long, device=self.config.device)
        batch_xpos_type_idxs = torch.tensor(batch_xpos_type_idxs, dtype=torch.long, device=self.config.device)
        batch_feats_type_idxs = torch.tensor(batch_feats_type_idxs, dtype=torch.long, device=self.config.device)

        batch_upos_ids = torch.tensor(batch_upos_ids, dtype=torch.long, device=self.config.device)
        batch_xpos_ids = torch.tensor(batch_xpos_ids, dtype=torch.long, device=self.config.device)

        batch_head_ids = torch.tensor(batch_head_ids, dtype=torch.long, device=self.config.device)
        batch_deprel_ids = torch.tensor(batch_deprel_ids, dtype=torch.long, device=self.config.device)
        batch_word_mask = torch.tensor(batch_word_mask, dtype=torch.bool, device=self.config.device)

        return Batch(
            sent_index=batch_sent_index,
            word_ids=batch_word_ids,
            words=batch_words,
            word_num=batch_word_num,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            word_lens=batch_word_lens,
            word_span_idxs=batch_word_span_idxs,
            edit_type_idxs=batch_edit_type_idxs,
            upos_type_idxs=batch_upos_type_idxs,
            xpos_type_idxs=batch_xpos_type_idxs,
            feats_type_idxs=batch_feats_type_idxs,
            upos_ids=batch_upos_ids,
            xpos_ids=batch_xpos_ids,
            feats_ids=batch_xpos_ids,
            head_idxs=batch_head_ids,
            deprel_idxs=batch_deprel_ids,
            word_mask=batch_word_mask
        )
