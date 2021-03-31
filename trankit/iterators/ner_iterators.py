from . import *


instance_fields = [
    'sent_index', 'word_ids',
    'words', 'word_num',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]

batch_fields = [
    'sent_index', 'word_ids',
    'words', 'word_num', 'word_mask',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]

Instance = namedtuple('Instance', field_names=instance_fields)

Batch = namedtuple('Batch', field_names=batch_fields)

train_instance_fields = [
    'words', 'word_num',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]

train_batch_fields = [
    'words', 'word_num', 'word_mask',
    'piece_idxs', 'attention_masks', 'word_lens',
    'entity_label_idxs'
]

Train_Instance = namedtuple('Train_Instance', field_names=train_instance_fields)

Train_Batch = namedtuple('Train_Batch', field_names=train_batch_fields)


class NERDatasetLive(Dataset):
    def __init__(self, config, tokenized_sentences):
        self.config = config
        self.wordpiece_splitter = config.wordpiece_splitter
        self.max_input_length = 512
        # load data
        self.data = [{'sent_index': sid, 'words': sentence, 'word_ids': list(range(len(sentence)))} for sid, sentence in
                     enumerate(tokenized_sentences)]

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
                for key in ['words', 'word_ids', 'flat_pieces']:
                    cur_inst[key] = []

                for i in range(len(inst['words'])):
                    for key in ['words', 'word_ids']:
                        cur_inst[key].append(inst[key][i])
                    cur_inst['flat_pieces'].extend(pieces[i])
                    if len(cur_inst['flat_pieces']) >= self.max_input_length - 10:
                        sub_insts.append(cur_inst)

                        cur_inst = deepcopy(inst)
                        for key in ['words', 'word_ids', 'flat_pieces']:
                            cur_inst[key] = []

                if len(cur_inst['flat_pieces']) > 0:
                    sub_insts.append(cur_inst)

                # all sub instances share the same sent_index,
                # 'word_ids' is used for later filling predictions into the right place
                new_data.extend(sub_insts)
            else:
                new_data.append(inst)
        self.data = new_data

        # load vocab
        self.vocabs = self.config.ner_vocabs[self.config.active_lang]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def numberize(self):
        data = []
        for inst in self.data:
            words = inst['words']
            # lowercase
            if self.config.lowercase:
                words = [w.lower() for w in words]
            # ---------------------
            pieces = [[p for p in self.wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
            for ps in pieces:
                if len(ps) == 0:
                    ps += ['-']
            word_lens = [len(x) for x in pieces]
            flat_pieces = [p for ps in pieces for p in ps]
            # Pad word pieces with special tokens
            piece_idxs = self.wordpiece_splitter.encode(
                flat_pieces,
                add_special_tokens=True,
                max_length=self.max_input_length,
                truncation=True
            )

            attn_masks = [1] * len(piece_idxs)
            piece_idxs = piece_idxs

            instance = Instance(
                sent_index=inst['sent_index'],
                word_ids=inst['word_ids'],
                words=inst['words'],
                word_num=len(inst['words']),
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                word_lens=word_lens,
                entity_label_idxs=[0 for _ in inst['words']]
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_sent_index = [inst.sent_index for inst in batch]
        batch_word_ids = [inst.word_ids for inst in batch]

        batch_words = [inst.words for inst in batch]
        batch_word_num = [inst.word_num for inst in batch]

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_word_lens = []

        max_word_num = max(batch_word_num)
        max_wordpiece_num = max([len(inst.piece_idxs) for inst in batch])
        batch_word_mask = []
        batch_entity_label_idxs = []

        for inst in batch:
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_masks + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_word_lens.append(inst.word_lens)
            batch_word_mask.append([1] * inst.word_num + [0] * (max_word_num - inst.word_num))
            batch_entity_label_idxs.append(inst.entity_label_idxs +
                                           [0] * (max_word_num - inst.word_num))

        batch_piece_idxs = torch.LongTensor(batch_piece_idxs).to(self.config.device)
        batch_attention_masks = torch.FloatTensor(batch_attention_masks).to(self.config.device)
        batch_word_num = torch.LongTensor(batch_word_num).to(self.config.device)
        batch_word_mask = torch.LongTensor(batch_word_mask).eq(0).to(self.config.device)
        batch_entity_label_idxs = torch.LongTensor(batch_entity_label_idxs).to(self.config.device)

        return Batch(
            sent_index=batch_sent_index,
            word_ids=batch_word_ids,
            words=batch_words,
            word_num=batch_word_num,
            word_mask=batch_word_mask,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            word_lens=batch_word_lens,
            entity_label_idxs=batch_entity_label_idxs
        )


class NERDataset(Dataset):
    def __init__(self, config, bio_fpath, evaluate=False):
        self.config = config
        self.evaluate = evaluate

        # load data
        self.config.vocab_fpath = os.path.join(self.config._save_dir, '{}.ner-vocab.json'.format(self.config.lang))
        self.data = get_examples_from_bio_fpath(self.config, bio_fpath, evaluate)

        with open(self.config.vocab_fpath) as f:
            self.vocabs = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def numberize(self):
        data = []
        skip = 0
        for inst in self.data:
            words = inst['words']
            pieces = [[p for p in self.config.wordpiece_splitter.tokenize(w) if p != '▁'] for w in words]
            for ps in pieces:
                if len(ps) == 0:
                    ps += ['-']
            word_lens = [len(x) for x in pieces]
            assert 0 not in word_lens
            flat_pieces = [p for ps in pieces for p in ps]
            assert len(flat_pieces) > 0

            if len(flat_pieces) > self.config.max_input_length - 2:
                skip += 1
                continue

            # Pad word pieces with special tokens
            piece_idxs = self.config.wordpiece_splitter.encode(
                flat_pieces,
                add_special_tokens=True,
                max_length=self.config.max_input_length,
                truncation=True
            )

            attn_masks = [1] * len(piece_idxs)
            piece_idxs = piece_idxs
            assert len(piece_idxs) > 0

            entity_label_idxs = [self.vocabs[label] for label in inst['entity-labels']]

            instance = Train_Instance(
                words=inst['words'],
                word_num=len(inst['words']),
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                word_lens=word_lens,
                entity_label_idxs=entity_label_idxs
            )
            data.append(instance)
        print('Skipped {} over-length examples'.format(skip))
        print('Loaded {} examples'.format(len(data)))
        self.data = data

    def collate_fn(self, batch):
        batch_words = [inst.words for inst in batch]
        batch_word_num = [inst.word_num for inst in batch]

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_word_lens = []
        batch_entity_label_idxs = []

        max_word_num = max(batch_word_num)
        max_wordpiece_num = max([len(inst.piece_idxs) for inst in batch])
        batch_word_mask = []

        for inst in batch:
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_masks + [0] * (max_wordpiece_num - len(inst.piece_idxs)))
            batch_word_lens.append(inst.word_lens)

            batch_entity_label_idxs.append(inst.entity_label_idxs +
                                           [0] * (max_word_num - inst.word_num))
            batch_word_mask.append([1] * inst.word_num + [0] * (max_word_num - inst.word_num))

        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(
            batch_attention_masks)
        batch_entity_label_idxs = torch.cuda.LongTensor(batch_entity_label_idxs)
        batch_word_num = torch.cuda.LongTensor(batch_word_num)
        batch_word_mask = torch.cuda.LongTensor(batch_word_mask).eq(0)

        return Train_Batch(
            words=batch_words,
            word_num=batch_word_num,
            word_mask=batch_word_mask,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            word_lens=batch_word_lens,
            entity_label_idxs=batch_entity_label_idxs
        )
