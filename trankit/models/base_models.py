from ..adapter_transformers import AdapterType, XLMRobertaModel
from ..adapter_transformers import AdapterConfig
from ..utils.base_utils import *


class Base_Model(nn.Module):  # currently assuming the pretrained transformer is XLM-Roberta
    def __init__(self, config, task_name):
        super().__init__()
        self.config = config
        self.task_name = task_name
        # xlmr encoder
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.xlmr = XLMRobertaModel.from_pretrained(config.embedding_name,
                                                    cache_dir=os.path.join(config._cache_dir, config.embedding_name),
                                                    output_hidden_states=True)
        self.xlmr_dropout = nn.Dropout(p=config.embedding_dropout)
        # add task adapters
        task_config = AdapterConfig.load("pfeiffer",
                                         reduction_factor=6 if config.embedding_name == 'xlm-roberta-base' else 4)
        self.xlmr.add_adapter(task_name, AdapterType.text_task, config=task_config)
        self.xlmr.train_adapter([task_name])
        self.xlmr.set_active_adapters([task_name])

    def encode(self, piece_idxs, attention_masks):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]

        wordpiece_reprs = xlmr_outputs[:, 1:-1, :]  # [batch size, max input length - 2, xlmr dim]
        wordpiece_reprs = self.xlmr_dropout(wordpiece_reprs)
        return wordpiece_reprs

    def encode_words(self, piece_idxs, attention_masks, word_lens):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]
        cls_reprs = xlmr_outputs[:, 0, :].unsqueeze(1)  # [batch size, 1, xlmr dim]

        # average all pieces for multi-piece words
        idxs, masks, token_num, token_len = word_lens_to_idxs_fast(word_lens)
        idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.xlmr_dim) + 1
        masks = xlmr_outputs.new(masks).unsqueeze(-1)
        xlmr_outputs = torch.gather(xlmr_outputs, 1,
                                    idxs) * masks
        xlmr_outputs = xlmr_outputs.view(batch_size, token_num, token_len, self.xlmr_dim)
        xlmr_outputs = xlmr_outputs.sum(2)
        return xlmr_outputs, cls_reprs

    def forward(self, batch):
        raise NotImplementedError


class Multilingual_Embedding(Base_Model):
    def __init__(self, config, model_name='embedding'):
        super(Multilingual_Embedding, self).__init__(config, task_name=model_name)

    def get_tokenizer_inputs(self, batch):
        wordpiece_reprs = self.encode(
            piece_idxs=batch.piece_idxs,
            attention_masks=batch.attention_masks
        )
        return wordpiece_reprs

    def get_tagger_inputs(self, batch):
        # encoding
        word_reprs, cls_reprs = self.encode_words(
            piece_idxs=batch.piece_idxs,
            attention_masks=batch.attention_masks,
            word_lens=batch.word_lens
        )
        return word_reprs, cls_reprs


class Deep_Biaffine(nn.Module):
    '''
    implemented based on the paper https://arxiv.org/abs/1611.01734
    '''

    def __init__(self, in_dim1, in_dim2, hidden_dim, output_dim):
        super().__init__()
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.ffn1 = nn.Sequential(
            nn.Linear(in_dim1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(in_dim2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # pairwise interactions
        self.pairwise_weight = nn.Parameter(torch.Tensor(in_dim1 + 1, in_dim2 + 1, output_dim))
        self.pairwise_weight.data.zero_()

    def forward(self, x1, x2):
        h1 = self.ffn1(x1)
        h2 = self.ffn2(x2)
        # make interactions
        g1 = torch.cat([h1, h1.new_ones(*h1.size()[:-1], 1)], len(h1.size()) - 1)
        g2 = torch.cat([h2, h2.new_ones(*h2.size()[:-1], 1)], len(h2.size()) - 1)

        g1_size = g1.size()
        g2_size = g2.size()

        g1_w = torch.mm(g1.view(-1, g1_size[-1]), self.pairwise_weight.view(-1, (self.in_dim2 + 1) * self.output_dim))
        g2 = g2.transpose(1, 2)
        g1_w_g2 = g1_w.view(g1_size[0], g1_size[1] * self.output_dim, g2_size[2]).bmm(g2)
        g1_w_g2 = g1_w_g2.view(g1_size[0], g1_size[1], self.output_dim, g2_size[1]).transpose(2, 3)
        return g1_w_g2
