'''
Adapted from https://github.com/stanfordnlp/stanza/blob/master/stanza/models/common/seq2seq_model.py
Date: 2021/01/06
'''
import torch.nn.functional as F
from trankit.utils.mwt_lemma_utils.seq2seq_utils import *


class BasicAttention(nn.Module):
    """
    A basic MLP attention layer.
    """

    def __init__(self, dim):
        super(BasicAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_c = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        target = self.linear_in(input)  # batch x dim
        source = self.linear_c(context.contiguous().view(-1, dim)).view(batch_size, source_len, dim)
        attn = target.unsqueeze(1).expand_as(context) + source
        attn = self.tanh(attn)  # batch x sourceL x dim
        attn = self.linear_v(attn.view(-1, dim)).view(batch_size, source_len)

        if mask is not None:
            attn.masked_fill_(mask, -INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        weighted_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LinearAttention(nn.Module):
    """ A linear attention form, inspired by BiDAF:
        a = W (u; v; u o v)
    """

    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.linear = nn.Linear(dim * 3, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)  # batch*sourceL x dim
        v = context.contiguous().view(-1, dim)
        attn_in = torch.cat((u, v, u.mul(v)), 1)
        attn = self.linear(attn_in).view(batch_size, source_len)

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        attn3 = attn.view(batch_size, 1, source_len)  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class DeepAttention(nn.Module):
    """ A deep attention form, invented by Robert:
        u = ReLU(Wx)
        v = ReLU(Wy)
        a = V.(u o v)
    """

    def __init__(self, dim):
        super(DeepAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)  # batch*sourceL x dim
        u = self.relu(self.linear_in(u))
        v = self.relu(self.linear_in(context.contiguous().view(-1, dim)))
        attn = self.linear_v(u.mul(v)).view(batch_size, source_len)

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        attn3 = attn.view(batch_size, 1, source_len)  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class LSTMAttention(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='soft'):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        if attn_type == 'soft':
            self.attention_layer = SoftDotAttention(hidden_size)
        elif attn_type == 'mlp':
            self.attention_layer = BasicAttention(hidden_size)
        elif attn_type == 'linear':
            self.attention_layer = LinearAttention(hidden_size)
        elif attn_type == 'deep':
            self.attention_layer = DeepAttention(hidden_size)
        else:
            raise Exception("Unsupported LSTM attention type: {}".format(attn_type))

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.lstm_cell(input[i], hidden)
            hy, cy = hidden
            h_tilde, alpha = self.attention_layer(hy, ctx, mask=ctx_mask)
            output.append(h_tilde)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class Beam:
    """
     Adapted and modified from the OpenNMT project.

     Class for managing the internals of the beam search process.


             hyp1-hyp1---hyp1 -hyp1
                     \             /
             hyp2 \-hyp2 /-hyp2hyp2
                                   /      \
             hyp3-hyp3---hyp3 -hyp3
             ========================

     Takes care of beams, back pointers, and scores.
    """

    def __init__(self, size, cuda=False):

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(PAD_ID)]
        self.nextYs[0][0] = SOS_ID

        # The copy indices for each time
        self.copy = []

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, copy_indices=None):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `copy_indices` - copy indices (K x ctx_len)

        Returns: True if beam search is complete.
        """
        if self.done:
            return True
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
        else:
            # first step, expand from the first position
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        # bestScoreId is the integer ids, and numWords is the integer length.
        # Need to do integer division
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords)
        if copy_indices is not None:
            self.copy.append(copy_indices.index_select(0, prevK))

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == EOS_ID:
            self.done = True
            self.allScores.append(self.scores)

        return self.done

    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        "Get the score of the best in the beam."
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    def get_hyp(self, k):
        """
        Walk back to construct the full hypothesis.

        Parameters:

             * `k` - the position in the beam to construct.

         Returns: The hypothesis
        """
        hyp = []
        cpy = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            if len(self.copy) > 0:
                cpy.append(self.copy[j][k])
            k = self.prevKs[j][k]

        hyp = hyp[::-1]
        cpy = cpy[::-1]
        # postprocess: if cpy index is not -1, use cpy index instead of hyp word
        for i, cidx in enumerate(cpy):
            if cidx >= 0:
                hyp[i] = -(cidx + 1)  # make index 1-based and flip it for token generation

        return hyp


class Seq2SeqModel(nn.Module):
    """
    A complete encoder-decoder model, with optional attention.
    """

    def __init__(self, args, emb_matrix=None, use_cuda=False, training_mode=False):
        super().__init__()
        self.vocab_size = args['vocab_size']
        self.emb_dim = args['emb_dim']
        self.hidden_dim = args['hidden_dim']
        self.nlayers = args['num_layers']  # encoder layers, decoder layers = 1
        self.emb_dropout = args.get('emb_dropout', 0.0)
        self.dropout = args['dropout']
        self.pad_token = PAD_ID
        self.max_dec_len = args['max_dec_len']
        self.use_cuda = use_cuda
        self.training_mode = training_mode
        self.top = args.get('top', 1e10)
        self.args = args
        self.emb_matrix = emb_matrix

        self.num_directions = 2
        self.enc_hidden_dim = self.hidden_dim // 2
        self.dec_hidden_dim = self.hidden_dim

        self.use_pos = args.get('pos', False)
        self.pos_dim = args.get('pos_dim', 0)
        self.pos_vocab_size = args.get('pos_vocab_size', 0)
        self.pos_dropout = args.get('pos_dropout', 0)
        self.edit = args.get('edit', False)
        self.num_edit = args.get('num_edit', 0)

        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.drop = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_token)
        self.encoder = nn.LSTM(self.emb_dim, self.enc_hidden_dim, self.nlayers, \
                               bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)
        self.decoder = LSTMAttention(self.emb_dim, self.dec_hidden_dim, \
                                     batch_first=True, attn_type=self.args['attn_type'])
        self.dec2vocab = nn.Linear(self.dec_hidden_dim, self.vocab_size)
        if self.use_pos and self.pos_dim > 0:
            self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.pos_dim, self.pad_token)
            self.pos_drop = nn.Dropout(self.pos_dropout)
        if self.edit:
            edit_hidden = self.hidden_dim // 2
            self.edit_clf = nn.Sequential(
                nn.Linear(self.hidden_dim, edit_hidden),
                nn.ReLU(),
                nn.Linear(edit_hidden, self.num_edit))

        self.SOS_tensor = torch.LongTensor([SOS_ID])
        self.SOS_tensor = self.SOS_tensor.cuda() if self.use_cuda else self.SOS_tensor

        self.init_weights()

    def init_weights(self):
        # initialize embeddings
        init_range = EMB_INIT_RANGE
        if self.emb_matrix is not None:
            if isinstance(self.emb_matrix, np.ndarray):
                self.emb_matrix = torch.from_numpy(self.emb_matrix)
            assert self.emb_matrix.size() == (self.vocab_size, self.emb_dim), \
                "Input embedding matrix must match size: {} x {}".format(self.vocab_size, self.emb_dim)
            self.embedding.weight.data.copy_(self.emb_matrix)
        else:
            self.embedding.weight.data.uniform_(-init_range, init_range)

        # initialize pos embeddings
        if self.use_pos:
            self.pos_embedding.weight.data.uniform_(-init_range, init_range)

    def cuda(self):
        super().cuda()
        self.use_cuda = True

    def cpu(self):
        super().cpu()
        self.use_cuda = False

    def zero_state(self, inputs):
        batch_size = inputs.size(0)
        h0 = torch.zeros(self.encoder.num_layers * 2, batch_size, self.enc_hidden_dim, requires_grad=False)
        c0 = torch.zeros(self.encoder.num_layers * 2, batch_size, self.enc_hidden_dim, requires_grad=False)
        if self.use_cuda:
            if self.training_mode:
                return h0.cuda(), c0.cuda()
            else:
                return h0.cuda().half(), c0.cuda().half()
        return h0, c0

    def encode(self, enc_inputs, lens):
        """ Encode source sequence. """
        self.h0, self.c0 = self.zero_state(enc_inputs)

        packed_inputs = nn.utils.rnn.pack_padded_sequence(enc_inputs, lens, batch_first=True)
        packed_h_in, (hn, cn) = self.encoder(packed_inputs, (self.h0, self.c0))
        h_in, _ = nn.utils.rnn.pad_packed_sequence(packed_h_in, batch_first=True)
        hn = torch.cat((hn[-1], hn[-2]), 1)
        cn = torch.cat((cn[-1], cn[-2]), 1)
        return h_in, (hn, cn)

    def decode(self, dec_inputs, hn, cn, ctx, ctx_mask=None):
        """ Decode a step, based on context encoding and source context states."""
        dec_hidden = (hn, cn)
        h_out, dec_hidden = self.decoder(dec_inputs, dec_hidden, ctx, ctx_mask)

        h_out_reshape = h_out.contiguous().view(h_out.size(0) * h_out.size(1), -1)
        decoder_logits = self.dec2vocab(h_out_reshape)
        decoder_logits = decoder_logits.view(h_out.size(0), h_out.size(1), -1)
        log_probs = self.get_log_prob(decoder_logits)
        return log_probs, dec_hidden

    def forward(self, src, src_mask, tgt_in, pos=None):
        # prepare for encoder/decoder
        batch_size = src.size(0)
        enc_inputs = self.emb_drop(self.embedding(src))
        dec_inputs = self.emb_drop(self.embedding(tgt_in))
        if self.use_pos:
            assert pos is not None, "Missing POS input for seq2seq lemmatizer."
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(0).long().sum(1))

        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)

        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None

        log_probs, _ = self.decode(dec_inputs, hn, cn, h_in, src_mask)
        return log_probs, edit_logits

    def get_log_prob(self, logits):
        logits_reshape = logits.view(-1, self.vocab_size)
        log_probs = F.log_softmax(logits_reshape, dim=1)
        if logits.dim() == 2:
            return log_probs
        return log_probs.view(logits.size(0), logits.size(1), logits.size(2))

    def predict_greedy(self, src, src_mask, pos=None):
        """ Predict with greedy decoding. """
        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        if self.use_pos:
            assert pos is not None, "Missing POS input for seq2seq lemmatizer."
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(PAD_ID).long().sum(1))

        # encode source
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)

        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None

        # greedy decode by step
        dec_inputs = self.embedding(self.SOS_tensor)
        dec_inputs = dec_inputs.expand(batch_size, dec_inputs.size(0), dec_inputs.size(1))

        done = [False for _ in range(batch_size)]
        total_done = 0
        max_len = 0
        output_seqs = [[] for _ in range(batch_size)]

        while total_done < batch_size and max_len < self.max_dec_len:
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, src_mask)
            assert log_probs.size(1) == 1, "Output must have 1-step of output."
            _, preds = log_probs.squeeze(1).max(1, keepdim=True)
            dec_inputs = self.embedding(preds)  # update decoder inputs
            max_len += 1
            for i in range(batch_size):
                if not done[i]:
                    token = preds.data[i][0].item()
                    if token == EOS_ID:
                        done[i] = True
                        total_done += 1
                    else:
                        output_seqs[i].append(token)
        return output_seqs, edit_logits

    def predict(self, src, src_mask, pos=None, beam_size=5):
        """ Predict with beam search. """
        if beam_size == 1:
            return self.predict_greedy(src, src_mask, pos=pos)

        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        if self.use_pos:
            assert pos is not None, "Missing POS input for seq2seq lemmatizer."
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(PAD_ID).long().sum(1))

        # (1) encode source
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)

        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None

        # (2) set up beam
        with torch.no_grad():
            h_in = h_in.data.repeat(beam_size, 1, 1)  # repeat data for beam search
            src_mask = src_mask.repeat(beam_size, 1)
            # repeat decoder hidden states
            hn = hn.data.repeat(beam_size, 1)
            cn = cn.data.repeat(beam_size, 1)
        beam = [Beam(beam_size, self.use_cuda) for _ in range(batch_size)]

        def update_state(states, idx, positions, beam_size):
            """ Select the states according to back pointers. """
            for e in states:
                br, d = e.size()
                s = e.contiguous().view(beam_size, br // beam_size, d)[:, idx]
                s.data.copy_(s.data.index_select(0, positions))

        # (3) main loop
        for i in range(self.max_dec_len):
            dec_inputs = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(-1, 1)
            dec_inputs = self.embedding(dec_inputs)
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, src_mask)
            log_probs = log_probs.view(beam_size, batch_size, -1).transpose(0, 1) \
                .contiguous()  # [batch, beam, V]

            # advance each beam
            done = []
            for b in range(batch_size):
                is_done = beam[b].advance(log_probs.data[b])
                if is_done:
                    done += [b]
                # update beam state
                update_state((hn, cn), b, beam[b].get_current_origin(), beam_size)

            if len(done) == batch_size:
                break

        # back trace and find hypothesis
        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[0]]
            k = ks[0]
            hyp = beam[b].get_hyp(k)
            hyp = prune_hyp(hyp)
            hyp = [i.item() for i in hyp]
            all_hyp += [hyp]

        return all_hyp, edit_logits
