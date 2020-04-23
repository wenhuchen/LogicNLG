import torch.optim as optim
from torch import nn
import torch
from torch import autograd
import torch.nn.functional as F
import math
import numpy as np
from transformers import BertModel


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

    def step_forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_query = dec_input[:, -1, :].unsqueeze(1)
        slf_attn_mask = slf_attn_mask[:, -1, :].unsqueeze(1)
        dec_enc_attn_mask = dec_enc_attn_mask[:, -1, :].unsqueeze(1)
        non_pad_mask = non_pad_mask[:, -1, :].unsqueeze(1)

        dec_output, dec_slf_attn = self.slf_attn(
            dec_query, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1).type(torch.bool)
    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device), diagonal=1).type(torch.bool)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class BERTGen(nn.Module):

    def __init__(self, vocab_size, dim, layers, head, modelpath):
        super(BERTGen, self).__init__()
        self.encoder = BertModel.from_pretrained(modelpath)
        self.model = TableDecoder(vocab_size, dim, layers, dim, head)

    def forward(self, trg_inp, caption):
        src_feat = self.encoder(caption)[0]
        tgt_feat = self.encoder(trg_inp)[0]

        src_feat = src_feat.repeat(tgt_feat.shape[0], 1, 1)
        logits = self.model(trg_inp, src_feat, tgt_feat)
        return logits

    def encode(self, caption):
        return self.encoder(caption)[0]

    def decode(self, trg_inp, src_feat, tgt_feat):
        return self.model(trg_inp, src_feat, tgt_feat)


class TableDecoder(nn.Module):

    def __init__(self, vocab_size, d_word_vec, n_layers, d_model, n_head, dropout=0.1, copy=False, with_bert=True):
        super(TableDecoder, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head
        d_inner = d_model * 4
        self.vocab_size = vocab_size

        self.dec_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tgt_seq, src_feat, tgt_feat):
        src_length = src_feat.shape[1]
        tgt_length = tgt_seq.shape[1]

        slf_mask = torch.zeros_like(tgt_seq).type(torch.bool).to(tgt_seq.device)
        slf_attn_mask = torch.zeros_like(tgt_seq).unsqueeze(2).repeat(
            1, 1, tgt_length).type(torch.bool).to(tgt_seq.device)
        non_pad_mask = (1 - slf_mask.float()).unsqueeze(-1)
        dec_enc_attn_mask = slf_mask.unsqueeze(2).repeat(1, 1, src_length)

        dec_output = tgt_feat
        for layer in self.dec_stack:
            dec_output, _, _ = layer(dec_output, src_feat,
                                     non_pad_mask=non_pad_mask,
                                     slf_attn_mask=slf_attn_mask,
                                     dec_enc_attn_mask=dec_enc_attn_mask)

        logits = self.tgt_word_prj(dec_output)
        return logits


class TableInfusing(nn.Module):
    def __init__(self, vocab_size, full_vocab_size, d_word_vec, n_layers, n_head, dropout=0.1):
        super(TableInfusing, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_word_vec, padding_idx=0)

        self.vocab_size = vocab_size
        self.full_vocab_size = full_vocab_size

        self.field_encoder = nn.LSTM(d_word_vec, d_word_vec)
        d_inner = 4 * d_word_vec
        d_k, d_v = d_word_vec // n_head, d_word_vec // n_head
        self.discount = 0.99

        self.enc_stack = nn.ModuleList([
            EncoderLayer(d_word_vec, d_inner, n_head, d_k, d_v)
            for _ in range(n_layers)])

        self.dec_stack = nn.ModuleList([
            DecoderLayer(d_word_vec, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.post_word_emb = PositionalEmbedding(d_model=d_word_vec)

        self.copy_gate = nn.Sequential(nn.Linear(d_word_vec, 1), nn.Sigmoid())

        self.tgt_word_prj = nn.Linear(d_word_vec, vocab_size, bias=False)

    def forward(self, seqs_in, table_in, table_scatters, lookups, line_nos, fields, indexes):
        enc_inp = self.encode(table_in, lookups, line_nos, fields, indexes)
        logits = self.decode(seqs_in, enc_inp, table_scatters)

        return logits

    def encode(self, table_in, lookups, line_nos, fields, indexes):
        field_emb = self.embed(fields).transpose(1, 0)

        out, hidden = self.field_encoder(field_emb)
        out = out.transpose(1, 0)

        field_mask = (fields != 0).unsqueeze(-1).float()
        out = out * field_mask

        extracted = torch.gather(out, 1, indexes[:, :, None].repeat(1, 1, out.shape[-1]))

        field_emb = torch.gather(extracted, 1, lookups[:, :, None].repeat(1, 1, extracted.shape[-1]))

        line_no_emb = self.embed(line_nos)

        word_emb = self.embed(table_in)

        cell_emb = field_emb + line_no_emb + word_emb

        src_slf_mask = (table_in == 0)

        src_src_mask = src_slf_mask.unsqueeze(1).expand(-1, src_slf_mask.shape[1], -1)
        src_non_pad_mask = (1 - src_slf_mask.float()).unsqueeze(-1)

        enc_inp = cell_emb
        for layer in self.enc_stack:
            enc_inp, _ = layer(enc_inp, src_non_pad_mask, src_src_mask)
            enc_inp *= src_non_pad_mask

        return enc_inp

    def decode(self, seqs_in, enc_inp, table_scatters):
        batch_size, length = seqs_in.shape[0], seqs_in.shape[1]

        tgt_emb = self.embed(seqs_in)
        dec_inp = tgt_emb + self.post_word_emb(seqs_in)

        src_slf_mask = (table_scatters == 0)
        tgt_slf_mask = (seqs_in == 0)

        non_pad_mask = get_non_pad_mask(seqs_in)

        slf_attn_mask_subseq = get_subsequent_mask(seqs_in)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=seqs_in, seq_q=seqs_in)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = src_slf_mask.unsqueeze(1).expand(batch_size, length, -1).type(torch.bool)

        for layer in self.dec_stack:
            dec_inp, _, _ = layer(dec_inp, enc_inp,
                                  non_pad_mask=non_pad_mask,
                                  slf_attn_mask=slf_attn_mask,
                                  dec_enc_attn_mask=dec_enc_attn_mask)

        gate = self.copy_gate(dec_inp)

        scores = torch.bmm(dec_inp, enc_inp.transpose(2, 1))
        oov_vocab_prob = torch.softmax(scores, -1)

        in_vocab_prob = torch.softmax(self.tgt_word_prj(dec_inp), -1)

        size = self.full_vocab_size - self.vocab_size
        add_on_prob = (1 - self.discount) / size
        add_on = torch.FloatTensor(batch_size, length, size).fill_(add_on_prob).to(in_vocab_prob.device)

        full_prob = torch.cat([in_vocab_prob * (1 - gate) * self.discount, add_on], -1)

        full_prob = full_prob.scatter_add(2, table_scatters.unsqueeze(1).repeat(1, length, 1), oov_vocab_prob * gate)
        full_logits = torch.log(full_prob)

        return full_logits


class Ranker(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, vocab_size, d_word_vec, n_layers, d_model, n_head, dropout=0.1):
        super(Ranker, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head
        d_inner = d_model * 4

        self.word_emb = nn.Embedding(vocab_size, d_word_vec, padding_idx=0)

        self.post_word_emb = PositionalEmbedding(d_model=d_word_vec)

        self.enc_stack = nn.ModuleList(
            [EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)])

        self.dec_stack = nn.ModuleList(
            [DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, 2, bias=True)

    def forward(self, prog, sent):
        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(sent)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=sent, seq_q=sent)
        # -- Forward Word Embedding
        enc_output = self.word_emb(sent) + self.post_word_emb(sent)

        for enc_layer in self.enc_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        non_pad_mask = get_non_pad_mask(prog)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=prog, seq_q=prog)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=sent, seq_q=prog)
        # -- Forward
        dec_output = self.word_emb(prog) + self.post_word_emb(prog)

        for dec_layer in self.dec_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

        logits = self.tgt_word_prj(dec_output[:, 0])
        return logits

    def prob(self, prog, sent):
        logits = self.forward(prog, sent)
        prob = torch.softmax(logits, -1)
        return prob[:, 1]


class BERTRanker(nn.Module):
    def __init__(self, model_class, model_name_or_path, config, cache_dir='/tmp/'):
        super(BERTRanker, self).__init__()
        self.base = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.proj = nn.Linear(768, 2)

    def forward(self, input_tokens, input_types, input_masks):
        inputs = {"input_ids": input_tokens, "token_type_ids": input_types, "attention_mask": input_masks}
        _, text_representation = self.base(**inputs)
        logits = self.proj(text_representation)
        return logits

    def prob(self, input_tokens, input_types, input_masks):
        inputs = {"input_ids": input_tokens, "token_type_ids": input_types, "attention_mask": input_masks}
        _, text_representation = self.base(**inputs)
        logits = self.proj(text_representation)
        prob = torch.softmax(logits, -1)
        return prob[:, 1]
