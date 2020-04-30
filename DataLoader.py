import os
import numpy as np
import json
import torch
import random
import pandas
import re
import copy
from torch.utils.data import DataLoader, Dataset

class Dataloader(object):
    def __init__(self, train_name, val_name, test_name):
        if train_name:
            with open(train_name) as f:
                self.train = json.load(f)
            self.train_ids = list(self.train.keys())

        if val_name:
            with open(val_name) as f:
                self.val = json.load(f)
            self.val_ids = list(self.val.keys())

        if test_name:
            with open(test_name) as f:
                self.test = json.load(f)
            self.test_ids = list(self.test.keys())

    def get_item(self, i, option='train'):
        if option == 'train':
            return self.train_ids[i]
        elif option == 'val':
            return self.val_ids[i]
        else:
            return self.test_ids[i]

    def get_reference(self, idx, option='test'):
        assert option == 'test'
        table_id = self.test_ids[idx]
        entry = self.test[table_id]
        return [_[0].lower().split(' ') for _ in entry]

    def get_table_id(self, idx, option='test'):
        assert option == 'test'
        table_id = self.test_ids[idx]
        return table_id

    def shuffle(self):
        random.shuffle(self.train_ids)

    def train_len(self):
        return len(self.train)

    def val_len(self):
        return len(self.val)

    def test_len(self):
        return len(self.test)

    def obtain_idx(self, idx, option):
        if option == 'train':
            table_id = self.train_ids[idx]
            entry = self.train[table_id]
        elif option == 'val':
            table_id = self.val_ids[idx]
            entry = self.val[table_id]
        elif option == 'test':
            table_id = self.test_ids[idx]
            entry = self.test[table_id]
        else:
            raise NotImplementedError
        return table_id, entry


class GPTTableDatabase(Dataloader):
    def __init__(self, train_name, val_name, test_name, tokenizer, batch_size=5, max_len=800):
        super(GPTTableDatabase, self).__init__(train_name, val_name, test_name)
        self.tokenizer = tokenizer
        # self.feat_prefix = 'table_feats/'
        self.batch_size = batch_size
        self.max_len = max_len

    def get_data(self, idx, option='train', details=False):
        table_id, entry = self.obtain_idx(idx, option)
        d = pandas.read_csv('data/all_csv/' + table_id, '#')

        columns = d.columns
        seqs = []
        descs = []

        if option in ['train']:
            random.shuffle(entry)
            entry = entry[:self.batch_size]

        for e in entry:
            seqs.append(self.tokenizer.encode(e[0]))
            tmp = ""
            for i in range(len(d)):
                tmp += 'In row {} , '.format(i + 1)
                for _ in e[1]:
                    if isinstance(d.iloc[i][columns[_]], str):
                        entity = map(lambda x: x.capitalize(), d.iloc[i][columns[_]].split(' '))
                        entity = ' '.join(entity)
                    else:
                        entity = str(d.iloc[i][columns[_]])

                    tmp += 'the {} is {} , '.format(columns[_], entity)
                tmp = tmp[:-3] + ' . '

            tmp_idx = self.tokenizer.tokenize(tmp)
            if len(tmp_idx) > self.max_len:
                tmp_idx = tmp_idx[:self.max_len]

            tmp_prefix = self.tokenizer.tokenize('Given the table title of "{}" . '.format(e[2]))
            tmp_suffix = self.tokenizer.tokenize('Start describing : ')

            descs.append(self.tokenizer.convert_tokens_to_ids(tmp_prefix + tmp_idx + tmp_suffix))

        length = max([len(_) for _ in seqs]) + 1

        for i in range(len(seqs)):
            seqs[i] += (length - len(seqs[i])) * [self.tokenizer.eos_token_id]
        seqs = torch.LongTensor(seqs)

        length = max([len(_) for _ in descs]) + 1
        for i in range(len(descs)):
            descs[i] = (length - len(descs[i])) * [self.tokenizer.eos_token_id] + descs[i]
        descs = torch.LongTensor(descs)

        # caption_mask = torch.FloatTensor(caption_ids.shape[0], caption_ids.shape[1]).zero_()
        seq_mask = (seqs != self.tokenizer.eos_token_id)[:, :-1].float()
        seq_mask = torch.cat([torch.FloatTensor(seq_mask.shape[0], 1).fill_(1.), seq_mask], 1)

        inputs = seqs[:, :-1]
        outputs = seqs

        if details:
            return inputs, outputs, seq_mask, descs, d, [_[1] for _ in entry], [_[2] for _ in entry]
        else:
            return inputs, outputs, seq_mask, descs

    def get_pair_data(self, idx, option='val'):
        table_id, entry = self.obtain_idx(idx, option)
        d = pandas.read_csv('data/all_csv/' + table_id, '#')
        columns = d.columns

        pairs = []
        for opt in ['pos', 'neg']:
            seqs = []
            descs = []
            for e in entry:
                if opt not in e and opt == 'pos':
                    opt = 'unknown1'
                if opt not in e and opt == 'neg':
                    opt = 'unknown2'
                    
                e = e[opt]
                seqs.append(self.tokenizer.encode(e[0]))
                tmp = ""
                for i in range(len(d)):
                    tmp += 'In row {} , '.format(i + 1)
                    for _ in e[1]:
                        if isinstance(d.iloc[i][columns[_]], str):
                            entity = map(lambda x: x.capitalize(), d.iloc[i][columns[_]].split(' '))
                            entity = ' '.join(entity)
                        else:
                            entity = str(d.iloc[i][columns[_]])

                        tmp += 'the {} is {} , '.format(columns[_], entity)
                    tmp = tmp[:-3] + ' . '

                tmp_idx = self.tokenizer.tokenize(tmp)
                if len(tmp_idx) > self.max_len:
                    tmp_idx = tmp_idx[:self.max_len]

                tmp_prefix = self.tokenizer.tokenize('Given the table title of "{}" . '.format(e[2]))
                tmp_suffix = self.tokenizer.tokenize('Start describing : ')

                descs.append(self.tokenizer.convert_tokens_to_ids(tmp_prefix + tmp_idx + tmp_suffix))

            length = max([len(_) for _ in seqs]) + 1

            for i in range(len(seqs)):
                seqs[i] += (length - len(seqs[i])) * [self.tokenizer.eos_token_id]
            seqs = torch.LongTensor(seqs)

            length = max([len(_) for _ in descs]) + 1
            for i in range(len(descs)):
                descs[i] = (length - len(descs[i])) * [self.tokenizer.eos_token_id] + descs[i]
            descs = torch.LongTensor(descs)

            # caption_mask = torch.FloatTensor(caption_ids.shape[0], caption_ids.shape[1]).zero_()
            seq_mask = (seqs != self.tokenizer.eos_token_id)[:, :-1].float()
            seq_mask = torch.cat([torch.FloatTensor(seq_mask.shape[0], 1).fill_(1.), seq_mask], 1)

            inputs = seqs[:, :-1]
            outputs = seqs        

            pairs.append((inputs, outputs, seq_mask, descs))

        return pairs

class GPTTableCoarseFineDatabase(Dataloader):
    def __init__(self, train_name, val_name, test_name, tokenizer, batch_size=5, max_len=800, stage=1, total_stage=2):
        super(GPTTableCoarseFineDatabase, self).__init__(train_name, val_name, test_name)
        self.tokenizer = tokenizer
        self.stage = stage
        self.batch_size = batch_size
        self.max_len = max_len

    def get_data(self, idx, option='train', details=False):
        table_id, entry = self.obtain_idx(idx, option)
        d = pandas.read_csv('data/all_csv/' + table_id, '#')

        columns = d.columns
        seqs = []
        templates = []
        descs = []
        seq_masks = []

        if option in ['train']:
            random.shuffle(entry)
            entry = entry[:self.batch_size]

        for e in entry:
            if self.stage == 1:
                seqs.append(self.tokenizer.encode(e[3], add_special_tokens=False))
                seq_masks.append([1] * len(seqs[-1]))
            elif self.stage == 2:
                part1 = self.tokenizer.encode(e[3] + ' [SEP] ', add_special_tokens=False)
                part2 = self.tokenizer.encode(e[0], add_special_tokens=False)
                seqs.append(part1 + part2)
                seq_masks.append([1] * len(part1) + [1] * len(part2))
            else:
                raise NotImplementedError

            tmp = ""
            for i in range(len(d)):
                tmp += 'In row {} , '.format(i + 1)
                for _ in e[1]:
                    if isinstance(d.iloc[i][columns[_]], str):
                        entity = map(lambda x: x.capitalize(), d.iloc[i][columns[_]].split(' '))
                        entity = ' '.join(entity)
                    else:
                        entity = str(d.iloc[i][columns[_]])

                    tmp += 'the {} is {} , '.format(columns[_], entity)
                tmp = tmp[:-3] + ' . '

            tmp_idx = self.tokenizer.tokenize(tmp)
            if len(tmp_idx) > self.max_len:
                tmp_idx = tmp_idx[:self.max_len]

            tmp_prefix = self.tokenizer.tokenize('Given the table title of "{}" . '.format(e[2]))

            descs.append(self.tokenizer.convert_tokens_to_ids(tmp_prefix + tmp_idx))

        length = max([len(_) for _ in seqs]) + 1

        for i in range(len(seqs)):
            seqs[i] += (length - len(seqs[i])) * [self.tokenizer.eos_token_id]
            seq_masks[i] = seq_masks[i] + [1] + (length - len(seq_masks[i]) - 1) * [0] 
        seqs = torch.LongTensor(seqs)
        seq_masks = torch.FloatTensor(seq_masks)

        length = max([len(_) for _ in descs]) + 1
        for i in range(len(descs)):
            descs[i] = (length - len(descs[i])) * [self.tokenizer.eos_token_id] + descs[i]
        descs = torch.LongTensor(descs)

        inputs = seqs[:, :-1]
        outputs = seqs

        if details:
            return inputs, outputs, seq_masks, descs, d, [_[1] for _ in entry], [_[2] for _ in entry]
        else:
            return inputs, outputs, seq_masks, descs

    def get_pair_data(self, idx, option='val', mask_type='both'):
        table_id, entry = self.obtain_idx(idx, option)
        d = pandas.read_csv('data/all_csv/' + table_id, '#')
        columns = d.columns

        pairs = []
        for opt in ['pos', 'neg']:
            seqs = []
            descs = []
            seq_masks = []
            for e in entry:
                if opt not in e and opt == 'pos':
                    opt = 'unknown1'
                if opt not in e and opt == 'neg':
                    opt = 'unknown2'

                e = e[opt]
                if self.stage == 1:
                    seqs.append(self.tokenizer.encode(e[3], add_special_tokens=False))
                    seq_masks.append([1] * len(seqs[-1]))
                else:
                    part1 = self.tokenizer.encode(e[3] + ' [SEP] ', add_special_tokens=False)
                    part2 = self.tokenizer.encode(e[0], add_special_tokens=False)
                    seqs.append(part1 + part2)
                    if mask_type == 'both':
                        seq_masks.append([1] * len(part1) + [1] * len(part2))
                    elif mask_type == 'single':
                        seq_masks.append([0] * len(part1) + [1] * len(part2))                    
                    else:
                        raise NotImplementedError
                
                tmp = ""
                for i in range(len(d)):
                    tmp += 'In row {} , '.format(i + 1)
                    for _ in e[1]:
                        if isinstance(d.iloc[i][columns[_]], str):
                            entity = map(lambda x: x.capitalize(), d.iloc[i][columns[_]].split(' '))
                            entity = ' '.join(entity)
                        else:
                            entity = str(d.iloc[i][columns[_]])

                        tmp += 'the {} is {} , '.format(columns[_], entity)
                    tmp = tmp[:-3] + ' . '

                tmp_idx = self.tokenizer.tokenize('Given the table title of "{}" . {}'.format(e[2], tmp))
                if len(tmp_idx) > self.max_len:
                    tmp_idx = tmp_idx[:self.max_len]            
                #tmp_suffix = self.tokenizer.tokenize('Start describing : ')

                descs.append(self.tokenizer.convert_tokens_to_ids(tmp_idx))

            length = max([len(_) for _ in seqs]) + 1
            for i in range(len(seqs)):
                seqs[i] += (length - len(seqs[i])) * [self.tokenizer.eos_token_id]
                seq_masks[i] = seq_masks[i] + [1] + (length - len(seq_masks[i]) - 1) * [0] 
            seqs = torch.LongTensor(seqs)
            seq_masks = torch.FloatTensor(seq_masks)

            length = max([len(_) for _ in descs]) + 1
            for i in range(len(descs)):
                descs[i] = (length - len(descs[i])) * [self.tokenizer.eos_token_id] + descs[i]
            descs = torch.LongTensor(descs)

            inputs = seqs[:, :-1]
            outputs = seqs

            pairs.append((inputs, outputs, seq_masks, descs))
        
        return pairs

class GPTTableCoarseFineDatabase2(Dataloader):
    def __init__(self, train_name, val_name, test_name, tokenizer, batch_size=5, max_len=800, stage=1, total_stage=2):
        super(GPTTableCoarseFineDatabase2, self).__init__(None, val_name, test_name)
        if train_name:
            with open(train_name, 'r') as f:
                self.train = json.load(f)
        
        self.tokenizer = tokenizer
        self.stage = stage
        self.batch_size = batch_size
        self.max_len = max_len
    
    def train_len(self):
        return int(len(self.train) // self.batch_size) 

    def get_train_data(self, idx):
        idx = random.choice(range(0, len(self.train)))

        window = 15#int(self.batch_size / 2)
        start_idx = max(0, idx - window)
        end_idx = min(idx + window, len(self.train))

        entries = copy.copy(self.train[start_idx: end_idx])
        random.shuffle(entries)
        entries = entries[:self.batch_size]

        seqs = []
        descs = []
        seq_masks = []
        for e in entries:
            if self.stage == 1:
                seqs.append(self.tokenizer.encode(e[3], add_special_tokens=False))
                seq_masks.append([1] * len(seqs[-1]))
            elif self.stage == 2:
                part1 = self.tokenizer.encode(e[3] + ' [SEP] ', add_special_tokens=False)
                part2 = self.tokenizer.encode(e[0], add_special_tokens=False)
                seqs.append(part1 + part2)
                seq_masks.append([1] * len(part1) + [1] * len(part2))
            else:
                raise NotImplementedError

            tmp = e[-1]

            tmp_idx = self.tokenizer.tokenize(tmp)
            if len(tmp_idx) > self.max_len:
                tmp_idx = tmp_idx[:self.max_len]

            tmp_prefix = self.tokenizer.tokenize('Given the table title of "{}" . '.format(e[2]))

            descs.append(self.tokenizer.convert_tokens_to_ids(tmp_prefix + tmp_idx))

        length = max([len(_) for _ in seqs]) + 1

        for i in range(len(seqs)):
            seqs[i] += (length - len(seqs[i])) * [self.tokenizer.eos_token_id]
            seq_masks[i] = seq_masks[i] + [1] + (length - len(seq_masks[i]) - 1) * [0] 
        seqs = torch.LongTensor(seqs)
        seq_masks = torch.FloatTensor(seq_masks)

        length = max([len(_) for _ in descs]) + 1
        for i in range(len(descs)):
            descs[i] = (length - len(descs[i])) * [self.tokenizer.eos_token_id] + descs[i]
        descs = torch.LongTensor(descs)

        inputs = seqs[:, :-1]
        outputs = seqs

        return inputs, outputs, seq_masks, descs

    def get_data(self, idx, option, details=False):
        table_id, entry = self.obtain_idx(idx, option)
        d = pandas.read_csv('data/all_csv/' + table_id, '#')

        columns = d.columns
        seqs = []
        descs = []
        seq_masks = []

        for e in entry:
            if self.stage == 1:
                seqs.append(self.tokenizer.encode(e[3], add_special_tokens=False))
                seq_masks.append([1] * len(seqs[-1]))
            elif self.stage == 2:
                part1 = self.tokenizer.encode(e[3] + ' [SEP] ', add_special_tokens=False)
                part2 = self.tokenizer.encode(e[0], add_special_tokens=False)
                seqs.append(part1 + part2)
                seq_masks.append([1] * len(part1) + [1] * len(part2))
            else:
                raise NotImplementedError

            tmp = ""
            for i in range(len(d)):
                tmp += 'In row {} , '.format(i + 1)
                for _ in e[1]:
                    if isinstance(d.iloc[i][columns[_]], str):
                        entity = map(lambda x: x.capitalize(), d.iloc[i][columns[_]].split(' '))
                        entity = ' '.join(entity)
                    else:
                        entity = str(d.iloc[i][columns[_]])

                    tmp += 'the {} is {} , '.format(columns[_], entity)
                tmp = tmp[:-3] + ' . '

            tmp_idx = self.tokenizer.tokenize(tmp)
            if len(tmp_idx) > self.max_len:
                tmp_idx = tmp_idx[:self.max_len]

            tmp_prefix = self.tokenizer.tokenize('Given the table title of "{}" . '.format(e[2]))

            descs.append(self.tokenizer.convert_tokens_to_ids(tmp_prefix + tmp_idx))

        length = max([len(_) for _ in seqs]) + 1

        for i in range(len(seqs)):
            seqs[i] += (length - len(seqs[i])) * [self.tokenizer.eos_token_id]
            seq_masks[i] = seq_masks[i] + [1] + (length - len(seq_masks[i]) - 1) * [0] 
        seqs = torch.LongTensor(seqs)
        seq_masks = torch.FloatTensor(seq_masks)

        length = max([len(_) for _ in descs]) + 1
        for i in range(len(descs)):
            descs[i] = (length - len(descs[i])) * [self.tokenizer.eos_token_id] + descs[i]
        descs = torch.LongTensor(descs)

        inputs = seqs[:, :-1]
        outputs = seqs

        if details:
            return inputs, outputs, seq_masks, descs, d, [_[1] for _ in entry], [_[2] for _ in entry]
        else:
            return inputs, outputs, seq_masks, descs

    def get_pair_data(self, idx, option='val', mask_type='both'):
        table_id, entry = self.obtain_idx(idx, option)
        d = pandas.read_csv('data/all_csv/' + table_id, '#')
        columns = d.columns

        pairs = []
        for opt in ['pos', 'neg']:
            seqs = []
            descs = []
            seq_masks = []
            for e in entry:
                if opt not in e and opt == 'pos':
                    opt = 'unknown1'
                if opt not in e and opt == 'neg':
                    opt = 'unknown2'

                e = e[opt]
                if self.stage == 1:
                    seqs.append(self.tokenizer.encode(e[3], add_special_tokens=False))
                    seq_masks.append([1] * len(seqs[-1]))
                else:
                    part1 = self.tokenizer.encode(e[3] + ' [SEP] ', add_special_tokens=False)
                    part2 = self.tokenizer.encode(e[0], add_special_tokens=False)
                    seqs.append(part1 + part2)
                    if mask_type == 'both':
                        seq_masks.append([1] * len(part1) + [1] * len(part2))
                    elif mask_type == 'single':
                        seq_masks.append([0] * len(part1) + [1] * len(part2))                    
                    else:
                        raise NotImplementedError
                
                tmp = ""
                for i in range(len(d)):
                    tmp += 'In row {} , '.format(i + 1)
                    for _ in e[1]:
                        if isinstance(d.iloc[i][columns[_]], str):
                            entity = map(lambda x: x.capitalize(), d.iloc[i][columns[_]].split(' '))
                            entity = ' '.join(entity)
                        else:
                            entity = str(d.iloc[i][columns[_]])

                        tmp += 'the {} is {} , '.format(columns[_], entity)
                    tmp = tmp[:-3] + ' . '

                tmp_idx = self.tokenizer.tokenize('Given the table title of "{}" . {}'.format(e[2], tmp))
                if len(tmp_idx) > self.max_len:
                    tmp_idx = tmp_idx[:self.max_len]            
                #tmp_suffix = self.tokenizer.tokenize('Start describing : ')

                descs.append(self.tokenizer.convert_tokens_to_ids(tmp_idx))

            length = max([len(_) for _ in seqs]) + 1
            for i in range(len(seqs)):
                seqs[i] += (length - len(seqs[i])) * [self.tokenizer.eos_token_id]
                seq_masks[i] = seq_masks[i] + [1] + (length - len(seq_masks[i]) - 1) * [0] 
            seqs = torch.LongTensor(seqs)
            seq_masks = torch.FloatTensor(seq_masks)

            length = max([len(_) for _ in descs]) + 1
            for i in range(len(descs)):
                descs[i] = (length - len(descs[i])) * [self.tokenizer.eos_token_id] + descs[i]
            descs = torch.LongTensor(descs)

            inputs = seqs[:, :-1]
            outputs = seqs

            pairs.append((inputs, outputs, seq_masks, descs))
        
        return pairs

class GPTTableDataset2(Dataset):
    def __init__(self, train_name, tokenizer, max_len):
        super(GPTTableDataset2, self).__init__()
        with open(train_name, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seq_len = 50

    def __getitem__(self, index):
        e = self.data[index]

        tmp = e[-1]

        tmp_idx = self.tokenizer.tokenize('Given the table title of "{}" . {} Start describing :'.format(e[2], tmp))
        if len(tmp_idx) > self.max_len:
            tmp_idx = tmp_idx[:self.max_len]
        else:
            tmp_idx = [self.tokenizer.eos_token] * (self.max_len - len(tmp_idx)) + tmp_idx

        desc = self.tokenizer.convert_tokens_to_ids(tmp_idx)
        assert len(desc) == self.max_len

        seq = self.tokenizer.tokenize(e[0])
        if len(seq) >= self.seq_len:
            seq = seq[:self.seq_len]
            seq_mask = [1] * self.seq_len
        else:
            seq_mask = [1] * (len(seq) + 1) + [0] * (self.seq_len - len(seq) - 1)
            seq = seq + [self.tokenizer.eos_token_id] * (self.seq_len - len(seq))
        seq = self.tokenizer.convert_tokens_to_ids(seq)

        desc = torch.tensor(desc, dtype=torch.long)
        inputs = torch.tensor(seq[:-1], dtype=torch.long)
        outputs = torch.tensor(seq, dtype=torch.long)
        seq_mask = torch.tensor(seq_mask, dtype=torch.long)

        return inputs, outputs, seq_mask, desc

    def __len__(self):
        return len(self.data)

class NormalTableDatabase(Dataloader):
    def __init__(self, train_name, val_name, test_name, batch_size=5, max_len=800):
        super(NormalTableDatabase, self).__init__(train_name, val_name, test_name)
        self.batch_size = batch_size
        self.max_len = max_len
        with open('data/vocab.json') as f:
            self.vocab = json.load(f)
        self.ivocab = {v: k for k, v in self.vocab.items()}

        with open('data/full_vocab.json') as f:
            self.full_vocab = json.load(f)
        self.full_ivocab = {v: k for k, v in self.full_vocab.items()}

    def get_data(self, idx, option='train', debug=False, with_fields=False):
        table_id, entry = self.obtain_idx(idx, option)
        d = pandas.read_csv('data/all_csv/' + table_id, '#')

        columns = d.columns

        table_in = []
        table_scatters = []
        seqs_in = []
        seqs_out = []
        captions_in = []
        captions_scatters = []
        fields = []
        indexes = []
        lookups = []
        line_nos = []
        input_fields = []

        if option in ['train']:
            random.shuffle(entry)
            entry = entry[:self.batch_size]

        for e in entry:
            e[0] = e[0].lower()
            e_in = "<SOS> " + e[0]
            e_out = e[0] + " <EOS>"
            input_fields.append(e[1])

            seqs_out.append([self.full_vocab.get(_, self.full_vocab['<UNK>']) for _ in e_out.split(' ')])
            seqs_in.append([self.vocab.get(_, self.vocab['<UNK>']) for _ in e_in.split(' ')])

            cap_in = [self.vocab.get(_, self.vocab['<UNK>']) for _ in e[2].split(' ')]
            cap_scatter = [self.full_vocab.get(_, self.vocab['<UNK>']) for _ in e[2].split(' ')]

            table_orig = []
            table_unk = []
            field = []
            index = []
            lookup = []
            line_no = []
            for i in range(len(d)):
                for j, col_no in enumerate(e[1]):
                    if i == 0:
                        res = columns[col_no].split(' ') + ['<SEP>']
                        field.extend([self.vocab.get(_, self.vocab['<UNK>']) for _ in res])
                        index.append(len(field) - 1)

                    cell = str(d.iloc[i][columns[col_no]]).split(' ')

                    table_orig.extend([self.full_vocab.get(_, self.full_vocab['<UNK>']) for _ in cell])
                    table_unk.extend([self.vocab.get(_, self.vocab['<UNK>']) for _ in cell])

                    lookup.extend([j] * len(cell))
                    line_no.extend([self.vocab['#{}'.format(i + 1)]] * len(cell))

            table_orig.extend(cap_scatter)
            table_unk.extend(cap_in)

            lookup.extend([len(e[1])] * len(cap_in))
            line_no.extend([self.vocab['#0']] * len(cap_in))

            table_in.append(table_unk)
            table_scatters.append(table_orig)

            fields.append(field)
            indexes.append(index)
            lookups.append(lookup)
            line_nos.append(line_no)

        length = max([len(_) for _ in seqs_in]) + 1
        for i in range(len(seqs_in)):
            seqs_out[i] += (length - len(seqs_out[i])) * [self.vocab['<PAD>']]
            seqs_in[i] += (length - len(seqs_in[i])) * [self.vocab['<PAD>']]
        seqs_out = torch.LongTensor(seqs_out)
        seqs_in = torch.LongTensor(seqs_in)

        length = max([len(_) for _ in table_in]) + 1
        for i in range(len(table_in)):
            table_in[i] += (length - len(table_in[i])) * [self.vocab['<PAD>']]
            table_scatters[i] += (length - len(table_scatters[i])) * [self.vocab['<PAD>']]
            lookups[i] += (length - len(lookups[i])) * [0]
            line_nos[i] += (length - len(line_nos[i])) * [self.vocab['<PAD>']]
        table_in = torch.LongTensor(table_in)
        table_scatters = torch.LongTensor(table_scatters)

        length = max([len(_) for _ in fields]) + 1
        for i in range(len(fields)):
            fields[i] += (length - len(fields[i])) * [self.vocab['<PAD>']]
        fields = torch.LongTensor(fields)

        length = max([len(_) for _ in indexes]) + 1
        for i in range(len(indexes)):
            indexes[i] += (length - len(indexes[i])) * [len(fields[i]) - 1]
        indexes = torch.LongTensor(indexes)

        lookups = torch.LongTensor(lookups)
        line_nos = torch.LongTensor(line_nos)

        if with_fields:
            return seqs_in, seqs_out, table_in, table_scatters, lookups, line_nos, fields, indexes, input_fields
        else:
            return seqs_in, seqs_out, table_in, table_scatters, lookups, line_nos, fields, indexes

    def get_pair_data(self, idx, option='val'):
        table_id, entry = self.obtain_idx(idx, option)
        d = pandas.read_csv('data/all_csv/' + table_id, '#')

        columns = d.columns
        pairs = []
        for opt in ['pos', 'neg']:
            table_in = []
            table_scatters = []
            seqs_in = []
            seqs_out = []
            captions_in = []
            captions_scatters = []
            fields = []
            indexes = []
            lookups = []
            line_nos = []
            for e in entry:
                e = e[opt]

                e[0] = e[0].lower()
                e_in = "<SOS> " + e[0]
                e_out = e[0] + " <EOS>"

                seqs_out.append([self.full_vocab.get(_, self.full_vocab['<UNK>']) for _ in e_out.split(' ')])
                seqs_in.append([self.vocab.get(_, self.vocab['<UNK>']) for _ in e_in.split(' ')])

                cap_in = [self.vocab.get(_, self.vocab['<UNK>']) for _ in e[2].split(' ')]
                cap_scatter = [self.full_vocab.get(_, self.vocab['<UNK>']) for _ in e[2].split(' ')]

                table_orig = []
                table_unk = []
                field = []
                index = []
                lookup = []
                line_no = []
                for i in range(len(d)):
                    for j, col_no in enumerate(e[1]):
                        if i == 0:
                            res = columns[col_no].split(' ') + ['<SEP>']
                            field.extend([self.vocab.get(_, self.vocab['<UNK>']) for _ in res])
                            index.append(len(field) - 1)

                        cell = str(d.iloc[i][columns[col_no]]).split(' ')

                        table_orig.extend([self.full_vocab.get(_, self.full_vocab['<UNK>']) for _ in cell])
                        table_unk.extend([self.vocab.get(_, self.vocab['<UNK>']) for _ in cell])

                        lookup.extend([j] * len(cell))
                        line_no.extend([self.vocab['#{}'.format(i + 1)]] * len(cell))

                table_orig.extend(cap_scatter)
                table_unk.extend(cap_in)

                lookup.extend([len(e[1])] * len(cap_in))
                line_no.extend([self.vocab['#0']] * len(cap_in))

                table_in.append(table_unk)
                table_scatters.append(table_orig)

                fields.append(field)
                indexes.append(index)
                lookups.append(lookup)
                line_nos.append(line_no)

            length = max([len(_) for _ in seqs_in]) + 1
            for i in range(len(seqs_in)):
                seqs_out[i] += (length - len(seqs_out[i])) * [self.vocab['<PAD>']]
                seqs_in[i] += (length - len(seqs_in[i])) * [self.vocab['<PAD>']]
            seqs_out = torch.LongTensor(seqs_out)
            seqs_in = torch.LongTensor(seqs_in)

            length = max([len(_) for _ in table_in]) + 1
            for i in range(len(table_in)):
                table_in[i] += (length - len(table_in[i])) * [self.vocab['<PAD>']]
                table_scatters[i] += (length - len(table_scatters[i])) * [self.vocab['<PAD>']]
                lookups[i] += (length - len(lookups[i])) * [0]
                line_nos[i] += (length - len(line_nos[i])) * [self.vocab['<PAD>']]
            table_in = torch.LongTensor(table_in)
            table_scatters = torch.LongTensor(table_scatters)

            length = max([len(_) for _ in fields]) + 1
            for i in range(len(fields)):
                fields[i] += (length - len(fields[i])) * [self.vocab['<PAD>']]
            fields = torch.LongTensor(fields)

            length = max([len(_) for _ in indexes]) + 1
            for i in range(len(indexes)):
                indexes[i] += (length - len(indexes[i])) * [len(fields[i]) - 1]
            indexes = torch.LongTensor(indexes)

            lookups = torch.LongTensor(lookups)
            line_nos = torch.LongTensor(line_nos)

            pairs.append((seqs_in, seqs_out, table_in, table_scatters, lookups, line_nos, fields, indexes))

        return pairs


class BERTTableDatabase(Dataloader):
    def __init__(self, train_name, val_name, test_name, tokenizer, batch_size=5, max_len=800):
        super(BERTTableDatabase, self).__init__(train_name, val_name, test_name)
        self.tokenizer = tokenizer
        self.batch_size = 1
        self.max_len = max_len
        self.seq_len = 30

    def get_data(self, idx, option='train', debug=False):
        table_id, entry = self.obtain_idx(idx, option)
        d = pandas.read_csv('data/all_csv/' + table_id, '#')

        columns = d.columns
        if option == 'train':
            random.shuffle(entry)
            e = entry[0]

            desc = linearize_table(d, e[1], e[2])

            e_tokens = self.tokenizer.tokenize(e[0])

            e_idx = self.tokenizer.convert_tokens_to_ids(e_tokens)

            inputs = []
            outputs = []
            for i in range(len(e_tokens)):
                inputs.append(self.tokenizer.convert_tokens_to_ids(e_tokens[:i] + ['[MASK]'] * (len(e_tokens) - i)))
                outputs.append([-1] * i + [e_idx[i]] + [-1] * (len(e_tokens) - i - 1))

            desc = torch.LongTensor(desc).unsqueeze(0)
            inputs = torch.LongTensor(inputs)
            outputs = torch.LongTensor(outputs)

            return inputs, outputs, None, desc
        else:
            desc = []
            output = []
            for e in entry:
                desc.append(linearize_table(d, e[1], e[2]))

                e_tokens = self.tokenizer.tokenize(e[0])
                if len(e_tokens) > self.seq_len:
                    e_tokens = e_tokens[:self.seq_len - 1] + ['[PAD]']
                else:
                    e_tokens = e_tokens + ['[PAD]'] * (self.seq_len - len(e_tokens))

                e_idx = self.tokenizer.convert_tokens_to_ids(e_tokens)
                output.append(e_idx)

            max_len = max([len(_) for _ in desc])
            for i in range(len(desc)):
                desc[i] = desc[i] + [0] * (max_len - len(desc[i]))

            desc = torch.LongTensor(desc)
            output = torch.LongTensor(output)

            return desc, output


class BERTScorerDatabase(Dataloader):
    def __init__(self, train_name, val_name, test_name, tokenizer, batch_size=5, max_len=800):
        super(BERTScorerDatabase, self).__init__(train_name, val_name, test_name)
        self.tokenizer = tokenizer
        # self.feat_prefix = 'table_feats/'
        self.batch_size = 1
        self.max_len = max_len
        self.seq_len = 30

    def get_data(self, idx, option='train', debug=False):
        table_id, entry = self.obtain_idx(idx, option)
        d = pandas.read_csv('data/all_csv/' + table_id, '#')

        random.shuffle(entry)

        e = entry[0]

        e_tokens = self.tokenizer.tokenize(e[0])
        desc = linearize_table(d, e[1], e[2])

        e_idx = self.tokenizer.convert_tokens_to_ids(e_tokens)

        inputs = []
        outputs = []
        for i in range(len(e_tokens)):
            inputs.append(self.tokenizer.convert_tokens_to_ids(e_tokens[:i] + ['[MASK]'] + e_tokens[i + 1:]))
            outputs.append([-1] * i + [e_idx[i]] + [-1] * (len(e_tokens) - i - 1))

        desc = torch.LongTensor(desc).unsqueeze(0)
        inputs = torch.LongTensor(inputs)
        outputs = torch.LongTensor(outputs)

        return inputs, outputs, None, desc


def linearize_table(d, sub_columns, title, tokenizer):
    columns = d.columns
    tmp = ""
    for i in range(len(d)):
        tmp += 'In row {} , '.format(i + 1)
        for _ in sub_columns:
            if isinstance(d.iloc[i][columns[_]], str):
                entity = map(lambda x: x.capitalize(), d.iloc[i][columns[_]].split(' '))
                entity = ' '.join(entity)
            else:
                entity = str(d.iloc[i][columns[_]])

            tmp += 'the {} is {} , '.format(columns[_], entity)
        tmp = tmp[:-3] + ' . '

    tmp_idx = tokenizer.tokenize(tmp)
    if len(tmp_idx) > 480:
        tmp_idx = tmp_idx[:480]

    tmp_prefix = tokenizer.tokenize('Given the table title of "{}" . '.format(title))
    desc = tokenizer.convert_tokens_to_ids(tmp_prefix + tmp_idx)

    return desc