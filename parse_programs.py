import json
import os
import sys
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from Model import BERTRanker
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from multiprocessing import Pool
import multiprocessing
from parser import Parser, recursion, program_eq, split_prog
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import random
import warnings
from datetime import datetime 
warnings.filterwarnings("ignore", 'This pattern has match groups')
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer, 
                        BertModel, get_linear_schedule_with_warmup, 
                        squad_convert_examples_to_features)
from APIs import all_funcs

device = torch.device('cuda')
MODEL_CLASSES = {"bert": (BertConfig, BertModel, BertTokenizer)}

class ParseNLIDataset(Dataset):
    def __init__(self, bootstrap_data, weakly_data, tokenizer):
        self.bootstrap_data = bootstrap_data
        self.weakly_data = weakly_data
        self.tokenizer = tokenizer

        #self.sent_max_len = 80
        self.max_len = 120

    @classmethod
    def convert(cls, sent, prog, title, tokenizer, max_len):
        title = '[CLS] title : {} [SEP]'.format(title)
        title_ids = tokenizer.encode(title, add_special_tokens=False)
        types = [1] * len(title_ids)

        sent = '{} [SEP]'.format(sent)
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        types += [0] * len(sent_ids)

        prog = '{} [SEP]'.format(prog)
        prog_ids = tokenizer.encode(prog, add_special_tokens=False)
        types += [1] * len(prog_ids)

        token_ids = title_ids + sent_ids + prog_ids

        if len(types) > max_len:
            token_ids = token_ids[:max_len]
            masks = [1] * max_len            
            types = types[:max_len]
        else:
            token_ids = token_ids + [tokenizer.pad_token_id] * (max_len - len(types))
            masks = [1] * len(types) + [0] * (max_len - len(types))            
            types = types + [0] * (max_len - len(types))

        return token_ids, masks, types

    def __getitem__(self, index):
        if random.random() < 0.4:
            entry = random.choice(self.bootstrap_data)
        else:
            entry = random.choice(self.weakly_data)
        #entry = random.choice(self.weakly_data)

        sent = entry[0]
        prog = entry[1]
        title = entry[2]
        label = entry[3]

        token_ids, masks, types = self.convert(sent, prog, title, self.tokenizer, self.max_len)

        token_ids = np.array(token_ids, 'int64')
        types = np.array(types, 'int64')
        masks = np.array(masks, 'int64')
        label = np.array(label, 'int64')

        return token_ids, types, masks, label
    
    def __len__(self):
        return len(self.bootstrap_data) + len(self.weakly_data)
        #return len(self.weakly_data)

def get_model(model_type, model_name_or_path, cache_dir):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
    )
    tokenizer = tokenizer_class.from_pretrained(
        model_name_or_path,
        do_lower_case=True,
        cache_dir=cache_dir,
    )
    tokenizer.add_tokens(["[{}]".format(_) for _ in all_funcs])
    tokenizer.add_tokens(["all_rows"])
    model = BERTRanker(model_class, model_name_or_path, config, cache_dir)
    model.base.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

def convert_program(program):
    arrays, _ = split_prog(program, True)
    for i in range(len(arrays) - 1):
        if arrays[i + 1] == '{':
            arrays[i] = '[{}]'.format(arrays[i])
    return " ".join(arrays)

def evaluate(tokenizer, model, parser):
    with open('data/test_lm_pos_neg.json') as f:
        data = json.load(f)
    data = dict(list(data.items())[:200])

    positive_sents = []
    negative_sents = []
    table_names = []
    for k, vs in data.items():
        for v in vs:
            positive_sents.append(v['pos'][0])
            negative_sents.append(v['neg'][0])
            table_names.append(k)

    cores = multiprocessing.cpu_count()
    print("Using {} cores to run on {} instances".format(cores, len(positive_sents)))
    pool = Pool(cores)
    pos_res = pool.map(parser.distribute_parse, zip(table_names, positive_sents))
    pool.close()
    pool.join()

    print("Using {} cores to run on {} instances".format(cores, len(negative_sents)))
    pool = Pool(cores)
    neg_res = pool.map(parser.distribute_parse, zip(table_names, negative_sents))
    pool.close()
    pool.join()

    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    for res, polarity in zip([pos_res, neg_res], [True, False]):
        for sent, programs, title in res:
            labels = []
            token_ids = []
            types = []
            masks = []
            
            if len(programs) > 0:
                for prog in programs[:64]:
                    token_id, type_, mask = ParseNLIDataset.convert(sent, convert_program(prog), title, tokenizer, 120)
                    token_ids.append(token_id)
                    types.append(type_)
                    masks.append(mask)
                    labels.append('=True' in prog)

                token_ids = torch.LongTensor(token_ids).to(device)
                types = torch.LongTensor(types).to(device)
                masks = torch.LongTensor(masks).to(device)
                
                probs = model.prob(token_ids, types, masks)
                pred = labels[torch.argmax(probs, 0).item()]
            else:
                pred = False

            if pred and polarity:
                tp += 1
            elif pred and not polarity:
                fp += 1
            elif not pred and polarity:
                fn += 1
            else:
                tn += 1

            sys.stdout.write("TP={},TN={},FP={},FN={},ACC={} \r".format(tp, tn, fp, fn, (tp + tn)/(tp + tn + fp + fn)))

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    print("TP={},TN={},FP={},FN={},ACC={}".format(tp, tn, fp, fn, accuracy))
    model.train()

    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_prog", default=False, action="store_true", help="whether to generate programs")
    parser.add_argument("--rank_prog_bert", default=False, action="store_true", help="whether to rank programs")
    parser.add_argument("--parse", default=False, action="store_true", help="whether to parse the output file")
    parser.add_argument("--compute_score", default=False, action="store_true", help="whether to compute SP-Acc score")
    parser.add_argument("--score_file", default=None, type=str, help="The input file to be scored")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size for training")
    parser.add_argument("--model_type", default='bert', type=str, help="the model type")
    parser.add_argument("--preprocess", default=False, action="store_true", help="whether to rank programs")
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, help="batch size for training")    
    parser.add_argument("--num_workers", default=16, type=int, help="number of workers in the dataloader")
    parser.add_argument("--load_from", default=None, type=str, help="which model to load from")
    parser.add_argument("--csv_path", default='data/all_csv', type=str, help="all_csv path")
    parser.add_argument("--cache_dir", default='/tmp/', type=str, help="where to cache the BERT model")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")        
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--logging_steps", default=50, type=int, help="Epsilon for Adam optimizer.")    
    args = parser.parse_args()
    
    if args.gen_prog:
        model = Parser(args.csv_path)
        with open('data/bootstrap.json') as f:
            data = json.load(f)
        
        indexing = []
        table_names = []
        titles = []
        sents = []
        parses = []
        for k, vs in data.items():
            for v in vs:
                indexing.append(len(table_names))
                table_names.append(k)
                titles.append(v[2])
                sents.append(v[0])
                parses.append(v[-1])


        def func(inputs):
            index, table_name, title, sent, parse = inputs
            
            if os.path.exists('data/bootstrap_programs/{}/{}'.format('fail', table_name)) or\
                os.path.exists('data/bootstrap_programs/{}/{}'.format('success', table_name)):
                return
            else:
                rs, masked_sent, mapping = model.parse(table_name, sent)
                result = 'fail'
                for r in rs:
                    if program_eq(r, parse):
                        result = 'success'
                        break
                output_file = 'data/bootstrap_programs/{}/{}'.format(result, table_name)

                with open(output_file, 'w') as f:
                    json.dump((table_name, sent, title, rs, parse, masked_sent, mapping), f, indent=2)

        cores = multiprocessing.cpu_count()
        print("Using {} cores for {} instances".format(cores, len(sents)))
        pool = Pool(cores)
        res = pool.map(func, zip(indexing, table_names, titles, sents, parses))

        pool.close()
        pool.join()
        model = Parser(args.csv_path)

        with open('data/train_lm.json') as f:
            data = json.load(f)

        indexing = []
        table_names = []
        titles = []
        sents = []
        for k, vs in data.items():
            for v in vs:
                indexing.append(len(table_names))
                table_names.append(k)
                titles.append(v[2])
                sents.append(v[0])

        def func(inputs):
            index, table_name, title, sent = inputs
            output_file = 'data/all_programs/{}_program.json'.format(index)
            if not os.path.exists(output_file):
                rs, masked_sent, mapping = model.parse(table_name, sent)
                with open(output_file, 'w') as f:
                    json.dump((table_name, sent, title, rs, None, masked_sent, mapping), f, indent=2)
            else:
                pass

        cores = multiprocessing.cpu_count()
        print("Using {} cores for {} instances".format(cores, len(sents)))
        pool = Pool(cores)
        res = pool.map(func, zip(indexing, table_names, titles, sents))
        pool.close()
        pool.join()

        with open('data/train_adv_lm.json') as f:
            data = json.load(f)

        indexing = []
        table_names = []
        titles = []
        sents = []
        for k, vs in data.items():
            for v in vs:
                indexing.append(len(table_names))
                table_names.append(k)
                titles.append(v[2])
                sents.append(v[0])

        def func(inputs):
            index, table_name, title, sent = inputs
            output_file = 'data/all_adv_programs/{}_program.json'.format(index)
            if not os.path.exists(output_file):
                rs, masked_sent, mapping = model.parse(table_name, sent)
                with open(output_file, 'w') as f:
                    json.dump((table_name, sent, title, rs, None, masked_sent, mapping), f, indent=2)
            else:
                pass

        cores = multiprocessing.cpu_count()
        print("Using {} cores for {} instances".format(cores, len(sents)))
        pool = Pool(cores)
        res = pool.map(func, zip(indexing, table_names, titles, sents))
        pool.close()
        pool.join()

    if args.rank_prog_bert:
        if not os.path.exists('data/training_data_for_ranker.json'):
            bootstrap_results = []
            for d in os.listdir('data/bootstrap_programs/success/'):
                if d.endswith('.csv'):
                    with open('data/bootstrap_programs/success/{}'.format(d), 'r') as f:
                        data = json.load(f)
                    
                    gt_program = data[-3]
                    mapping = data[-1]
                    imapping = {v:k for k, v in mapping.items()}
                    for prog in data[3]:
                        if program_eq(prog, gt_program):
                            label = 1
                        else:
                            label = 0
                        # split program
                        bootstrap_results.append((data[1], convert_program(prog), data[2], label))

            print("done constructing the bootstrap postive and negative samples")
            
            # Constructing the bootstrap instances        
            weakly_results = []
            for d in os.listdir('data/all_programs/'):
                if d.endswith('.json'):
                    with open('data/all_programs/{}'.format(d), 'r') as f:
                        data = json.load(f)
                    
                    mapping = data[-1]
                    imapping = {v:k for k, v in mapping.items()}
                    for prog in data[3]:
                        if '=True' in prog:
                            label = 1
                        elif '=False':
                            label = 0
                        else:
                            raise NotImplementedError
                        weakly_results.append((data[1], convert_program(prog), data[2], label))

            for d in os.listdir('data/all_adv_programs/'):
                if d.endswith('.json'):
                    with open('data/all_adv_programs/{}'.format(d), 'r') as f:
                        data = json.load(f)
                    
                    mapping = data[-1]
                    imapping = {v:k for k, v in mapping.items()}
                    for prog in data[3]:
                        if '=True' in prog:
                            label = 0
                        elif '=False':
                            label = 1
                        else:
                            raise NotImplementedError
                        weakly_results.append((data[1], convert_program(prog), data[2], label))

            print("done constructing the weakly postive and negative samples")

            with open('data/training_data_for_ranker.json', 'w') as f:
                json.dump({'bootstrap': bootstrap_results, 'weakly': weakly_results}, f, indent=2)
        else:
            with open('data/training_data_for_ranker.json', 'r') as f:
                data = json.load(f)
            bootstrap_results = data['bootstrap']
            weakly_results = data['weakly']

        tokenizer, model = get_model(args.model_type, args.model_name_or_path, args.cache_dir)
        model.to(device)
        model.train()

        dataset = ParseNLIDataset(bootstrap_results, weakly_results, tokenizer)

        train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        criterion = nn.CrossEntropyLoss()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        recording_time = datetime.now().strftime('%m_%d_%H_%M')
        tb_writer = SummaryWriter(log_dir='tmp/{}'.format(recording_time))
        
        global_step = 0
        tr_loss = 0
        parser = Parser(args.csv_path)

        evaluate_every = int(len(train_dataloader) / 10)
        print("evaluating the model every {} steps".format(evaluate_every))
        for epoch in trange(0, 10, desc='Epoch'):
            for i, batch in enumerate(tqdm(train_dataloader, 'Iteration')):
                batch = tuple(Variable(t).to(device) for t in batch)

                token_ids, types, masks, label = batch

                model.zero_grad()
                optimizer.zero_grad()

                logits = model(token_ids, types, masks)

                loss = criterion(logits, label)

                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                global_step += 1
                tr_loss += loss.item()

                if global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("loss", tr_loss / args.logging_steps, global_step)
                    tr_loss = 0

                if i % evaluate_every == 0 and i > 0:
                    acc = format(evaluate(tokenizer, model, parser), '.2f')
                    torch.save(model.state_dict(), 'parser_models/parser_step{}_acc{}.pt'.format(global_step, acc))      
        
        tb_writer.close()

    if args.parse:
        with open(args.score_file, 'r') as f:
            data = json.load(f)

        parser = Parser(args.csv_path)
        table_names = []
        sents = []
        for k, vs in data.items():
            for v in vs:
                table_names.append(k)
                sents.append(v)
        
        cores = multiprocessing.cpu_count()
        print("Using {} cores to run on {} instances".format(cores, len(sents)))
        pool = Pool(cores)
        results = pool.map(parser.distribute_parse, zip(table_names, sents))
        pool.close()
        pool.join()
        
        with open("program_{}".format(args.score_file), 'w') as f:
            json.dump(results, f, indent=2)

    if args.compute_score:
        tokenizer, model = get_model(args.model_type, args.model_name_or_path, args.cache_dir)
        model.to(device)
        model.eval()
        model.load_state_dict(torch.load(args.load_from))

        with open(args.score_file, 'r') as f:
            results = json.load(f)

        succ, total = 0, 0
        for sent, programs, title in results:
            labels = []
            token_ids = []
            types = []
            masks = []
            
            if len(programs) > 0:
                for prog in programs[:36]:
                    token_id, type_, mask = ParseNLIDataset.convert(sent, convert_program(prog), title, tokenizer, 120)
                    token_ids.append(token_id)
                    types.append(type_)
                    masks.append(mask)
                    labels.append(1 if '=True' in prog else 0)

                token_ids = torch.LongTensor(token_ids).to(device)
                types = torch.LongTensor(types).to(device)
                masks = torch.LongTensor(masks).to(device)
                
                probs = model.prob(token_ids, types, masks)
                if len(labels) > 8:
                    tmp = []
                    for _ in probs.topk(3)[1].tolist():
                        tmp.append(labels[_])
                    if sum(tmp) > 0:
                        pred = 1
                    else:
                        pred = 0
                else:
                    pred = labels[torch.argmax(probs, 0).item()]
            else:
                pred = 0
            
            if pred:
                succ += 1
            total += 1
            sys.stdout.write('accuracy = {} \r'.format(succ / total))

        print("accuracy = {}".format(succ / total))


    if args.preprocess:
        with open('data/test_lm.json') as f:
            data = json.load(f)
        data = dict(list(data.items())[:200])

        parser = Parser(args.csv_path)
        positive_sents = []
        table_names = []
        for k, vs in data.items():
            for v in vs:
                positive_sents.append(v[0])
                table_names.append(k)

        for _ in zip(table_names, positive_sents):
            print(parser.preprocess(_))