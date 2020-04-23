from DataLoader import NormalTableDatabase
from Model import TableInfusing
import sys
from torch.autograd import Variable
import torch
import torch.optim as optim
from torch import nn
import argparse
import pandas
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
from utils import sample_sequence
import numpy as np
import math
import nltk
import json

device = torch.device('cuda')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_verify', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_ppl', default=False, action="store_true", help="whether to train or test the model")    
    parser.add_argument('--epoch', default=10, type=int, help="whether to train or test the model")
    parser.add_argument('--every', default=50, type=int, help="whether to train or test the model")
    parser.add_argument('--dim', default=256, type=int, help="whether to train or test the model")
    parser.add_argument('--layers', default=3, type=int, help="whether to train or test the model")
    parser.add_argument('--head', default=4, type=int, help="whether to train or test the model")
    parser.add_argument('--load_from', default='', type=str, help="whether to train or test the model")
    parser.add_argument('--dataset', default='table', type=str, help="whether to train or test the model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_opt()

    # if args.dataset == 'table':
    dataset = NormalTableDatabase('data/train_lm.json', 'data/val_lm.json', 'data/test_lm.json')
    model = TableInfusing(len(dataset.vocab), len(dataset.full_vocab), args.dim, args.layers, args.head)
    model.to(device)

    if args.do_train:
        model.train()
        optimizer = optim.Adam(model.parameters(), 2e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        if args.load_from != "":
            model.load_state_dict(torch.load(args.load_from))
            print("loading model from {}".format(args.load_from))

        avg_loss = 0
        for epoch_idx in range(args.epoch):
            print("start training {}th epoch".format(epoch_idx))
            dataset.shuffle()
            for idx in range(0, dataset.train_len()):
                batch = dataset.get_data(idx)
                batch = tuple(Variable(t).to(device) for t in batch)
                seqs_in, seqs_out, table_in, table_scatters, lookups, line_nos, fields, indexes = batch

                model.zero_grad()
                optimizer.zero_grad()
                logits = model(seqs_in, table_in, table_scatters, lookups, line_nos, fields, indexes)

                loss = criterion(logits.view(-1, logits.shape[-1]), seqs_out.view(-1))

                avg_loss += loss.item()

                loss.backward()
                optimizer.step()

                if idx % args.every == 0 and idx > 0:
                    #sys.stdout.write('finished {} samples loss = {} \r'.format(idx, avg_loss / 50))
                    print('finished {}/{} samples loss = {}, perpelexity = {}'.format(
                        idx, dataset.train_len(), avg_loss / args.every, math.exp(avg_loss / args.every)))

                    avg_loss = 0

            model.eval()

            with torch.no_grad():
                losses = []
                for idx in range(0, dataset.val_len()):
                    batch = dataset.get_data(idx)
                    batch = tuple(Variable(t).to(device) for t in batch)
                    seqs_in, seqs_out, table_in, table_scatters, lookups, line_nos, fields, indexes = batch
                    logits = model(seqs_in, table_in, table_scatters, lookups, line_nos, fields, indexes)
                    loss = criterion(logits.view(-1, logits.shape[-1]), seqs_out.view(-1))
                    losses.append(loss)
                    # ppls.append(math.exp(loss))
                    avg_loss = sum(losses) / len(losses)
                    sys.stdout.write("perplexity is {} \r".format(math.exp(avg_loss)))

            avg_loss = sum(losses) / len(losses)
            print("total perplexity is {}".format(math.exp(avg_loss)))
            torch.save(model.state_dict(), 'models/transformer_ep{}.pt'.format(epoch_idx))

            model.train()

    if args.do_val:
        model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        model.load_state_dict(torch.load(args.load_from))
        print("loading model from {}".format(args.load_from))
        with torch.no_grad():
            losses = []
            for idx in range(0, dataset.val_len()):
                batch = dataset.get_data(idx)
                batch = tuple(Variable(t).to(device) for t in batch)
                seqs_in, seqs_out, table_in, table_scatters, lookups, line_nos, fields, indexes = batch
                logits = model(seqs_in, table_in, table_scatters, lookups, line_nos, fields, indexes)
                loss = criterion(logits.view(-1, logits.shape[-1]), seqs_out.view(-1))
                losses.append(loss)
                avg_loss = sum(losses) / len(losses)
                sys.stdout.write("perplexity is {} \r".format(math.exp(avg_loss)))

        avg_loss = sum(losses) / len(losses)
        print("total perplexity is {}".format(math.exp(avg_loss)))

    if args.do_ppl:
        model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        model.load_state_dict(torch.load(args.load_from))
        print("loading model from {}".format(args.load_from))
        with torch.no_grad():
            losses = []
            for idx in range(0, dataset.test_len()):
                batch = dataset.get_data(idx, 'test', False)
                batch = tuple(Variable(t).to(device) for t in batch)
                seqs_in, seqs_out, table_in, table_scatters, lookups, line_nos, fields, indexes = batch
                logits = model(seqs_in, table_in, table_scatters, lookups, line_nos, fields, indexes)
                loss = criterion(logits.view(-1, logits.shape[-1]), seqs_out.view(-1))
                losses.append(loss)
                avg_loss = sum(losses) / len(losses)
                sys.stdout.write("perplexity is {} \r".format(math.exp(avg_loss)))

        avg_loss = sum(losses) / len(losses)
        print("total perplexity is {}".format(math.exp(avg_loss)))        

    if args.do_test:
        model.eval()
        model.load_state_dict(torch.load(args.load_from))
        print("loading model from {}".format(args.load_from))

        results = {}
        sent_bleus_1 = []
        sent_bleus_2 = []
        sent_bleus_3 = []
        sent_bleus_4 = []

        with open('data/table_to_page.json') as f:
            mapping = json.load(f)

        with torch.no_grad():
            for idx in range(0, dataset.test_len()):
                #print("sampling from {}".format(dataset.get_item(idx, 'test')), file=f)
                table_id = dataset.get_item(idx, 'test')
                table = pandas.read_csv('data/all_csv/' + table_id, '#')

                results[table_id] = []

                *batch, input_fields = dataset.get_data(idx, 'test', False, with_fields=True)
                references = dataset.get_reference(idx, 'test')

                batch = tuple(Variable(t).to(device) for t in batch)
                seqs_in, seqs_out, table_in, table_scatters, lookups, line_nos, fields, indexes = batch

                enc_inp = model.encode(table_in, lookups, line_nos, fields, indexes)

                sents = [[] for _ in range(seqs_in.shape[0])]
                preds = seqs_in[:, :1]
                finished = set()
                for i in range(30):
                    logits = model.decode(preds, enc_inp, table_scatters)[:, -1, :]

                    preds_i = torch.argmax(logits, -1)

                    tmp = []
                    for j, _ in enumerate(preds_i):
                        word = dataset.full_ivocab[_.item()]

                        if word == '<EOS>':
                            finished.add(j)
                        elif j not in finished:
                            sents[j].append(word)

                        if _.item() >= len(dataset.vocab):
                            tmp.append(dataset.vocab['<UNK>'])
                        else:
                            tmp.append(_.item())

                    preds = torch.cat([preds, torch.LongTensor(tmp).to(device).unsqueeze(-1)], -1)

                preds = preds.cpu().data.numpy()

                for hypothesis, input_field in zip(sents, input_fields):
                    results[table_id].append(' '.join(hypothesis))
                    sent_bleus_1.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(1, 0, 0)))
                    sent_bleus_2.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(0.5, 0.5, 0)))
                    sent_bleus_3.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(0.33, 0.33, 0.33)))

                bleu_1 = sum(sent_bleus_1) / len(sent_bleus_1)
                bleu_2 = sum(sent_bleus_2) / len(sent_bleus_2)
                bleu_3 = sum(sent_bleus_3) / len(sent_bleus_3)
                sys.stdout.write("finished {}/{} BLEU score {}/{}/{} \r".format(idx,
                                                                                dataset.test_len(), bleu_1, bleu_2, bleu_3))

        print("total corpus BLEU score = {}/{}/{}".format(bleu_1, bleu_2, bleu_3))
        with open('outputs/field_infusing.json', 'w') as f:
            json.dump(results, f, indent=2)

    if args.do_verify:
        dataset = NormalTableDatabase(None, 'data/val_lm_pos_neg.json', 'data/test_lm_pos_neg.json')
        model.load_state_dict(torch.load(args.load_from))
        model.eval()
        print("loading model from {}".format(args.load_from))
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

        correct, total = 0, 0
        with torch.no_grad():
            for idx in range(0, dataset.test_len()):
                batch_pos, batch_neg = dataset.get_pair_data(idx, option='test')
                batch = tuple(Variable(t).to(device) for t in batch_pos)
                seqs_in, seqs_out, table_in, table_scatters, lookups, line_nos, fields, indexes = batch
                logits = model(seqs_in, table_in, table_scatters, lookups, line_nos, fields, indexes)
                loss = criterion(logits.view(-1, logits.shape[-1]), seqs_out.view(-1))
                loss = loss.view(seqs_in.shape[0], -1)
                mask = (loss > 0).float()
                loss_per_instance = (loss * mask).sum(1) / mask.sum(1)
                pos_perpelexity_per_instance = torch.exp(loss_per_instance.cpu().data)

                batch = tuple(Variable(t).to(device) for t in batch_neg)
                seqs_in, seqs_out, table_in, table_scatters, lookups, line_nos, fields, indexes = batch
                logits = model(seqs_in, table_in, table_scatters, lookups, line_nos, fields, indexes)
                loss = criterion(logits.view(-1, logits.shape[-1]), seqs_out.view(-1))
                loss = loss.view(seqs_in.shape[0], -1)
                mask = (loss > 0).float()
                loss_per_instance = (loss * mask).sum(1) / mask.sum(1)
                neg_perpelexity_per_instance = torch.exp(loss_per_instance.cpu().data)

                comparison = (pos_perpelexity_per_instance < neg_perpelexity_per_instance).float()
                correct += comparison.sum(-1).item()
                total += comparison.shape[0]
                sys.stdout.write('finished {}/{} accuracy {} \r'.format(idx, dataset.test_len(), correct / total))
        
        print('total accuracy = {}'.format(correct / total))
