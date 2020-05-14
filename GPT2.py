import argparse
import logging
from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable
from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer
from DataLoader import *
from Model import BERTGen
from utils import sample_sequence
import torch.optim as optim
import math
import sys
import pandas
import os
import numpy
import nltk
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda')

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt2', type=str)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_rl', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to compute the BLEU scores on test split")
    parser.add_argument('--do_test_challenge', default=False, action="store_true", help="whether to compute the BLEU scores on challenge split")
    parser.add_argument('--do_ppl', default=False, action="store_true", help="whether to compute perplexity of the model")
    parser.add_argument('--do_verify', default=False, action="store_true", help="whether compute the adv-acc score on test split")
    parser.add_argument('--do_verify_challenge', default=False, action="store_true", help="whether compute the adv-acc score on challenge split")
    parser.add_argument('--epoch', default=10, type=int, help="whether to train or test the model")
    parser.add_argument('--batch_size', default=5, type=int, help="whether to train or test the model")
    parser.add_argument('--learning_rate', default=2e-6, type=float, help="whether to train or test the model")
    parser.add_argument('--dataset', default='table', type=str, help="whether to train or test the model")
    parser.add_argument('--every', default=50, type=int, help="whether to train or test the model")
    parser.add_argument('--load_from', default='', type=str, help="whether to train or test the model")
    parser.add_argument('--id', default='models', type=str, help="specify the id of the experiment")
    parser.add_argument('--max_len', default=800, type=int, help="whether to train or test the model")
    parser.add_argument('--dim', default=768, type=int, help="whether to train or test the model")
    parser.add_argument('--layers', default=3, type=int, help="whether to train or test the model")
    parser.add_argument('--head', default=4, type=int, help="whether to train or test the model")
    parser.add_argument("--modelpath", type=str, default="bert-base-uncased",
                        help="For distributed training: local_rank")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=5, help="accumulation steps for gradient")
    parser.add_argument('--decode_first_K', type=int, default=10000, help="For debugging purpose")
    args = parser.parse_args()

    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    if args.model == 'gpt2-medium':
        args.batch_size = 2
    else:
        args.batch_size = 5

    if args.do_rl:
        args.batch_size = 1

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model = nn.DataParallel(model)
    model.to(args.device)

    if not os.path.exists(args.id):
        os.mkdir(args.id)

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    if args.do_train:
        tb_writer = SummaryWriter(log_dir='tensorboard/GPT2-{}'.format(args.model))
        dataset = GPTTableDatabase('data/train_lm.json', None, None, tokenizer, args.batch_size, args.max_len)        
        model.train()
        optimizer = optim.Adam(model.parameters(), args.learning_rate)

        avg_loss = 0
        global_step = 0
        for epoch_idx in range(args.epoch):
            print("start training {}th epoch".format(epoch_idx))
            dataset.shuffle()
            for idx in range(0, dataset.train_len()):
                batch = dataset.get_data(idx)
                batch = tuple(Variable(t).to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch
                inputs = torch.cat([caption, trg_inp], 1)

                model.zero_grad()
                optimizer.zero_grad()

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))

                loss = loss * mask.view(-1)
                loss = loss.sum() / mask.sum()

                avg_loss += loss.item()

                loss.backward()
                optimizer.step()
                global_step += 1

                if idx % args.every == 0 and idx > 0:
                    tb_writer.add_scalar("perplexity", math.exp(avg_loss / args.every), global_step)

                    fake_inputs = caption
                    gt_inputs = trg_out.cpu().data.numpy()

                    #samples = model.sample(fake_inputs, tabfeat, caption, highlight_idx, bert)
                    samples = sample_sequence(model, 30, fake_inputs, [])
                    samples = samples[:, caption.shape[1]:]
                    samples = samples.cpu().data.numpy()

                    for s, gt in zip(samples, gt_inputs):
                        text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                        text = text[: text.find(tokenizer.eos_token)]
                        print("PREDICTION |||||| ", text)
                        text = tokenizer.decode(gt, clean_up_tokenization_spaces=True)
                        text = text[: text.find(tokenizer.eos_token)]
                        print("GROUNDTRUH |||||| ",text)
                        break

                    avg_loss = 0

            if args.model == 'gpt2':
                torch.save(model.state_dict(), '{}/GPT_ep{}.pt'.format(args.id, epoch_idx))
            else:
                torch.save(model.state_dict(), '{}/GPT_medium_ep{}.pt'.format(args.id, epoch_idx))
        tb_writer.close()

    if args.do_val:
        dataset = GPTTableDatabase(None, 'data/val_lm.json', None, tokenizer, args.batch_size, args.max_len)
        model.load_state_dict(torch.load(args.load_from))
        model.eval()

        with torch.no_grad():
            losses = []
            for idx in range(0, dataset.val_len()):
                batch = dataset.get_data(idx, 'val')
                batch = tuple(Variable(t).to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch

                inputs = torch.cat([caption, trg_inp], 1)

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))

                loss = loss * mask.view(-1)
                loss = loss.sum() / mask.sum()

                losses.append(loss.item())

                avg_loss = sum(losses) / len(losses)
                perpelexity = math.exp(avg_loss)

                sys.stdout.write("validation perplexity is {} \r".format(perpelexity))

            avg_loss = sum(losses) / len(losses)
            perplexity = math.exp(avg_loss)

            print("validation perplexity is {}".format(perplexity))

    if args.do_ppl:
        dataset = GPTTableDatabase(None, None, 'data/test_lm.json',
                                   tokenizer, args.batch_size, args.max_len)
        model.load_state_dict(torch.load(args.load_from))
        model.eval()

        with torch.no_grad():
            losses = []
            for idx in range(0, dataset.test_len()):
                batch = dataset.get_data(idx, 'test')
                batch = tuple(Variable(t).to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch

                inputs = torch.cat([caption, trg_inp], 1)

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))

                loss = loss * mask.view(-1)
                loss = loss.sum() / mask.sum()

                losses.append(loss.item())

                avg_loss = sum(losses) / len(losses)
                perplexity = math.exp(avg_loss)                

            print("test perplexity is {}".format(perplexity))

    if args.do_test:
        dataset = GPTTableDatabase(None, None, 'data/test_lm.json', tokenizer, args.batch_size, args.max_len)
        model.load_state_dict(torch.load(args.load_from))
        model.eval()

        sent_bleus_1 = []
        sent_bleus_2 = []
        sent_bleus_3 = []

        results = {}
        with torch.no_grad():
            for idx in range(0, min(args.decode_first_K, dataset.test_len())):
                batch = dataset.get_data(idx, 'test')
                references = dataset.get_reference(idx, 'test')
                table_id = dataset.get_table_id(idx, 'test')
                results[table_id] = []

                batch = tuple(Variable(t).to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch

                fake_inputs = caption

                samples = sample_sequence(model, 30, fake_inputs, [], top_k=1)

                samples = samples[:, caption.shape[1]:]
                samples = samples.cpu().data.numpy()

                for s in samples:
                    text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                    text = text[: text.find(tokenizer.eos_token)]
                    results[table_id].append(text)

                    hypothesis = text.lower().split(' ')
                    sent_bleus_1.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(1, 0, 0)))
                    sent_bleus_2.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(0.5, 0.5, 0)))
                    sent_bleus_3.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(0.33, 0.33, 0.33)))

                bleu_1 = format((sum(sent_bleus_1) / len(sent_bleus_1) * 100), '.2f')
                bleu_2 = format((sum(sent_bleus_2) / len(sent_bleus_2) * 100), '.2f')
                bleu_3 = format((sum(sent_bleus_3) / len(sent_bleus_3) * 100), '.2f')

                sys.stdout.write("finished {}/{} BLEU score {}/{}/{} \r".format(idx, dataset.test_len(), bleu_1, bleu_2, bleu_3))

            print("total corpus BLEU score = {}/{}/{}".format(bleu_1, bleu_2, bleu_3))

        with open('outputs/GPT_{}_{}.json'.format(args.model, bleu_3), 'w') as f:
            json.dump(results, f, indent=2)

    if args.do_test_challenge:
        dataset = GPTTableDatabase(None, None, 'challenge/blind_test_lm_inputs.json', tokenizer, args.batch_size, args.max_len)
        model.load_state_dict(torch.load(args.load_from))
        model.eval()

        results = {}
        with torch.no_grad():
            for idx in range(0, min(args.decode_first_K, dataset.test_len())):
                batch = dataset.get_data(idx, 'test')
                references = dataset.get_reference(idx, 'test')
                table_id = dataset.get_table_id(idx, 'test')
                results[table_id] = []

                batch = tuple(Variable(t).to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch

                fake_inputs = caption

                samples = sample_sequence(model, 30, fake_inputs, [], top_k=1)

                samples = samples[:, caption.shape[1]:]
                samples = samples.cpu().data.numpy()

                for s in samples:
                    text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                    text = text[: text.find(tokenizer.eos_token)]
                    results[table_id].append(text)

                sys.stdout.write("finished {}/{}; speed={}s/sent \r".format(idx, 
                                 dataset.test_len(), (time.time() - start_time) / len(results)))
        
        with open('challenge/test_results.json', 'w') as f:
            json.dump(results, f, indent=2)        

    if args.do_verify:
        dataset = GPTTableDatabase(None, None, 'data/test_lm_pos_neg.json', tokenizer, args.batch_size, args.max_len)
        model.load_state_dict(torch.load(args.load_from))
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for idx in range(0, dataset.test_len()):
                batch_pos, batch_neg = dataset.get_pair_data(idx, 'test')

                batch = tuple(Variable(t).to(device) for t in batch_pos)
                trg_inp, trg_out, mask, caption = batch

                inputs = torch.cat([caption, trg_inp], 1)

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
                loss = loss.reshape(logits.shape[0], -1)
                loss_per_instance = (loss * mask).sum(1) / mask.sum(1)
                pos_perpelexity_per_instance = torch.exp(loss_per_instance.cpu().data)

                batch = tuple(Variable(t).to(device) for t in batch_neg)
                trg_inp, trg_out, mask, caption = batch

                inputs = torch.cat([caption, trg_inp], 1)

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
                loss = loss.reshape(logits.shape[0], -1)
                loss_per_instance = (loss * mask).sum(1) / mask.sum(1)
                neg_perpelexity_per_instance = torch.exp(loss_per_instance.cpu().data)

                comparison = (pos_perpelexity_per_instance < neg_perpelexity_per_instance).float()
                correct += comparison.sum(-1).item()
                total += comparison.shape[0]
                sys.stdout.write('finished {}/{} accuracy {} \r'.format(idx, dataset.test_len(), correct / total))
        print('total accuracy = {}'.format(correct / total))

    if args.do_verify_challenge:
        dataset = GPTTableDatabase(None, None, 'challenge/blind_test_lm_pos_neg.json', tokenizer, args.batch_size, args.max_len)
        model.load_state_dict(torch.load(args.load_from))
        model.eval()
        correct, total = 0, 0
        results = {}
        with torch.no_grad():
            for idx in range(0, dataset.test_len()):
                batch_pos, batch_neg = dataset.get_pair_data(idx, 'test')

                table_name = dataset.get_item(idx, 'test')
                results[table_name] = []

                batch = tuple(Variable(t).to(device) for t in batch_pos)
                trg_inp, trg_out, mask, caption = batch

                inputs = torch.cat([caption, trg_inp], 1)

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
                loss = loss.reshape(logits.shape[0], -1)
                loss_per_instance = (loss * mask).sum(1) / mask.sum(1)
                pos_perpelexity_per_instance = torch.exp(loss_per_instance.cpu().data).tolist()

                batch = tuple(Variable(t).to(device) for t in batch_neg)
                trg_inp, trg_out, mask, caption = batch

                inputs = torch.cat([caption, trg_inp], 1)

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
                loss = loss.reshape(logits.shape[0], -1)
                loss_per_instance = (loss * mask).sum(1) / mask.sum(1)
                neg_perpelexity_per_instance = torch.exp(loss_per_instance.cpu().data).tolist()

                for p1, p2 in zip(pos_perpelexity_per_instance, neg_perpelexity_per_instance):
                    if p1 < p2:
                        results[table_name].append('unknown1')
                    else:
                        results[table_name].append('unknown2')

                sys.stdout.write('finished {}/{}\r'.format(idx, dataset.test_len()))
        
        with open('challenge/verify_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    if args.do_rl:
        def assemble_distribute(GPT_tokens, rewards, tokenizer, bert_tokenizer):
            GPT_tokens = tokenizer.convert_ids_to_tokens(GPT_tokens)
            gpt_mapping = []
            count = 0
            for i, x in enumerate(GPT_tokens):
                if x[0] == '\u0120' or i == 0:
                    gpt_mapping.append(count)
                    count += 1
                else:
                    count -= 1
                    gpt_mapping.append(count)
                    count += 1
            sentence = tokenizer.convert_tokens_to_string(GPT_tokens)
            ids = bert_tokenizer.tokenize(sentence)
            bert_mapping = []
            count = 0
            for i, x in enumerate(ids):
                if x.startswith('##'):
                    count -= 1
                    bert_mapping.append(count)
                    count += 1
                else:
                    bert_mapping.append(count)
                    count += 1
            # start calculating rewards
            sent_rewards = []
            tmp = []
            for i, x in enumerate(bert_mapping):
                if i > 0 and x != bert_mapping[i - 1]:
                    sent_rewards.append(sum(tmp) / len(tmp))
                    tmp = [rewards[i]]
                else:
                    tmp.append(rewards[i])
            sent_rewards.append(sum(tmp) / len(tmp))

            token_rewards = []
            for _ in gpt_mapping:
                token_rewards.append(sent_rewards[_])

            return token_rewards

        model.load_state_dict(torch.load(args.load_from))
        print("loading from {}".format(args.load_from))
        model.train()

        bert_tokenizer = BertTokenizer.from_pretrained(args.modelpath)
        scorer = BERTGen(bert_tokenizer.vocab_size, args.dim, args.layers, args.head, args.modelpath)
        scorer.to(args.device)
        scorer.load_state_dict(torch.load('models/BERT_scorer_ep9.pt'))
        scorer.eval()

        optimizer = optim.Adam(model.parameters(), 5e-7)

        avg_loss = 0
        for epoch_idx in range(args.epoch):
            print("start training {}th epoch".format(epoch_idx))
            dataset.shuffle()
            for idx in range(0, dataset.train_len()):
                batch = dataset.get_data(idx, details=True)
                table, sub_columns, title = batch[4:]

                batch = tuple(Variable(t).to(device) for t in batch[:4])
                trg_inp, trg_out, mask, caption = batch

                if (idx + 1) % 2 == 0:
                    # Do RL training
                    samples = sample_sequence(model, 30, caption, [])
                    samples = samples[:, caption.shape[1]:][0]
                    samples = samples.cpu().data.numpy()

                    end = numpy.where(samples == tokenizer.eos_token_id)[0]
                    if len(end) > 0:
                        samples = samples[:end[0]]

                    sentence = tokenizer.decode(samples)

                    e_tokens = bert_tokenizer.tokenize(sentence)
                    desc = linearize_table(table, sub_columns[0], title[0], bert_tokenizer)
                    e_idx = bert_tokenizer.convert_tokens_to_ids(e_tokens)

                    inputs = []
                    outputs = []
                    for i in range(len(e_tokens)):
                        inputs.append(bert_tokenizer.convert_tokens_to_ids(
                            e_tokens[:i] + ['[MASK]'] + e_tokens[i + 1:]))
                        outputs.append([-1] * i + [e_idx[i]] + [-1] * (len(e_tokens) - i - 1))

                    desc = torch.LongTensor(desc).unsqueeze(0).to(device)
                    inputs = torch.LongTensor(inputs).to(device)
                    outputs = torch.LongTensor(outputs).to(device)

                    with torch.no_grad():
                        logits = scorer(inputs, desc)
                        loss = criterion(logits.view(-1, logits.shape[-1]),
                                         outputs.view(-1)).view(logits.shape[0], -1).cpu().data
                        prob = torch.exp(-loss)
                        indexes = torch.arange(0, logits.shape[0])[:, None].long()
                        probs = torch.gather(prob, -1, indexes)
                        rewards = assemble_distribute(samples, probs.view(-1).numpy(), tokenizer, bert_tokenizer)

                    rewards = torch.FloatTensor(rewards).to(args.device)
                    #rewards = torch.cat([rewards, torch.FloatTensor([1.0]).to(device)], 0)
                    #rewards = rewards - torch.mean(rewards)
                    rewards = Variable(rewards)

                    samples = torch.from_numpy(samples).to(args.device).unsqueeze(0)
                    # samples = torch.cat([samples, torch.LongTensor(
                    #    [tokenizer.eos_token_id]).to(device)], 0).unsqueeze(0)
                    samples = Variable(samples)

                    inputs = torch.cat([caption, samples], 1)
                    logits = model(inputs)[0]
                    logits = logits[:, -samples.shape[1] - 1:-1, :].contiguous()

                    loss = criterion(logits.view(-1, logits.shape[-1]), samples.view(-1))
                    loss = (loss * rewards).mean()
                else:
                    # Do MLE training
                    inputs = torch.cat([caption, trg_inp], 1)

                    logits = model(inputs)[0]
                    logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                    loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))

                    loss = loss * mask.view(-1)
                    loss = loss.sum() / mask.sum()

                avg_loss += loss.item()

                if (idx + 1) % args.gradient_accumulation_steps == 0:
                    loss.backward()
                    optimizer.step()

                    model.zero_grad()
                    optimizer.zero_grad()

                if (idx + 1) % args.every == 0:
                    #sys.stdout.write('finished {} samples loss = {} \r'.format(idx, avg_loss / 50))
                    print('finished {} samples loss = {}, perpelexity = {}'.format(
                        idx, avg_loss / args.every, math.exp(avg_loss / args.every)))

                    fake_inputs = caption
                    gt_inputs = trg_out.cpu().data.numpy()

                    #samples = model.sample(fake_inputs, tabfeat, caption, highlight_idx, bert)
                    samples = sample_sequence(model, 30, fake_inputs, [])
                    samples = samples[:, caption.shape[1]:]
                    samples = samples.cpu().data.numpy()

                    for s, gt in zip(samples, gt_inputs):
                        text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                        text = text[: text.find(tokenizer.eos_token)]
                        print(text)
                        break

                    avg_loss = 0

            torch.save(model.state_dict(), '{}/GPT_RL_ep{}.pt'.format(args.id, epoch_idx))
