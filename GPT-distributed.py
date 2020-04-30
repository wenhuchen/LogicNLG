import argparse
import logging
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
from tqdm import tqdm, trange
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader as DL
import torch
from torch.utils.data.distributed import DistributedSampler

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
    parser.add_argument('--batch_size', default=6, type=int, help="whether to train or test the model")
    parser.add_argument('--local_rank', default=-1, type=int, help="whether to train or test the model")
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

    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    #model = nn.DataParallel(model)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    if args.do_train:
        if args.local_rank in [-1, 0]:
            if not os.path.exists(args.id):
                os.mkdir(args.id)
            tb_writer = SummaryWriter(log_dir='tensorboard/GPT2-{}'.format(args.model))
        
        dataset = GPTTableDataset2('data/train_lm_preprocessed.json', tokenizer, args.max_len)
        
        if args.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        
        train_dataloader = DL(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=0)

        model.train()
        optimizer = optim.Adam(model.parameters(), args.learning_rate)

        avg_loss = 0
        global_step = 0

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], 
                                        output_device=args.local_rank, find_unused_parameters=True)
        else:
            model = torch.nn.DataParallel(model)

        for epoch_idx in trange(0, args.epoch, desc='Epoch', disable=args.local_rank not in [-1, 0]):
            #for idx in range(0, dataset.train_len()):
            for idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
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

                if args.local_rank in [-1, 0] and idx % args.every == 0 and idx > 0:
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

            if args.local_rank in [-1, 0]:
                if args.model == 'gpt2':
                    torch.save(model.state_dict(), '{}/GPT_ep{}.pt'.format(args.id, epoch_idx))
                else:
                    torch.save(model.state_dict(), '{}/GPT_medium_ep{}.pt'.format(args.id, epoch_idx))
        
        if args.local_rank in [-1, 0]:
            tb_writer.close()