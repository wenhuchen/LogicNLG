import json
import nltk
import sys
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

assert len(sys.argv) == 3

with open(sys.argv[1], 'r') as f:
    hypothesis = json.load(f)

with open(sys.argv[2], 'r') as f:
    reference = json.load(f)

def get_reference(table_id, option='test'):
    assert option == 'test'
    entry = reference[table_id]
    return [_[0].lower().split(' ') for _ in entry]

def func_compute_bleu(f):
    sent_bleus_1, sent_bleus_2, sent_bleus_3 = [], [], []
    reference = get_reference(f)
    for hyp in hypothesis[f]:
        hyps = hyp.lower().split()        
        sent_bleus_1.append(nltk.translate.bleu_score.sentence_bleu(reference, hyps, weights=(1, 0, 0)))
        sent_bleus_2.append(nltk.translate.bleu_score.sentence_bleu(reference, hyps, weights=(0.5, 0.5, 0)))
        sent_bleus_3.append(nltk.translate.bleu_score.sentence_bleu(reference, hyps, weights=(0.33, 0.33, 0.33)))    
    return sent_bleus_1, sent_bleus_2, sent_bleus_3

assert len(hypothesis) == len(reference)

pool = Pool(64)
bleus = pool.map(func_compute_bleu, hypothesis.keys())

sent_bleus_1, sent_bleus_2, sent_bleus_3 = [], [], []
for _ in bleus:
    sent_bleus_1.extend(_[0])
    sent_bleus_2.extend(_[1])
    sent_bleus_3.extend(_[2])

bleu_1 = sum(sent_bleus_1) / len(sent_bleus_1)
bleu_2 = sum(sent_bleus_2) / len(sent_bleus_2)
bleu_3 = sum(sent_bleus_3) / len(sent_bleus_3)

print("{}/{}/{}".format(bleu_1, bleu_2, bleu_3))
