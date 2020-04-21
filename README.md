# LogicNLG
The data and code for ACL2020 paper "Logical Natural Language Generation from Open-Domain Tables"


## Download the NLI scorer
```
wget https://logicnlg.s3-us-west-2.amazonaws.com/NLI_models.zip
unzip NLI_models.zip
```

## Compute SP-ACC score
```
CUDA_VISIBLE_DEVICES=0 python parse_programs.py --compute_score --load_from parser_models/parser_step49161_acc0.58.pt --score_file outputs/[Your_File]
```
## Compute NLI-ACC score
```
CUDA_VISIBLE_DEVICES=0 python gnn.py --model bert-base-multilingual-uncased --do_verify --no_numeric --encoding gnn --load_from models/gnn_fp16_no_numeric/model_ep4.pt --fp16 --verify_file outputs/[Your_File] --verify_linking verification/test_lm.json
```
