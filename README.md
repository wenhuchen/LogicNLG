# LogicNLG
The data and code for ACL2020 paper "Logical Natural Language Generation from Open-Domain Tables"



# Compute SP-ACC score
```
CUDA_VISIBLE_DEVICES=0 python parse_programs.py --compute_score --load_from parser_models/parser_step49161_acc0.58.pt --score_file outputs/[Your_File]
```
