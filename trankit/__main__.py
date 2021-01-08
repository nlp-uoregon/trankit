from .pipeline import Pipeline
from .pipeline import treebank2lang, get_ud_score, get_ud_performance_table
import sys, os
from time import time
from datetime import datetime

# evaluate pretrained pipelines
if len(sys.argv) < 4:
    print(
        'Not enough specified arguments! Please use the command, for example: python -m trankit UD_English-EWT test.input-text.txt test.gold.conllu $device\nwhere $device=gpu or $device=cpu')

tbname = sys.argv[1]
input_fpath = sys.argv[2]
gold_conllu = sys.argv[3]
tblang = treebank2lang[tbname]

pretrained_pipeline = Pipeline(tblang)
pretrained_pipeline._ud_eval = True
start_time = time()
pred_conllu = pretrained_pipeline._conllu_predict(input_fpath)
print(get_ud_performance_table(get_ud_score(pred_conllu, gold_conllu)))
