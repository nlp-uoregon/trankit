from .pipeline import Pipeline
from .pipeline import treebank2lang, get_ud_score, get_ud_performance_table
import sys, os
from time import time
import glob

tbname = sys.argv[1]
input_fpath = glob.glob(os.path.join('ud-treebanks-v2.5', tbname, '*-test.txt'))[0]
gold_conllu = glob.glob(os.path.join('ud-treebanks-v2.5', tbname, '*-test.conllu'))[0]
tblang = treebank2lang[tbname]

pretrained_pipeline = Pipeline(tblang)
pretrained_pipeline._ud_eval = True
start_time = time()
pred_conllu = pretrained_pipeline._conllu_predict(input_fpath)
print(get_ud_performance_table(get_ud_score(pred_conllu, gold_conllu)))
