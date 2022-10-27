

import trankit
import os
tokenizer = trankit.TPipeline(
    training_config={
    'category': 'customized', # pipeline category
    'task': 'tokenize', # task name
    'save_dir': './save_dir/hd', # directory for saving trained model
    'train_txt_fpath': './hindi_kannada_train.txt', # raw text file
    'dev_txt_fpath': './hindi_kannada_dev.txt', # raw text file
    'train_conllu_fpath': './hindi_kannada_train.dat', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': './hindi_kannada_dev.dat',
    'max_epoch': 10}
)

# start training
tokenizer.train()

lemmatizer = trankit.TPipeline(
    training_config={
    'category': 'customized', # pipeline category
    'task': 'lemmatize', # task name
    'save_dir': './save_dir/hd', # directory for saving trained model
    'train_conllu_fpath': './hindi_kannada_train.dat', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': './hindi_kannada_dev.dat',
    'max_epoch': 10}
)

# start training
lemmatizer.train()


trainer = trankit.TPipeline(
    training_config={
    'category': 'customized', # pipeline category
    'task': 'posdep', # task name
    'save_dir': './save_dir/hd', # directory for saving trained model
    'train_conllu_fpath': './hindi_kannada_train.dat', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': './hindi_kannada_dev.dat',
    'max_epoch': 10}
)

# start training
trainer.train()
import pickle as pkl
from trankit.iterators.tagger_iterators import TaggerDataset




test_set = TaggerDataset(
    config=trainer._config,
    input_conllu="./kannada_test.dat",
    gold_conllu="./kannada_test.dat",
    evaluate=True
)
test_set.numberize()
test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
result = trainer._eval_posdep(data_set=test_set, batch_num=test_batch_num,
                           name='test', epoch=-1)


for i in result[0]:
  try:
    print(i,result[0][i].f1)
  except:
    pass


