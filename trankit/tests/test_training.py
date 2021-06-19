import trankit

# initialize a trainer for the task
trainer = trankit.TPipeline(
    training_config={
    'category': 'customized-ner',  # pipeline category
    'task': 'ner', # task name
    'save_dir': './save_dir', # directory to save the trained model
    'train_bio_fpath': './train.bio', # training data in BIO format
    'dev_bio_fpath': './dev.bio', # training data in BIO format
    'max_epoch': 1
    }
)

# start training
trainer.train()

trankit.download_missing_files(
    category='customized-ner',
    save_dir='./save_dir',
    embedding_name='xlm-roberta-base',
    language='english'
)

trankit.verify_customized_pipeline(
    category='customized-ner', # pipeline category
    save_dir='./save_dir', # directory used for saving models in previous steps
    embedding_name='xlm-roberta-base'
)

p = trankit.Pipeline(lang='customized-ner', cache_dir='./save_dir')

print(trankit.trankit2conllu(p('I love you more than I can say. Do you know that?')))
