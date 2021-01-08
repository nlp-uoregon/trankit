import trankit

trainer = trankit.TPipeline(
    training_config={
        'lang': 'english',
        'text_split_by_space': True,
        'task': 'tokenize',
        'save_dir': './saved_models/english',
        'gpu': True,
        'max_epoch': 100,
        'batch_size': 16,
        'max_input_length': 512,
        'train_txt_fpath': '../ud-treebanks-v2.5/UD_English-EWT/en_ewt-ud-train.txt',
        'train_conllu_fpath': '../ud-treebanks-v2.5/UD_English-EWT/en_ewt-ud-train.conllu',
        'dev_txt_fpath': '../ud-treebanks-v2.5/UD_English-EWT/en_ewt-ud-dev.txt',
        'dev_conllu_fpath': '../ud-treebanks-v2.5/UD_English-EWT/en_ewt-ud-dev.conllu',
        'test_txt_fpath': '../ud-treebanks-v2.5/UD_English-EWT/en_ewt-ud-test.txt',
        'test_conllu_fpath': '../ud-treebanks-v2.5/UD_English-EWT/en_ewt-ud-test.conllu',
    }
)

trainer.train()
