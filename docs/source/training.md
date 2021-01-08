# Building customized pipelines

## Training customized models
Training your own pipelines is easy with *trankit*. This is done via the class `TPipeline` while `Pipeline` is used for evaluating only. The training pipeline `TPipeline` works with training data of [CONLLU](https://universaldependencies.org/format.html) format and [BIO2](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) format. Below are some examples:

*NOTE*: Please check out the list of [supported languages](https://trankit.readthedocs.io/en/latest/pkgnames.html#trainable-languages) to make sure that the language of your training data is supported by Trankit.

### Training a token and sentence splitter
```python
import trankit

# initialize a Training Pipeline
trainer = trankit.TPipeline(
    training_config={
    'task': 'tokenize',
    'save_dir': 'path/to/your/save/dir',
    'train_txt_fpath': 'path/to/your/train-txt-file', # raw text file
    'train_conllu_fpath': 'path/to/your/train-conllu-file', # annotations for raw text file in CONLLU format
    'dev_txt_fpath': 'path/to/your/dev-txt-file', # raw text file
    'dev_conllu_fpath': 'path/to/your/dev-conllu-file' # annotations for raw text file in CONLLU format
    }
)

# start training
trainer.train()
```
### Training a multi-Word token expander
```python
import trankit

# initialize a Training Pipeline
trainer = trankit.TPipeline(
    training_config={
    'task': 'mwt',
    'save_dir': 'path/to/your/save/dir',
    'train_conllu_fpath': 'path/to/your/train-conllu-file',
    'dev_conllu_fpath': 'path/to/your/dev-conllu-file'
    }
)

# start training
trainer.train()
```
### Training the joint model for part-of-speech, morphological tagging and dependency parsing
```python
import trankit

# initialize a Training Pipeline
trainer = trankit.TPipeline(
    training_config={
    'task': 'posdep',
    'save_dir': 'path/to/your/save/dir',
    'train_conllu_fpath': 'path/to/your/train-conllu-file',
    'dev_conllu_fpath': 'path/to/your/dev-conllu-file'
    }
)

# start training
trainer.train()
```
### Training a lemmatizer
```python
import trankit

# initialize a Training Pipeline
trainer = trankit.TPipeline(
    training_config={
    'task': 'tokenize',
    'save_dir': 'path/to/your/save/dir',
    'train_txt_fpath': 'path/to/your/train-txt-file',
    'train_conllu_fpath': 'path/to/your/train-conllu-file',
    'dev_txt_fpath': 'path/to/your/dev-txt-file',
    'dev_conllu_fpath': 'path/to/your/dev-conllu-file'
    }
)

# start training
trainer.train()
```

### Training a named entity recognition tagger
```python
import trankit

# initialize a Training Pipeline
trainer = trankit.TPipeline(
    training_config={
    'task': 'ner',
    'save_dir': 'path/to/your/save/dir',
    'train_bio2_fpath': 'path/to/your/train-BIO2-file',
    'dev_bio2_fpath': 'path/to/your/dev-BIO2-file'
    }
)

# start training
trainer.train()
`````

## Loading customized pipelines
