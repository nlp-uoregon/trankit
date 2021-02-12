# Building a customized pipeline

*NOTE*: Please check out the list of [supported languages](https://trankit.readthedocs.io/en/latest/pkgnames.html#trainable-languages) to check if the language of your interest is already supported by Trankit.

Building customized pipelines are easy with Trankit. The training is done via the class `TPipeline` and the loading is done via the class `Pipeline` as usual. To achieve this, customized pipelines are currently organized into 4 categories:
- `customized-mwt-ner`: a pipeline of this category consists of 5 models: (i) joint token and sentence splitter, (ii) multi-word token expander, (iii) joint model for part-of-speech tagging, morphologicial feature tagging, and dependency parsing, (iv) lemmatizer, and (v) named entity recognizer.
- `customized-mwt`: a pipeline of this category doesn't have the (v) named entity recognizer.
- `customized-ner`: a pipeline of this category doesn't have the (ii) multi-word token expander.
- `customized`: a pipeline of this cateogry doesn't have the (ii) multi-word token expander and the (v) named entity recognizer.

The category names are used as the special identities in the `Pipeline` class. Thus, we need to choose one of the categories to start training our customized pipelines. Below we show the example for training and loading a `customized-mwt-ner` pipeline. Training procedure for customized pipelines of other categories can be obtained by obmitting the steps related to the models that those pipelines don't have. For data format, `Tpipeline` accepts training data in [CONLL-U format](https://github.com/UniversalDependencies/UD_English-EWT) for the Universal Dependencies tasks, and training data in [BIO format](https://www.clips.uantwerpen.be/conll2003/ner/) for the NER task.

## Training
### Training a joint token and sentence splitter
```python
import trankit

# initialize a trainer for the task
trainer = trankit.TPipeline(
    training_config={
    'category': 'customized-mwt-ner', # pipeline category
    'task': 'tokenize', # task name
    'save_dir': './save_dir', # directory for saving trained model
    'train_txt_fpath': './train.txt', # raw text file
    'train_conllu_fpath': './train.conllu', # annotations file in CONLLU format for training
    'dev_txt_fpath': './dev.txt', # raw text file
    'dev_conllu_fpath': './dev.conllu' # annotations file in CONLLU format for development
    }
)

# start training
trainer.train()
```

### Training a multi-word token expander
```python
import trankit

# initialize a trainer for the task
trainer = trankit.TPipeline(
    training_config={
    'category': 'customized-mwt-ner', # pipeline category
    'task': 'mwt', # task name
    'save_dir': './save_dir', # directory for saving trained model
    'train_conllu_fpath': './train.conllu', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': './dev.conllu' # annotations file in CONLLU format for development
    }
)

# start training
trainer.train()
```

### Training a joint model for part-of-speech tagging, morphologicial feature tagging, and dependency parsing
```python
import trankit

# initialize a trainer for the task
trainer = trankit.TPipeline(
    training_config={
    'category': 'customized-mwt-ner', # pipeline category
    'task': 'posdep', # task name
    'save_dir': './save_dir', # directory for saving trained model
    'train_conllu_fpath': './train.conllu', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': './dev.conllu' # annotations file in CONLLU format for development
    }
)

# start training
trainer.train()
```

### Training a lemmatizer
```python
import trankit

# initialize a trainer for the task
trainer = trankit.TPipeline(
    training_config={
    'category': 'customized-mwt-ner',  # pipeline category
    'task': 'lemmatize', # task name
    'save_dir': './save_dir', # directory for saving trained model
    'train_conllu_fpath': './train.conllu', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': './dev.conllu' # annotations file in CONLLU format for development
    }
)

# start training
trainer.train()
```

### Training a named entity recognizer
Training data for a named entity recognizer should be in the BIO format in which the first column contains the words and the last column contains the BIO annotations. Users can refer to [this file](https://github.com/nlp-uoregon/trankit/tree/master/docs/source/sample-data.bio) to get the sample data in the required format.

```python
import trankit

# initialize a trainer for the task
trainer = trankit.TPipeline(
    training_config={
    'category': 'customized-mwt-ner',  # pipeline category
    'task': 'ner', # task name
    'save_dir': './save_dir', # directory to save the trained model
    'train_bio_fpath': './train.bio', # training data in BIO format
    'dev_bio_fpath': './dev.bio' # training data in BIO format
    }
)

# start training
trainer.train()
`````
A colab tutorial on how to train, evaluate, and use a custom NER model is also available [here](https://github.com/nlp-uoregon/trankit/blob/master/examples/colab/trankit_ner_GermEval14.ipynb). Thanks [@mrshu](https://github.com/mrshu) for contributing this to Trankit.

## Loading
After the training steps, we need to verify that all the models of our customized pipeline are trained. The following code shows how to do it:
```python
import trankit

trankit.verify_customized_pipeline(
    category='customized-mwt-ner', # pipeline category
    save_dir='./save_dir' # directory used for saving models in previous steps
)
```
If the verification is success, this would printout the following:
```python
Customized pipeline is ready to use!
It can be initialized as follows:

from trankit import Pipeline
p = Pipeline(lang='customized-mwt-ner', cache_dir='./save_dir')
```
From now on, the customized pipeline can be used as a normal pretrained pipeline.
