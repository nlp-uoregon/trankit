<h2 align="center">Trankit: A Light-Weight Transformer-based Python Toolkit for Multilingual Natural Language Processing</h2>

<div align="center">
    <a href="https://github.com/nlp-uoregon/trankit/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/nlp-uoregon/trankit.svg?color=blue">
    </a>
    <a href='https://trankit.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/trankit/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="http://nlp.uoregon.edu/trankit">
        <img alt="Demo Website" src="https://img.shields.io/website/http/trankit.readthedocs.io/en/latest/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://pypi.org/project/trankit/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/trankit?color=blue">
    </a>
    <a href="https://pypi.org/project/trankit/">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/trankit?colorB=blue">
    </a>
</div>

Trankit is a **light-weight Transformer-based Python** Toolkit for multilingual Natural Language Processing (NLP). It provides a trainable pipeline for fundamental NLP tasks over [100 languages](https://trankit.readthedocs.io/en/latest/pkgnames.html#trainable-languages), and 90 [downloadable](https://trankit.readthedocs.io/en/latest/pkgnames.html#pretrained-languages-their-code-names) pretrained pipelines for [56 languages](https://trankit.readthedocs.io/en/latest/pkgnames.html#pretrained-languages-their-code-names).

<div align="center"><img src="https://raw.githubusercontent.com/nlp-uoregon/trankit/master/docs/source/architecture.jpg" height="300px"/></div>

**Trankit outperforms the current state-of-the-art multilingual toolkit Stanza (StanfordNLP)** in many tasks over [90 Universal Dependencies v2.5 treebanks of 56 different languages](https://trankit.readthedocs.io/en/latest/performance.html#universal-dependencies-v2-5) while still being efficient in memory usage and
speed, making it *usable for general users*.

In particular, for **English**, **Trankit is significantly better than Stanza** on sentence segmentation (**+7.22%**) and dependency parsing (**+3.92%** for UAS and **+4.37%** for LAS). For **Arabic**, our toolkit substantially improves sentence segmentation performance by **16.16%** while **Chinese** observes **12.31%** and **12.72%** improvement of UAS and LAS for dependency parsing. Detailed comparison between Trankit, Stanza, and other popular NLP toolkits (i.e., spaCy, UDPipe) in other languages can be found [here](https://trankit.readthedocs.io/en/latest/performance.html#universal-dependencies-v2-5) on [our documentation page](https://trankit.readthedocs.io/en/latest/index.html).

We also created a Demo Website for Trankit, which is hosted at: http://nlp.uoregon.edu/trankit

Technical details about Trankit are presented in [our following paper](https://arxiv.org/pdf/2101.03289.pdf). Please cite the paper if you use Trankit in your research.

```bibtex
@misc{nguyen2021trankit,
      title={Trankit: A Light-Weight Transformer-based Toolkit for Multilingual Natural Language Processing}, 
      author={Minh Nguyen and Viet Lai and Amir Pouran Ben Veyseh and Thien Huu Nguyen},
      year={2021},
      eprint={2101.03289},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


### Installation
Trankit can be easily installed via one of the following methods:
#### Using pip
```
pip install trankit
```
The command would install Trankit and all dependent packages automatically. 
Note that, due to [this issue](https://github.com/nlp-uoregon/trankit/issues/3) relating to `adapter-transformers`, users may need to uninstall `transformers` before installing trankit.

#### From source
```
git clone https://github.com/nlp-uoregon/trankit.git
cd trankit
pip install -e .
```
This would first clone our github repo and install Trankit.

### Usage
Trankit can process inputs which are untokenized (raw) or pretokenized strings, at
both sentence and document level. Currently, Trankit supports the following tasks:
- Sentence segmentation.
- Tokenization.
- Multi-word token expansion.
- Part-of-speech tagging.
- Morphological feature tagging.
- Dependency parsing.
- Named entity recognition.
#### Initialize a pretrained pipeline
The following code shows how to initialize a pretrained pipeline for English; it is instructed to run on GPU, automatically download pretrained models, and store them to the specified cache directory. Trankit will not download pretrained models if they already exist.
```python
from trankit import Pipeline

# initialize a multilingual pipeline
p = Pipeline(lang='english', gpu=True, cache_dir='./cache')
```

#### Perform all tasks on the input
After initializing a pretrained pipeline, it can be used to process the input on all tasks as shown below. If the input is a sentence, the tag `is_sent` must be set to True. 
```python
from trankit import Pipeline

p = Pipeline(lang='english', gpu=True, cache_dir='./cache')

######## document-level processing ########
untokenized_doc = '''Hello! This is Trankit.'''
pretokenized_doc = [['Hello', '!'], ['This', 'is', 'Trankit', '.']]

# perform all tasks on the input
processed_doc1 = p(untokenized_doc)
processed_doc2 = p(pretokenized_doc)

######## sentence-level processing ####### 
untokenized_sent = '''This is Trankit.'''
pretokenized_sent = ['This', 'is', 'Trankit', '.']

# perform all tasks on the input
processed_sent1 = p(untokenized_sent, is_sent=True)
processed_sent2 = p(pretokenized_sent, is_sent=True)
```
Note that, although pretokenized inputs can always be processed, using pretokenized inputs for languages that require multi-word token expansion such as Arabic or French might not be the correct way. Please check out the column `Requires MWT expansion?` of [this table](https://trankit.readthedocs.io/en/latest/pkgnames.html#pretrained-languages-their-code-names) to see if a particular language requires multi-word token expansion or not.  
For more detailed examples, please check out our [documentation page](https://trankit.readthedocs.io/en/latest/overview.html).

#### Multilingual usage
In case we want to process inputs of different languages, we need to initialize a multilingual pipeline.
```python
from trankit import Pipeline

# initialize a multilingual pipeline
p = Pipeline(lang='english', gpu=True, cache_dir='./cache')

langs = ['arabic', 'chinese', 'dutch']
for lang in langs:
    p.add(lang)

# tokenize an English input
p.set_active('english')
en = p.tokenize('Rich was here before the scheduled time.')

# get ner tags for an Arabic input
p.set_active('arabic')
ar = p.ner('وكان كنعان قبل ذلك رئيس جهاز الامن والاستطلاع للقوات السورية العاملة في لبنان.')
```
In this example, `.set_active()` is used to switch between languages.

#### Building a customized pipeline
Training customized pipelines is easy with Trankit via the class `TPipeline`. Below we show how we can train a token and sentence splitter on customized data.
```python
from trankit import TPipeline

tp = TPipeline(training_config={
    'task': 'tokenize',
    'save_dir': './saved_model',
    'train_txt_fpath': './train.txt',
    'train_conllu_fpath': './train.conllu',
    'dev_txt_fpath': './dev.txt',
    'dev_conllu_fpath': './dev.conllu'
    }
)

trainer.train()
```
Detailed guidelines for training and loading a customized pipeline can be found [here](https://trankit.readthedocs.io/en/latest/training.html) 

### To-do list
- Language Identification

### Acknowledgements
We use [XLM-Roberta](https://arxiv.org/abs/1911.02116) and [Adapters](https://arxiv.org/abs/2005.00247) as our shared multilingual encoder for different tasks and languages. The [AdapterHub](https://github.com/Adapter-Hub/adapter-transformers) is used to implement our plug-and-play mechanism with Adapters. To speed up the development process, the implementations for the MWT expander and the lemmatizer are adapted from [Stanza](https://github.com/stanfordnlp/stanza).
