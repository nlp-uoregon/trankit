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

Trankit is a **light-weight Transformer-based Python** Toolkit for multilingual Natural Language Processing (NLP). It provides a trainable pipeline for fundamental NLP tasks over [100 languages](https://trankit.readthedocs.io/en/latest/pkgnames.html#trainable-languages), and 90 [downloadable](https://trankit.readthedocs.io/en/latest/pkgnames.html#pretrained-languages-their-code-names) pretrained pipelines for [56 languages](https://trankit.readthedocs.io/en/latest/pkgnames.html#pretrained-languages-their-code-names). Trankit can process inputs which are untokenized (raw) or pretokenized strings, at
both sentence and document level. Currently, Trankit supports the following tasks:
- Sentence segmentation.
- Tokenization.
- Multi-word token expansion.
- Part-of-speech tagging.
- Morphological feature tagging.
- Dependency parsing.
- Named entity recognition.

Built on the state-of-the-art multilingual pretrained transformer [XLM-Roberta](https://arxiv.org/abs/1911.02116), Trankit significantly *outperforms* prior multilingual NLP pipelines (e.g., UDPipe, Stanza) in many tasks over 90 [Universal Dependencies v2.5 treebanks](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3105) while still being efficient in memory usage and
speed, making it *usable for general users*. Below is the performance comparison between Trankit and other NLP toolkits on Arabic, Chinese, and English.

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Treebank</th>
    <th class="tg-0pky">System</th>
    <th class="tg-0pky">Tokens</th>
    <th class="tg-0pky">Sents.</th>
    <th class="tg-0pky">Words</th>
    <th class="tg-0pky">UPOS</th>
    <th class="tg-0pky">XPOS</th>
    <th class="tg-0pky">UFeats</th>
    <th class="tg-0pky">Lemmas</th>
    <th class="tg-0pky">UAS</th>
    <th class="tg-0pky">LAS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="2"><br>Overall (90 treebanks)</td>
    <td class="tg-0pky">Trankit</td>
    <td class="tg-c3ow">99.23</td>
    <td class="tg-7btt"><b>91.82</b></td>
    <td class="tg-7btt"><b>99.02</b></td>
    <td class="tg-7btt"><b>95.65</b></td>
    <td class="tg-7btt"><b>94.05</b></td>
    <td class="tg-7btt"><b>93.21</b></td>
    <td class="tg-7btt"><b>94.27</b></td>
    <td class="tg-7btt"><b>87.06</b></td>
    <td class="tg-7btt"><b>83.69</b></td>
  </tr>
  <tr>
    <td class="tg-0pky">Stanza</td>
    <td class="tg-7btt"><b>99.26</b></td>
    <td class="tg-c3ow">88.58</td>
    <td class="tg-c3ow">98.90</td>
    <td class="tg-c3ow">94.21</td>
    <td class="tg-c3ow">92.50</td>
    <td class="tg-c3ow">91.75</td>
    <td class="tg-c3ow">94.15</td>
    <td class="tg-c3ow">83.06</td>
    <td class="tg-c3ow">78.68</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="3"><br><br>Arabic-PADT<br></td>
    <td class="tg-0pky">Trankit</td>
    <td class="tg-c3ow">99.93</td>
    <td class="tg-7btt"><b>96.59</b></td>
    <td class="tg-7btt"><b>99.22</b></td>
    <td class="tg-7btt"><b>96.31</b></td>
    <td class="tg-7btt"><b>94.08</b></td>
    <td class="tg-7btt"><b>94.28</b></td>
    <td class="tg-7btt"><b>94.65</b></td>
    <td class="tg-7btt"><b>88.39</b></td>
    <td class="tg-7btt"><b>84.68</b></td>
  </tr>
  <tr>
    <td class="tg-0pky">Stanza</td>
    <td class="tg-7btt"><b>99.98</b></td>
    <td class="tg-c3ow">80.43</td>
    <td class="tg-c3ow">97.88</td>
    <td class="tg-c3ow">94.89</td>
    <td class="tg-c3ow">91.75</td>
    <td class="tg-c3ow">91.86</td>
    <td class="tg-c3ow">93.27</td>
    <td class="tg-c3ow">83.27</td>
    <td class="tg-c3ow">79.33</td>
  </tr>
  <tr>
    <td class="tg-0pky">UDPipe</td>
    <td class="tg-c3ow">99.98</td>
    <td class="tg-c3ow">82.09</td>
    <td class="tg-c3ow">94.58</td>
    <td class="tg-c3ow">90.36</td>
    <td class="tg-c3ow">84.00</td>
    <td class="tg-c3ow">84.16</td>
    <td class="tg-c3ow">88.46</td>
    <td class="tg-c3ow">72.67</td>
    <td class="tg-c3ow">68.14</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="3"><br><br>Chinese-GSD</td>
    <td class="tg-0pky">Trankit</td>
    <td class="tg-7btt"><b>97.01</b></td>
    <td class="tg-7btt"><b>99.70</b></td>
    <td class="tg-7btt"><b>97.01</b></td>
    <td class="tg-7btt"><b>94.21</b></td>
    <td class="tg-7btt"><b>94.02</b></td>
    <td class="tg-7btt"><b>96.59</b></td>
    <td class="tg-7btt"><b>97.01</b></td>
    <td class="tg-7btt"><b>85.19</b></td>
    <td class="tg-7btt"><b>82.54</b></td>
  </tr>
  <tr>
    <td class="tg-0pky">Stanza</td>
    <td class="tg-c3ow">92.83</td>
    <td class="tg-c3ow">98.80</td>
    <td class="tg-c3ow">92.83</td>
    <td class="tg-c3ow">89.12</td>
    <td class="tg-c3ow">88.93</td>
    <td class="tg-c3ow">92.11</td>
    <td class="tg-c3ow">92.83</td>
    <td class="tg-c3ow">72.88</td>
    <td class="tg-c3ow">69.82</td>
  </tr>
  <tr>
    <td class="tg-0pky">UDPipe</td>
    <td class="tg-c3ow">90.27</td>
    <td class="tg-c3ow">99.10</td>
    <td class="tg-c3ow">90.27</td>
    <td class="tg-c3ow">84.13</td>
    <td class="tg-c3ow">84.04</td>
    <td class="tg-c3ow">89.05</td>
    <td class="tg-c3ow">90.26</td>
    <td class="tg-c3ow">61.60</td>
    <td class="tg-c3ow">57.81</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4"><br><br><br>English-EWT</td>
    <td class="tg-0pky">Trankit</td>
    <td class="tg-c3ow">98.48</td>
    <td class="tg-7btt"><b>88.35</b></td>
    <td class="tg-c3ow">98.48</td>
    <td class="tg-7btt"><b>95.95</b></td>
    <td class="tg-7btt"><b>95.71</b></td>
    <td class="tg-7btt"><b>96.26</b></td>
    <td class="tg-7btt">96.84</td>
    <td class="tg-7btt"><b>90.14</b></td>
    <td class="tg-7btt"><b>87.96</b></td>
  </tr>
  <tr>
    <td class="tg-0pky">Stanza</td>
    <td class="tg-7btt"><b>99.01</b></td>
    <td class="tg-c3ow">81.13</td>
    <td class="tg-7btt"><b>99.01</b></td>
    <td class="tg-c3ow">95.40</td>
    <td class="tg-c3ow">95.12</td>
    <td class="tg-c3ow">96.11</td>
    <td class="tg-c3ow"><b>97.21</b></td>
    <td class="tg-c3ow">86.22</td>
    <td class="tg-c3ow">83.59</td>
  </tr>
  <tr>
    <td class="tg-0pky">UDPipe</td>
    <td class="tg-c3ow">98.90</td>
    <td class="tg-c3ow">77.40</td>
    <td class="tg-c3ow">98.90</td>
    <td class="tg-c3ow">93.26</td>
    <td class="tg-c3ow">92.75</td>
    <td class="tg-c3ow">94.23</td>
    <td class="tg-c3ow">95.45</td>
    <td class="tg-c3ow">80.22</td>
    <td class="tg-c3ow">77.03</td>
  </tr>
  <tr>
    <td class="tg-0pky">spaCy</td>
    <td class="tg-c3ow">97.30</td>
    <td class="tg-c3ow">61.19</td>
    <td class="tg-c3ow">97.30</td>
    <td class="tg-c3ow">86.72</td>
    <td class="tg-c3ow">90.83</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">87.05</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</tbody>
</table>


Performance comparison between Trankit and these toolkits on other languages can be found [here](https://trankit.readthedocs.io/en/latest/performance.html#universal-dependencies-v2-5) on our documentation page.

We also created a Demo Website for Trankit, which is hosted at: http://nlp.uoregon.edu/trankit

Technical details about Trankit are presented in [our following paper](). Please cite the paper if you use Trankit in your software or research.

```bibtex
@article{unset,
  title={Trankit: A Light-Weight Transformer-based Toolkit for Multilingual Natural Language Processing},
  author={unset},
  journal={arXiv preprint arXiv:},
  year={2021}
}
```


### Installation
Trankit can be easily installed via one of the following methods:
#### Using pip
```
pip install trankit
```
The command would install Trankit and all dependent packages automatically.

#### From source
```
git clone https://github.com/nlp-uoregon/trankit.git
cd trankit
pip install -e .
```
This would first clone our github repo and automatically install Trankit.

### Quick Examples

#### Initialize a pretrained pipeline
The following code shows how to initialize a pretrained pipeline for English; it is instructed to run on GPU, automatically downloaded pretrained models and and store them to the specified cache directory. Trankit will not download pretrained models if they already exist.
```python
from trankit import Pipeline

# initialize a multilingual pipeline
p = Pipeline(lang='english', gpu=True, cache_dir='./cache')
```

#### Basic functions
Trankit can process inputs which are untokenized (raw) or pretokenized strings, at both sentence and document level. A pretokenized input can be a list of strings (i.e., a tokenized sentence) or a list of lists of strings (i.e., a tokenized document with multiple tokenized sentences) which is automatically recognized by Trankit. If the input is a sentence, the tag `is_sent` must be set to True. 
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

# perform separate tasks on the input
sents = p.ssplit(untokenized_doc) # sentence segmentation
tokenized_doc = p.tokenize(untokenized_doc) # sentence segmentation and tokenization
tokenized_sent = p.tokenize(untokenized_sent, is_sent=True) # tokenization only
posdeps = p.posdep(untokenized_doc) # upos, xpos, ufeats, dep parsing
ners = p.ner(untokenized_doc) # ner tagging
lemmas = p.lemmatize(untokenized_doc) # lemmatization
```
Note that, although pretokenized inputs can always be processed, using pretokenized inputs for languages that require multi-word token expansion such as Arabic or French might not be the correct way. Please check out the column `Requires MWT expansion` of [this table](https://trankit.readthedocs.io/en/latest/pkgnames.html#pretrained-languages-their-code-names) to see if a particular language requires multi-word token expansion or not.  
For more detailed examples, please checkout our [documentation page](https://trankit.readthedocs.io/en/latest/overview.html).

#### Multilingual usage
In case we want to process inputs of different languages, we need to initialize a multilingual pipeline. The following code shows an example for initializing a multilingual pipeline for Arabic, Chinese, Dutch, and English.
```python
from trankit import Pipeline

# initialize a multilingual pipeline
p = Pipeline(lang='english', gpu=True, cache_dir='./cache')

langs = ['arabic', 'chinese', 'dutch']
for lang in langs:
    p.add(lang)

# tokenize English input
p.set_active('english')
en = p.tokenize('Rich was here before the scheduled time.')

# get ner tags for Arabic input
p.set_active('arabic')
ar = p.ner('وكان كنعان قبل ذلك رئيس جهاز الامن والاستطلاع للقوات السورية العاملة في لبنان.')
```
In this example, `.set_active()` is used to switch between languages.

### Training your own pipelines
Training customized pipelines is easy with Trankit via the class `TPipeline`. Below is a code for training a token and sentence splitter with Trankit.
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
Detailed guidelines for training customized pipelines can be found [here](https://trankit.readthedocs.io/en/latest/training.html) 

### Acknowledgements
We use the [AdapterHub](https://github.com/Adapter-Hub/adapter-transformers) to implement our plug-and-play mechanism with Adapters. To speed up the development process, the implementations for the MWT expander and the lemmatizer are adapted from [Stanza](https://github.com/stanfordnlp/stanza).
