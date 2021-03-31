# Quick examples

## News: Trankit v1.0.0 is out
### Trankit large
Starting from version 1.0.0, Trankit supports using XLM-Roberta large as the multilingual embedding (i.e., Trankit large), which further boosts the performance over 90 Universal Dependencies treebanks. The usage of the large version is the same as before except that users need to specify the embedding for pipeline initialization. Below is an example for initializing a monolingual and multilingual pipeline.

```python
from trankit import Pipeline 

# Initialize an English pipeline with XLM-Roberta large
p = Pipeline('english', embedding='xlm-roberta-large')

# Initialize a multilingual pipeline for ['english', 'chinese', 'arabic'] with XLM-Roberta large
p = Pipeline('english', embedding='xlm-roberta-large')
p.add('chinese')
p.add('arabic')
```
Currently, the argument `embedding` can only be set to one of the two values `['xlm-roberta-large', 'xlm-roberta-base']`. If the argument `embedding` is not specifically set, Trankit will use `'xlm-roberta-base'` for its multilingual embedding by default.

### Auto mode for multilingual pipelines
Starting from version v1.0.0, Trankit supports a handy *Auto Mode* in which users do not have to set a particular language active before processing the input. In the Auto Mode, Trankit will automatically detect the language of the input and use the corresponding language-specific models, thus avoiding switching back and forth between languages in a multilingual pipeline. Specifically, there are two methods to turn on the Auto Mode.

The first method is to initialize a multilingual pipeline with a special code `'auto'`. After the initialization, the pipeline would be automatically set in the Auto Mode.
```python 
from trankit import Pipeline

p = Pipeline('auto')

# Tokenizing an English input
en_output = p.tokenize('''I figured I would put it out there anyways.''') 

# POS, Morphological tagging and Dependency parsing a French input
fr_output = p.posdep('''On pourra toujours parler à propos d'Averroès de "décentrement du Sujet".''')

# NER tagging a Vietnamese input
vi_output = p.ner('''Cuộc tiêm thử nghiệm tiến hành tại Học viện Quân y, Hà Nội''')
```
Note that, the multilingual pipeline in this case is initialized with all supported languages and all these languages would be considered for the language detection process. In some cases where we want to constrain the detected language to belong to a specific set, the second method is used:
```python
from trankit import Pipeline

p = Pipeline('english')
p.add('french')
p.add('vietnamese')
p.set_auto(True)

# Tokenizing an English input
en_output = p.tokenize('''I figured I would put it out there anyways.''') 

# POS, Morphological tagging and Dependency parsing a French input
fr_output = p.posdep('''On pourra toujours parler à propos d'Averroès de "décentrement du Sujet".''')

# NER tagging a Vietnamese input
vi_output = p.ner('''Cuộc tiêm thử nghiệm tiến hành tại Học viện Quân y, Hà Nội''')
```
In this way, we are guaranteed that the detected language can only be one of the added languages `["english", "french", "vietnamese"]`. Suppose that at some point later, we want to turn off the Auto Mode, this can be done easily with a single line of code as follows:
```python
p.set_auto(False)
```
After this, our multilingual pipeline can be used in the manual mode where we can manually set a particular language active. As a final note, we use [langid](https://github.com/saffsd/langid.py) to perform language detection. The detected language for each input can be inspected by accessing the field `"lang"` of the output.

### Command-line interface
Starting from version v1.0.0, Trankit supports processing text via command-line interface. This helps users who are not familiar with Python programming language can use Trankit more easily. Please check out [this page]() for tutorials and examples.

## Initialize a pipeline
### Monolingual usage
Before using any function of Trankit, we need to initialize a pipeline. Here is how we can do it for English:
```python
from trankit import Pipeline

p = Pipeline('english')
```
In this example, *Trankit* receives the string `'english'` specifying which language package it should use to initialize a pipeline. To know which language packages are supported we can check this [table](https://trankit.readthedocs.io/en/latest/pkgnames.html#pretrained-languages-their-code-names) or directly print out the attribute `trankit.supported_langs`:
```python
import trankit

print(trankit.supported_langs)
# Output: ['afrikaans', 'ancient-greek-perseus', 'ancient-greek', 'arabic', 'armenian', 'basque', 'belarusian', 'bulgarian', 'catalan', 'chinese', 'traditional-chinese', 'classical-chinese', 'croatian', 'czech-cac', 'czech-cltt', 'czech-fictree', 'czech', 'danish', 'dutch', 'dutch-lassysmall', 'english', 'english-gum', 'english-lines', 'english-partut', 'estonian', 'estonian-ewt', 'finnish-ftb', 'finnish', 'french', 'french-partut', 'french-sequoia', 'french-spoken', 'galician', 'galician-treegal', 'german', 'german-hdt', 'greek', 'hebrew', 'hindi', 'hungarian', 'indonesian', 'irish', 'italian', 'italian-partut', 'italian-postwita', 'italian-twittiro', 'italian-vit', 'japanese', 'kazakh', 'korean', 'korean-kaist', 'kurmanji', 'latin', 'latin-perseus', 'latin-proiel', 'latvian', 'lithuanian', 'lithuanian-hse', 'marathi', 'norwegian-nynorsk', 'norwegian-nynorsklia', 'norwegian-bokmaal', 'old-french', 'old-russian', 'persian', 'polish-lfg', 'polish', 'portuguese', 'portuguese-gsd', 'romanian-nonstandard', 'romanian', 'russian-gsd', 'russian', 'russian-taiga', 'scottish-gaelic', 'serbian', 'slovak', 'slovenian', 'slovenian-sst', 'spanish', 'spanish-gsd', 'swedish-lines', 'swedish', 'tamil', 'telugu', 'turkish', 'ukrainian', 'urdu', 'uyghur', 'vietnamese']
```
By default, *trankit* would try to use GPU if a GPU device is available. However, we can force it to run on CPU by setting the tag `gpu=False`:
```python
from trankit import Pipeline

p = Pipeline('english', gpu=False)
```
Another tag that we can use is `cache_dir`. By default, *Trankit* would check if the pretrained model files exist. If they don't, it would download all pretrained files including the shared XLMR-related files and the separate language-related files, then store them to `./cache/trankit`. However, we can change this by setting the tag `cache_dir`:
```python
from trankit import Pipeline

p = Pipeline('english', cache_dir='./path-to-your-desired-location/')
```
### Multilingual usage
Processing multilingual inputs is easy and effective with *Trankit*. For example, to initilize a pipeline that can process inputs of the 3 languages English, Chinese, and Arabic, we can do as follows:
```python
from trankit import Pipeline

p = Pipeline('english')
p.add('chinese')
p.add('arabic')
```
Each time the `add` function is called for a particular language (e.g., `'chinese'` and `'arabic'` in this case), *Trankit* would only download the language-related files. Therefore, the downloading would be very fast. Here is what will show up when the above snippet is executed:
```python 
from trankit import Pipeline

p = Pipeline('english')
# Output:
# Downloading: 100%|██| 5.07M/5.07M [00:00<00:00, 9.28MB/s]
# http://nlp.uoregon.edu/download/trankit/english.zip
# Downloading: 100%|█| 47.9M/47.9M [00:00<00:00, 89.2MiB/s]
# Loading pretrained XLM-Roberta, this may take a while...
# Downloading: 100%|███████| 512/512 [00:00<00:00, 330kB/s]
# Downloading: 100%|██| 1.12G/1.12G [00:14<00:00, 74.8MB/s]
# Loading tokenizer for english
# Loading tagger for english
# Loading lemmatizer for english
# Loading NER tagger for english
# ==================================================
# Active language: english
# ==================================================

p.add('chinese')
# http://nlp.uoregon.edu/download/trankit/chinese.zip
# Downloading: 100%|█| 40.4M/40.4M [00:00<00:00, 81.3MiB/s]
# Loading tokenizer for chinese
# Loading tagger for chinese
# Loading lemmatizer for chinese
# Loading NER tagger for chinese
# ==================================================
# Added languages: ['english', 'chinese']
# Active language: english
# ==================================================

p.add('arabic')
# http://nlp.uoregon.edu/download/trankit/arabic.zip
# Downloading: 100%|█| 38.6M/38.6M [00:00<00:00, 76.8MiB/s]
# Loading tokenizer for arabic
# Loading tagger for arabic
# Loading multi-word expander for arabic
# Loading lemmatizer for arabic
# Loading NER tagger for arabic
# ==================================================
# Added languages: ['english', 'chinese', 'arabic']
# Active language: english
# ==================================================
```
As we can see, each time a new language is added, the list of the added languages inreases. However, the active langage remains the same, i.e., `'english'`. This indicates that the pipeline can work with inputs of the 3 specified languages, however, it is assuming that the inputs that it will receive are in `'english'`. To change this assumption, we need to "tell" the pipeline that we're going to process inputs of a particular language, for example:
```python
p.set_active('chinese')
# ==================================================
# Active language: chinese
# ==================================================
```
From now, the pipeline is ready to process `'chinese'` inputs. To make sure that the language is activated successfully, we can access the attribute `active_lang` of the pipeline:
```python
print(p.active_lang)
# 'chinese'
```

## Document-level processing
The following lines of code show the basic use of *Trankit* with English inputs.
```python
from trankit import Pipeline
# initialize a pipeline for English
p = Pipeline('english')

# a non-empty string to process, which can be a document or a paragraph with multiple sentences
doc_text = '''Hello! This is Trankit.'''

# perform all tasks on the input
all = p(doc_text)
```
Here, `doc_text` is assumed to be a document. Then, the sentence segmentation and tokenization are then jointly done. For each sentence, *Trankit* performs part-of-speech tagging, morphological feature tagging, dependency parsing, and also named entity recognition (NER) if the pretrained NER model for that language is available. The result of the entire process is stored in the variable `all`, which is a hierarchical native Python dictionary that we can retrieve different types of information at both the document and sentence level. The output would look like this (we use [...] to improve the visualization):
```python
{
  'text': 'Hello! This is Trankit.',  # input string
  'sentences': [ # list of sentences
    {
      'id': 1, 'text': 'Hello!', 'dspan': (0, 6), 'tokens': [...]
    },
    {
      'id': 2,  # sentence index
      'text': 'This is Trankit.',  'dspan': (7, 23), # sentence span
      'tokens': [ # list of tokens
        {
          'id': 1, # token index
          'text': 'This', 'upos': 'PRON', 'xpos': 'DT',
          'feats': 'Number=Sing|PronType=Dem',
          'head': 3, 'deprel': 'nsubj', 'lemma': 'this', 'ner': 'O',
          'dspan': (7, 11), # document-level span of the token
          'span': (0, 4) # sentence-level span of the token
        },
        {'id': 2...},
        {'id': 3...},
        {'id': 4...}
      ]
    }
  ]
}
```
Below we show some examples for accessing different information of the output.

### Text
At document level, we have two fields to access, which are `'text'` storing the input string and `'sentences'` storing the tagged sentences of the input. Suppose that we want to get the text form of the first sentence, we would first access the field `'sentences'` of the output to get the list of sentences, use an index to locate the sentence in the list, then finally access the `'text'` field:
```python
sent_text = all['sentences'][1]['text'] 
print(sent_text)
# Output: This is Trankit.
```

### Span

we can also get the text form of the sentence manually with the `'dspan'` field that provides the span-based location of the sentence in the document:
```python
dspan = all['sentences'][1]['dspan']
print(dspan) 
# Output: (7, 23)
sent_text = doc_text[dspan[0]: dspan[1]]
print(sent_text) 
# Output: This is Trankit.
```
Note that, we use `'dspan'` with the prefix `d` to indicate that this information is at document level.

### Token list

Each sentence is associated with a list of tokens, which can be accessed via the `'tokens'` field. Each token is in turn a dictionary with different types of information.
For example, we can get the information of the first token of the second sentence as follows:
```python
token = all['sentences'][1]['tokens'][0]
print(token)
```
The information of the token is stored in a Python dictionary:
```python
{
  'id': 1,                   # token index
  'text': 'This',            # text form of the token
  'upos': 'PRON',            # UPOS tag of the token
  'xpos': 'DT',              # XPOS tag of the token
  'feats': 'Number=Sing|PronType=Dem',    # morphological feature of the token
  'head': 3,                 # index of the head token
  'deprel': 'nsubj',         # dependency relation between from the current token and the head token
  'dspan': (7, 11),          # document-level span of the token
  'span': (0, 4),            # sentence-level span of the token
  'lemma': 'this',           # lemma of the token
  'ner': 'O'             # Named Entity Recognitation (NER) tag of the token
}
```
Here, we provide two different types of span for each token: `'dspan'` and `'span'`. `'dspan'` is used for the global location of the token in the document while `'span'` provides the local location of the token in the sentence. We can use either one of these two fields to manually retrieve the text form of the token like this:
```python
# retrieve the text form via 'dspan'
dspan = token['dspan']
print(doc_text[dspan[0]: dspan[1]])
# Output: This

# retrieve the text form via 'span'
span = token['span']
print(sent_text[span[0]: span[1]])
# Output: This
```  

## Sentence-level processing
In many cases, we may want to use *Trankit* to process a sentence instead of a document. This can be achieved by setting the tag `is_sent=True`:
```python
sent_text = '''Hello! This is Trankit.'''
tokens = p(sent_text, is_sent=True)
```
The output is now a dictionary with a list of all tokens, instead of a list of sentences as before.

```python
{
  'text': 'Hello! This is Trankit.',
  'tokens': [
    {
      'id': 1,
      'text': 'Hello',
      'upos': 'INTJ',
      'xpos': 'UH',
      'head': 5,
      'deprel': 'discourse',
      'lemma': 'hello',
      'ner': 'O',
      'span': (0, 5)
    },
    {'id': 2...},
    {'id': 3...},
    {'id': 4...},
    {'id': 5...},
    {'id': 6...},
  ]
}
```

For more examples on other functions, please refer to the following sections: [Sentence Segmentation](ssplit.md), [Tokenization](tokenize.md), [Part-of-speech, Morphological tagging and Dependency parsing](posdep.md), [Lemmatization](lemmatize.md), [Named entity recognition](ner.md), and [Building a customized pipeline](training.md).
