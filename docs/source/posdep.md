# Part-of-speech, Morphological tagging and Dependency parsing

*NOTE*: [Quick examples](overview.md) might be helpful for using this function.

In *trankit*, part-of-speech, morphological tagging, and dependency parsing are jointly performed. The module can work with either untokenized or pretokenized inputs, at both sentence and document level.

## Document-level processing

### Untokenized input
The sample code for this module is:
```python
from trankit import Pipeline
# initialize a pipeline for English
p = Pipeline('english')

# a non-empty string to process, which can be a document or a paragraph with multiple sentences
doc_text = '''Hello! This is Trankit.'''

all = p.posdep(doc_text)
```
*Trankit* first performs tokenization and sentence segmentation for the input document, then performs the part-of-speech, morphological tagging, and dependency parsing for the tokenized document. The output of the whole process is a native Python dictionary with list of sentences, each sentence contains a list of tokens with the predicted part-of-speech, the morphological feature, the index of the head token, and the corresponding dependency relation for each token. The output would look like this:
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
          'text': 'This',  # text form of the token
          'upos': 'PRON',  # UPOS tag of the token
          'xpos': 'DT',    # XPOS tag of the token
          'feats': 'Number=Sing|PronType=Dem', # morphological feature of the token
          'head': 3,  # index of the head token
          'deprel': 'nsubj', # dependency relation for the token
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

### Pretokenized input

In some cases, we might already have a tokenized document and want to use this module. Here is how we can do it:
```python
pretokenized_doc = [
  ['Hello', '!'],
  ['This', 'is', 'Trankit', '.']
]

tagged_doc = p.posdep(pretokenized_doc)
```
Pretokenized inputs are automatically recognized by *Trankit*. That's why we don't have to specify any additional tag when calling the function `.posdep()`. The output in this case will be the same as in the previous case except that now we don't have any span information.

## Sentence-level processing
Sometimes we want to use this module for sentence inputs. To achieve that, we can simply set `is_sent=True` when we call the function `.posdep()`:

### Untokenized input
```python
sent_text = '''This is Trankit.'''

tagged_sent = p.posdep(sent_text, is_sent=True)
```

### Pretokenized input
```python
pretokenized_sent = ['This', 'is', 'Trankit', '.']

tagged_sent = p.posdep(pretokenized_sent, is_sent=True)
```