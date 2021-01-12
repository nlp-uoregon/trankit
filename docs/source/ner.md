# Named entity recognition

*NOTE*: [Quick examples](overview.md) might be helpful for using this function.

Currently, *Trankit* provides the Named Entity Recognition (NER) module for 8 languages which are Arabic, Chinese, Dutch, English, French, German, Russian, and Spanish. The NER module accepts the inputs that can be untokenized or pretokenized, at both sentence and document level. Below are some examples:

## Document-level

### Untokenized input
```python
from trankit import Pipeline
# initialize a pipeline for English
p = Pipeline('english')

# a non-empty string to process, which can be a document or a paragraph with multiple sentences
doc_text = '''Hello! This is Trankit.'''

tagged_doc = p.ner(doc_text)
```
The output would look like this:
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
          'text': 'This',
          'ner': 'O', # ner tag of the token
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
```python
from trankit import Pipeline

p = Pipeline('english')

pretokenized_doc = [
  ['Hello', '!'],
  ['This', 'is', 'Trankit', '.']
]

tagged_doc = p.ner(pretokenized_doc)
```
The output will look the same as in the untokenized case, except that now we don't have the text form for the input document as well as the span information for the sentences and the tokens.

## Sentence-level
To enable the NER module to work with sentence-level instead of document-level inputs, we can set the tag `is_sent=True`:
### Untokenized input
```python
from trankit import Pipeline

p = Pipeline('english')

sent_text = 'This is Trankit.'

tagged_sent = p.ner(sent_text, is_sent=True)
```

### Pretokenized input
```python
from trankit import Pipeline

p = Pipeline('english')

pretokenized_sent = ['This', 'is', 'Trankit', '.']

tagged_sent = p.ner(pretokenized_sent, is_sent=True)
```