# Lemmatization

*NOTE*: [Quick examples](overview.md) might be helpful for using this function.

*Trankit* supports lemmatization for both untokenized and pretokenized inputs, at both sentence and document level. Here are some examples:

## Document-level lemmatization
In this case, the input is assumed to be a document.
### Untokenized input
```python
from trankit import Pipeline

p = Pipeline('english')

doc_text = '''Hello! This is Trankit.'''

lemmatized_doc = p.lemmatize(doc_text)
```
*Trankit* would first perform tokenization and sentence segmentation for the input document. Next, it assigns a lemma to each token in the sentences. The output would look like this:
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
          'lemma': 'this', # lemma of the token
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
For illustration purpose, we only show the first sentence.

### Pretokenized input
Pretokenized inputs are automatically recognized by *Trankit*. The following snippet performs lemmatization on a pretokenized document, which is a list of lists of strings:
```python
from trankit import Pipeline

p = Pipeline('english')

pretokenized_doc = [
  ['Hello', '!'],
  ['This', 'is', 'Trankit', '.']
]

lemmatized_doc = p.lemmatize(pretokenized_doc)
```
The output will look slightly different without the spans of the sentences and the tokens:
```python
{
  'sentences': [
    {
      'id': 1,
      'tokens': [
        {
          'id': 1,
          'text': 'Hello',
          'lemma': 'hello'
        },
        {
          'id': 2,
          'text': '!',
          'lemma': '!'
        }
      ]
    },
    {
      'id': 2,
      'tokens': [
        {
          'id': 1,
          'text': 'This',
          'lemma': 'this'
        },
        {
          'id': 2,
          'text': 'is',
          'lemma': 'be'
        },
        {
          'id': 3,
          'text': 'Trankit',
          'lemma': 'trankit'
        },
        {
          'id': 4,
          'text': '.',
          'lemma': '.'
        }
      ]
    }
  ]
}
```
## Sentence-level lemmatization
Lemmatization module also accepts inputs as sentences. This can be done by setting the tag `is_sent=True`. The output would be a dictionary with a list of lemmatized tokens.

### Untokenized input

```python
from trankit import Pipeline

p = Pipeline('english')

sent_text = '''This is Trankit.'''

lemmatized_sent = p.lemmatize(sent_text, is_sent=True)
```

### Pretokenized input
```python
from trankit import Pipeline

p = Pipeline('english')

sent_text = '''This is Trankit.'''

pretokenized_sent = ['This', 'is', 'Trankit', '.']

lemmatized_sent = p.lemmatize(pretokenized_sent, is_sent=True)
```