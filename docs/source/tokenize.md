# Tokenization

*NOTE*: [Quick examples](overview.md) might be helpful for using this function.

Tokenization module of *Trankit* can work with both sentence-level and document-level inputs. 

## Document-level tokenization
For document inputs, *trankit* first performs tokenization and sentence segmentation jointly to obtain a list of tokenized sentences. Below is how we can use this function:
```python
from trankit import Pipeline
# initialize a pipeline for English
p = Pipeline('english')

# a non-empty string to process, which can be a document or a paragraph with multiple sentences
doc_text = '''Hello! This is Trankit.'''

tokenized_doc = p.tokenize(doc_text)

print(tokenized_doc)
```
The returned `tokenized_doc` should look like this:
```python
{
  'text': 'Hello! This is Trankit.',
  'sentences': [
    {
      'id': 1,
      'text': 'Hello!', 'dspan': (0, 6),
      'tokens': [
        {
          'id': 1,
          'text': 'Hello',
          'dspan': (0, 5),
          'span': (0, 5)
        },
        {
          'id': 2,
          'text': '!',
          'dspan': (5, 6),
          'span': (5, 6)
        }
      ]
    },
    {
      'id': 2,
      'text': 'This is Trankit.', 'dspan': (7, 23)
      'tokens': [
        {
          'id': 1,
          'text': 'This',
          'dspan': (7, 11),
          'span': (0, 4)
        },
        {
          'id': 2,
          'text': 'is',
          'dspan': (12, 14),
          'span': (5, 7)
        },
        {
          'id': 3,
          'text': 'Trankit',
          'dspan': (15, 22),
          'span': (8, 15)
        },
        {
          'id': 4,
          'text': '.',
          'dspan': (22, 23),
          'span': (15, 16)
        }
      ]
    }
  ]
}
```
For each sentence, *trankit* provides the information of its location in the document via `'dspan'` field. For each token, there are two types of span that we can access: (i) Document-level span (via `'dspan'`) and (ii) Sentence-level span (via `'span'`). Check [this](overview.md) to know how these fields work.

## Sentence-level tokenization

In some cases, we might already have the sentences and only want to do tokenization for each sentence. This can be achieved by setting the tag `is_sent=True` when we call the function `.tokenize()`:
```python
from trankit import Pipeline
# initialize a pipeline for English
p = Pipeline('english')

# a non-empty string to process, which can be a document or a paragraph with multiple sentences
sent_text = '''This is Trankit.'''

tokens = p.tokenize(sent_text, is_sent=True)

print(tokens)
```
This will return a list of tokens. The output will look like this:
```python
{
  'text': 'This is Trankit.',
  'tokens': [
    {
      'id': 1,
      'text': 'This',
      'span': (0, 4)
    },
    {
      'id': 2,
      'text': 'is',
      'span': (5, 7)
    },
    {
      'id': 3,
      'text': 'Trankit',
      'span': (8, 15)
    },
    {
      'id': 4,
      'text': '.',
      'span': (15, 16)
    }
  ]
}
```
As the input is assumed to be a sentence, we only have the sentence-level span for each token.

## Multi-word token expansion
In addition to tokenization, some languages also require *Multi-word token expansion*. That means, each token can be expanded into multiple syntactic words. This process is helpful for these languages when performing later tasks such as part-of-speech, morphological tagging, dependency parsing, and lemmatization. Below is an example for such case in French:
```python 
from trankit import Pipeline

p = Pipeline('french')

doc_text = '''Je sens qu'entre ça et les films de médecins et scientifiques fous que nous
avons déjà vus, nous pourrions emprunter un autre chemin pour l'origine. On
pourra toujours parler à propos d'Averroès de "décentrement du Sujet".'''

out_doc = p.tokenize(doc_text)
print(out_doc['sentences'][1])
```
For illustration purpose, we only show part of the second sentence:
```python
{
    'text': 'Je sens qu\'entre ça et les films de médecins et scientifiques fous que nous\navons déjà vus, nous pourrions emprunter un autre chemin pour l\'origine. On\npourra toujours parler à propos d\'Averroès de "décentrement du Sujet".',
    'sentences': [
      ...
      ,
      {
        'id': 2,
        'text': 'On\npourra toujours parler à propos d\'Averroès de "décentrement du Sujet".',
        'dspan': (149, 222),
        'tokens': [
          ... 
          ,
          {
            'id': 11, 
            'text': 'décentrement',
            'dspan': (199, 211),
            'span': (50, 62)
          },
          {
            'id': (12, 13), # token index
            'text': 'du', # text form
            'expanded': [ # list of syntactic words
              {
                'id': 12, # token index
                'text': 'de' # text form
              },
              {
                'id': 13, # token index 
                'text': 'le' # text form
              }
            ],
            'span': (63, 65),
            'dspan': (212, 214)
          },
          {
            'id': 14,
            'text': 'Sujet',
            'dspan': (215, 220),
            'span': (66, 71)
          },
          {
            'id': 15,
            'text': '"',
            'dspan': (220, 221),
            'span': (71, 72)
          },
          {
            'id': 16,
            'text': '.',
            'dspan': (221, 222),
            'span': (72, 73)
          }
        ]
      }
    ]
```
The expanded tokens always have the indexes that are tuple objects, instead of integers as usual. In this example, the expanded token is the token with the index `(12, 13)`. The tuple indicates that this token is expanded into the syntactic words with the indexes ranging from `12` to `13`. The syntactic words are organized into a list stored in the field `'expanded'` of the original token. *Note that, part-of-speech, morphological tagging, dependency parsing, and lemmatization always work with the syntactic words instead of the original token in such case*. That's why we will only see additional features added to the syntactic words while the original token remains unchanged with only the information of its text form and spans. *As a last note, Named Entity Recognition (NER) module by contrast only works with the original tokens instead of the syntactic words, so we will not see the NER tags for the syntactic words*.