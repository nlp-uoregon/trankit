# Sentence segmentation

*NOTE*: [Quick examples](overview.md) might be helpful for using this function.

The sample code for performing sentence segmentation on a raw text is:
```python
from trankit import Pipeline
# initialize a pipeline for English
p = Pipeline('english')

# a non-empty string to process, which can be a document or a paragraph with multiple sentences
doc_text = '''Hello! This is Trankit.'''

sentences = p.ssplit(doc_text)

print(sentences)
```
The output of the sentence segmentation module is a native Python dictionary with a list of the split sentences. For each sentence, we can access its span which is handy for retrieving the sentence's location in the original document. The output would look like this:
```python
{
  'text': 'Hello! This is Trankit.',
  'sentences': [
    {
      'id': 1,
      'text': 'Hello!',
      'dspan': (0, 6)
    },
    {
      'id': 2,
      'text': 'This is Trankit.',
      'dspan': (7, 23)
    }
  ]
}
```
