# News: Trankit v1.0.0 is out

What's new in Trankit v1.0.0? Let's check out below.

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