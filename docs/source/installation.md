# Installation

Installing *Trankit* is easily done via one of the following methods:

## Using pip

```
pip install trankit
```
The command would install *Trankit* and all dependent packages automatically.

## From source
```
git clone https://github.com/nlp-uoregon/trankit
cd trankit
pip install -e .
```
This would first clone our github repo and install Trankit.

## Fixing the compatibility issue of Trankit with Transformers
Previous versions of Trankit have encountered the [compatibility issue](https://github.com/nlp-uoregon/trankit/issues/5) when using recent versions of [transformers](https://github.com/huggingface/transformers). To fix this issue, please install the new version of Trankit as follows:
```
pip install trankit==1.0.1
```
If you encounter any other problem with the installation, please raise an issue [here](https://github.com/nlp-uoregon/trankit/issues/new) to let us know. Thanks.
