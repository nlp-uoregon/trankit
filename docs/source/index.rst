.. trankit documentation master file, created by
   sphinx-quickstart on March 31 10:21:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Trankit's Documentation
================================================

**News**: Trankit v1.0.0 is out. The new version has pretrained pipelines using XLM-Roberta Large which is much better than our previous version with XLM-Roberta Base. Checkout the performance comparison `here <https://trankit.readthedocs.io/en/latest/performance.html>`_. Trankit v1.0.0 also provides a brand new `command-line interface <https://trankit.readthedocs.io/en/latest/commandline.html>`_ that helps users who are not familiar with Python can use Trankit more easily. Finally, the new version has a brand new `Auto Mode <https://trankit.readthedocs.io/en/latest/news.html#auto-mode-for-multilingual-pipelines>`_ in which a language detector is used to activate the language-specific models for processing the input, avoiding switching back and forth between languages in a multilingual pipeline.

Trankit is a *light-weight Transformer-based Python* Toolkit for multilingual Natural Language Processing (NLP). It provides a trainable pipeline for fundamental NLP tasks over 100 languages, and 90 pretrained pipelines for 56 languages. Built on a state-of-the-art pretrained language model, Trankit significantly outperforms prior multilingual NLP pipelines over sentence segmentation, part-of-speech tagging, morphological feature tagging, and dependency parsing while maintaining competitive performance for tokenization, multi-word token expansion, and lemmatization over 90 Universal Dependencies v2.5 treebanks. Our pipeline also obtains competitive or better named entity recognition (NER) performance compared to existing popular toolkits on 11 public NER datasets over 8 languages.

Trankit's Github Repo is available at: https://github.com/nlp-uoregon/trankit

Trankit's Demo Website is hosted at: http://nlp.uoregon.edu/trankit

Citation
========
If you use Trankit in your research or software. Please cite `our following paper <https://arxiv.org/pdf/2101.03289.pdf>`_:

.. code-block:: bibtex

   @inproceedings{nguyen2021trankit,
      title={Trankit: A Light-Weight Transformer-based Toolkit for Multilingual Natural Language Processing}, 
      author={Minh Van Nguyen, Viet Lai, Amir Pouran Ben Veyseh and Thien Huu Nguyen},
      booktitle="Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
      year={2021}
   }


.. toctree::
   :maxdepth: 2
   :caption: Introduction

   news
   installation
   overview
   howitworks
   performance

.. toctree::
   :maxdepth: 1
   :caption: Usage

   pkgnames
   ssplit
   tokenize
   posdep
   lemmatize
   ner
   training
   commandline
