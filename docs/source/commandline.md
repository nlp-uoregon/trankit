# Command-line interface

Starting from version v1.0.0, Trankit supports processing text via command-line interface. This helps users who are not familiar with Python programming language can use Trankit more easily.

## Requirements
Users need to install Trankit via one of the following methods:

Pip:
```
pip install trankit==1.1.0
```

From source:
```
git clone https://github.com/nlp-uoregon/trankit
cd trankit
pip install -e .
```

## Syntax
```
python -m trankit [OPTIONS] --embedding xlm-roberta-base --cpu --lang english --input path/to/your/input --output_dir path/to/your/output_dir
```
What this command does are:
- Forcing Trankit to run on CPU (`--cpu`). Without this `--cpu`, Trankit will run on GPU if a GPU device is available.
- Initializing an English pipeline with XLM-Roberta base as the multilingual embedding (`--embedding xlm-roberta-base`).
- Performing all tasks on the input stored at `path/to/your/input` which can be a single input file or a folder storing multiple input files (`--input path/to/your/input`).
- Writing the output to `path/to/your/output_dir` which stores the output files, each is a json file with the prefix is the file name of the processed input file (`--output_dir path/to/your/output_dir`).

In this command, we can put more processing options at `[OPTIONS]`. Detailed description of the options that can be used:

* `--lang`
    
    Language(s) of the pipeline to be initialized. Check out this [page](https://trankit.readthedocs.io/en/latest/pkgnames.html#pretrained-languages-their-code-names) to see the available language names.
    
    Example use:
    
    -Monolingual case:
    
        python -m trankit [other options] --lang english
    
    -Multilingual case with 3 languages:
    
        python -m trankit [other options] --lang english,chinese,arabic
    
    -Multilingual case with all supported languages:
    
        python -m trankit [other options] --lang auto
    
    In multilingual mode, trankit will automatically detect the language of the input file(s) to use corresponding models.
    
    Note that, language detection is done at file level.
 
* `--cpu`
    
    Forcing trankit to run on CPU. Default: False.Example use:
    
        python -m trankit [other options] --cpu

* `--embedding`
    
    Multilingual embedding for trankit. Default: xlm-roberta-base.
    
    Example use:
    
    -XLM-Roberta base:
        
        python -m trankit [other options] --embedding xlm-roberta-base
    
    -XLM-Roberta large:
        
        python -m trankit [other options] --embedding xlm-roberta-large
    
* `--cache_dir`
    
    Location to store downloaded model files. Default: "cache/trankit".
    
    Example use:
    
        python -m trankit [other options] --cache_dir your/cache/dir

* `--input`
    
    Location of the input.
    
    If it is a directory, trankit will process each file in the input directory at a time.
    
    If it is a file, trankit will process the file only.
    
    Example use:
    
    -Input is a directory:
    
        python -m trankit [other options] --input some_dir_path
    
    -Input is a file:
        
        python -m trankit [other options] --input some_file_path
    
* `--input_format`
    
    Indicating the input format.
    
    Case 1: Each input file is a single raw DOCUMENT string:
        
        python -m trankit [other options] --input_format plaindoc
    
    Case 2: Each input file contains multiple raw SENTENCE strings in each line:
    
        python -m trankit [other options] --input_format plainsen
        
    Case 3: Each input file contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word:
    
        python -m trankit [other options] --input_format pretok
    
    
    Sample inputs can be found here:
    
    [plaindoc](https://github.com/nlp-uoregon/trankit/blob/master/examples/commandline/plaindoc.txt)
    
    [plainsen](https://github.com/nlp-uoregon/trankit/blob/master/examples/commandline/plainsen.txt)
    
    [pretok](https://github.com/nlp-uoregon/trankit/blob/master/examples/commandline/pretok.txt)

* `--output_dir`
    
    Location of the output directory to store the processed files. Processed files will be in json format, with the naming convention as follows:
    
        processed_file_name = input_file_name + .processed.json
    
    Example use:
    
        python -m trankit [other options] --output_dir some_dir_path

* `--task`
    
    Task to be performed for the provided input.
    
    Use cases:
    
    -Sentence segmentation, assuming each input file is a single raw DOCUMENT string (`--input_format plaindoc`).
    
        python -m trankit [other options] --task ssplit
    
    -Sentence segmentation + Tokenization, assuming each input file is a single raw DOCUMENT string (`--input_format plaindoc`).
    
        python -m trankit [other options] --task dtokenize
    
    -Tokenization only, assuming each input file contains multiple raw SENTENCE strings in each line (`--input_format plainsen`).
    
        python -m trankit [other options] --task stokenize
    
    -Sentence segmentation, Tokenization, Part-of-speech tagging, Morphological tagging, Dependency parsing.
     
     Assuming each input file is a single raw DOCUMENT string (`--input_format plaindoc`).
      
       python -m trankit [other options] --task dposdep
    
    -Tokenization only, Part-of-speech tagging, Morphological tagging, Dependency parsing.
     
     Assuming each input file contains multiple raw SENTENCE strings in each line (`--input_format plainsen`).
     
       python -m trankit [other options] --task sposdep
    
    -Part-of-speech tagging, Morphological tagging, Dependency parsing.
     
     Assuming each input file contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word (`--input_format pretok`).
     
       python -m trankit [other options] --task pposdep
    
    -Sentence segmentation, Tokenization, Lemmatization
     
     Assuming each input file is a single raw DOCUMENT string (`--input_format plaindoc`).
     
       python -m trankit [other options] --task dlemmatize
    
    -Tokenization only, Lemmatization
     
     Assuming each input file contains multiple raw SENTENCE strings in each line (`--input_format plainsen`).
       
       python -m trankit [other options] --task slemmatize
    
    -Lemmatization
     
     Assuming each input file contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word (`--input_format pretok`).
       
       python -m trankit [other options] --task plemmatize
    
    -Sentence segmentation, Tokenization, Named Entity Recognition.
     
     Assuming each input file is a single raw DOCUMENT string (`--input_format plaindoc`).
      
       python -m trankit [other options] --task dner
    
    -Tokenization only, Named Entity Recognition.
     
     Assuming each input file contains multiple raw SENTENCE strings in each line (`--input_format plainsen`).
       
       python -m trankit [other options] --task sner
    
    -Named Entity Recognition.
     
     Assuming each input file contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word (`--input_format pretok`).
       
       python -m trankit [other options] --task pner
    
    -Sentence segmentation, Tokenization, Part-of-speech tagging, Morphological tagging, Dependency parsing, Named Entity Recognition.
    
    Assuming each input file is a single raw DOCUMENT string (`--input_format plaindoc`).
       
       python -m trankit [other options] --task dall
    
    -Tokenization only, Part-of-speech tagging, Morphological tagging, Dependency parsing, Named Entity Recognition.
     
     Assuming each input file contains multiple raw SENTENCE strings in each line (`--input_format plainsen`).
       
       python -m trankit [other options] --task sall
    
    -Part-of-speech tagging, Morphological tagging, Dependency parsing, Named Entity Recognition.

	 Assuming each input file contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word (`--input_format pretok`).
       
       python -m trankit [other options] --task pall
