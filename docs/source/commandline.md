# Command-line interface

Starting from version v1.0.0, Trankit supports processing text via command-line interface. This helps users who are not familiar with Python programming language can use Trankit more easily.

## Requirements
Users need to install Trankit via one of the following methods:

Pip:
```
pip install trankit==1.0.0
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
    
    Language(s) of the pipeline to be initialized. Check out this [page]() to see the available language names.
    
    Example use:
    
    -Monolingual case: `python -m trankit [other options] --lang english`
    
    -Multilingual case with 3 languages: `python -m trankit [other options] --lang english,chinese,arabic`
    
    -Multilingual case with all supported languages: `python -m trankit [other options] --lang auto`
    
    In multilingual mode, trankit will automatically detect the language of the input file(s) to use corresponding models.
    
    Note that, language detection is done at file level.
 
* `--cpu`
    
    Forcing trankit to run on CPU. Default: False.Example use:
    
    `python -m trankit [other options] --cpu`

* `--embedding`
    
    Multilingual embedding for trankit. Default: xlm-roberta-base.
    
    Example use:
    
    -XLM-Roberta base: `python -m trankit [other options] --embedding xlm-roberta-base`
    
    -XLM-Roberta large: `python -m trankit [other options] --embedding xlm-roberta-large`
    
* `--cache_dir`
    
    Location to store downloaded model files. Default: cache/trankit.
    
    Example use:
    
    `python -m trankit [other options] --cache_dir your/cache/dir`

* `--input`
    
    Location of the input.
    
    If it is a directory, trankit will process each file in the input directory at a time.
    
    If it is a file, trankit will process the file only.
    
    The input file(s) can be raw or pretokenized text. Sample input can be found here:
    
    [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs)
    
    Example use:
    
    -Input is a directory: `python -m trankit [other options] --input some_dir_path`
    
    -Input is a file: `python -m trankit [other options] --input some_file_path`
    
* `--input_format`
    
    Indicating that the input format. Sample input can be found here:
    
    [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs)

* `--output_dir`
    
    Location of the output directory to store the processed files. Processed files will be in json format, with the naming convention as follows:
    
        `processed_file_name = input_file_name + .processed.json`
    
    Example use:
    
    python -m trankit [other options] --output_dir some_dir_path

* `--task`
    
    Task to be performed for the provided input.
    
    Use cases:
    
    -Sentence segmentation, assuming input is a single DOCUMENT string.
    
       `python -m trankit [other options] --task ssplit`
    
     Sample input for ssplit: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt)
    
    -Sentence segmentation + Tokenization, assuming input is a single DOCUMENT string.
    
       `python -m trankit [other options] --task dtokenize`
    
     Sample input for dtokenize: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt)
    
    -Tokenization only, assuming input contains multiple raw SENTENCE strings in each line.
    
       python -m trankit [other options] --task stokenize
    
     Sample input for stokenize: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt)
    
    -Sentence segmentation, Tokenization, Part-of-speech tagging, Morphological tagging, Dependency parsing.
     
     Assuming input is a single DOCUMENT string.
      
       python -m trankit [other options] --task dposdep
     
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt)
    
    -Tokenization only, Part-of-speech tagging, Morphological tagging, Dependency parsing.
     
     Assuming input contains multiple raw SENTENCE strings in each line.
     
       python -m trankit [other options] --task sposdep
     
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt)
    
    -Part-of-speech tagging, Morphological tagging, Dependency parsing.
     
     Assuming input contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word.
     
       `python -m trankit [other options] --task pposdep`
     
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt)
    
    -Sentence segmentation, Tokenization, Lemmatization
     
     Assuming input is a single DOCUMENT string.
       `python -m trankit [other options] --task dlemmatize`
       
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt)
    
    -Tokenization only, Lemmatization
     
     Assuming input contains multiple raw SENTENCE strings in each line.
       
       `python -m trankit [other options] --task slemmatize`
     
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt)
    
    -Lemmatization
     
     Assuming input contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word.
       
       `python -m trankit [other options] --task plemmatize`
     
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt)
    
    -Sentence segmentation, Tokenization, Named Entity Recognition.
     
     Assuming input is a single DOCUMENT string.
      
       `python -m trankit [other options] --task dner`
     
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt)
    
    -Tokenization only, Named Entity Recognition.
     
     Assuming input contains multiple raw SENTENCE strings in each line.
       
       `python -m trankit [other options] --task sner`
     
     Sample input for dposdep: [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt)
    
    -Named Entity Recognition.
     
     Assuming input contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word.
       
       `python -m trankit [other options] --task pner`
     
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt)
    
    -Sentence segmentation, Tokenization, Part-of-speech tagging, Morphological tagging, Dependency parsing, Named Entity Recognition.
    
    Assuming input is a single DOCUMENT string.
       
       `python -m trankit [other options] --task dall`
     
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt)
    
    -Tokenization only, Part-of-speech tagging, Morphological tagging, Dependency parsing, Named Entity Recognition.
     
     Assuming input contains multiple raw SENTENCE strings in each line.
       
       `python -m trankit [other options] --task sall`
     
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt)
    
    -Part-of-speech tagging, Morphological tagging, Dependency parsing, Named Entity Recognition.
       
       `python -m trankit [other options] --task pall`
     
     Sample input for dposdep: 
     
     [https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt](https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt)