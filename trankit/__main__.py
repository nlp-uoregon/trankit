import sys, os, json
from time import time
import glob
from argparse import ArgumentParser
from tqdm import tqdm


def check_valid_format(input_format, task):
    if input_format == 'plaindoc':
        if task in ['ssplit', 'dtokenize', 'dposdep', 'dlemmatize', 'dner', 'dall']:
            return True
        else:
            return False
    elif input_format == 'plainsen':
        if task in ['stokenize', 'sposdep', 'slemmatize', 'sner', 'sall']:
            return True
        else:
            return False
    else:
        assert input_format == 'pretok'
        if task in ['pposdep', 'plemmatize', 'pner', 'pall']:
            return True
        else:
            return False


def process_a_file(input_fpath, input_format, output_dir, pipeline, task):
    if input_format == 'plaindoc':
        with open(input_fpath) as f:
            task_input = f.read()
    elif input_format == 'plainsen':
        with open(input_fpath) as f:
            task_input = [sent.strip() for sent in f.readlines() if sent.strip()]
    else:
        assert input_format == 'pretok'
        with open(input_fpath) as f:
            task_input = [[w.strip() for w in sent.strip().splitlines() if w.strip()] for sent in f.readlines() if
                          sent.strip()]

    if task == 'ssplit':
        output = pipeline.ssplit(task_input)
    elif task == 'stokenize':
        output = {'sentences': [pipeline.tokenize(sent, is_sent=True) for sent in task_input]}
    elif task == 'dtokenize':
        output = pipeline.tokenize(task_input)
    elif task == 'sposdep':
        output = {'sentences': [pipeline.posdep(sent, is_sent=True) for sent in task_input]}
    elif task == 'dposdep':
        output = pipeline.posdep(task_input)
    elif task == 'pposdep':
        output = pipeline.posdep(task_input)
    elif task == 'sner':
        output = {'sentences': [pipeline.ner(sent, is_sent=True) for sent in task_input]}
    elif task == 'dner':
        output = pipeline.ner(task_input)
    elif task == 'pner':
        output = pipeline.ner(task_input)
    elif task == 'slemmatize':
        output = {'sentences': [pipeline.lemmatize(sent, is_sent=True) for sent in task_input]}
    elif task == 'dlemmatize':
        output = pipeline.lemmatize(task_input)
    elif task == 'plemmatize':
        output = pipeline.lemmatize(task_input)
    elif task == 'dall':
        output = pipeline(task_input)
    elif task == 'sall':
        output = {'sentences': [pipeline(sent, is_sent=True) for sent in task_input]}
    else:
        assert task == 'pall'
        output = pipeline(task_input)

    with open(os.path.join(output_dir, os.path.basename(input_fpath) + '.{}.json'.format(task)), 'w') as f:
        json.dump(output, f, ensure_ascii=False)


# configuration
parser = ArgumentParser()
# model hyper-parameters
parser.add_argument('--lang', type=str, required=True,
                    help='Language(s) of the pipeline to be initialized.\n' + \
                         'Example use:\n' + \
                         '-Monolingual case: python -m trankit [other options] --lang english\n' + \
                         '-Multilingual case with 3 languages: python -m trankit [other options] --lang english,chinese,arabic\n' + \
                         '-Multilingual case with all supported languages: python -m trankit [other options] --lang auto\n' + \
                         'In multilingual mode, trankit will automatically detect the language of the input file(s) to use corresponding models.\n' + \
                         'Note that, language detection is done at file level.\n'
                    )
parser.add_argument('--cpu', action='store_true',
                    help='Forcing trankit to run on CPU. Default: False.\n' + \
                         'Example use:\n' + \
                         'python -m trankit [other options] --cpu\n'
                    )
parser.add_argument('--embedding', default='xlm-roberta-base', type=str,
                    choices=['xlm-roberta-base', 'xlm-roberta-large'],
                    help='Multilingual embedding for trankit. Default: xlm-roberta-base.\n' + \
                         'Example use:\n' + \
                         '-XLM-Roberta base: python -m trankit [other options] --embedding xlm-roberta-base\n' + \
                         '-XLM-Roberta large: python -m trankit [other options] --embedding xlm-roberta-large\n'
                    )
parser.add_argument('--cache_dir', default='cache/trankit', type=str,
                    help='Location to store downloaded model files. Default: cache/trankit.\n' + \
                         'Example use:\n' + \
                         'python -m trankit [other options] --cache_dir your/cache/dir\n'
                    )
parser.add_argument('--input', type=str, required=True,
                    help='Location of the input.\n' + \
                         'If it is a directory, trankit will process each file in the input directory at a time.\n' + \
                         'If it is a file, trankit will process the file only.\n' + \
                         'The input file(s) can be raw or pretokenized text. Sample input can be found here:\n' + \
                         'https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs\n' + \
                         'Example use:\n' + \
                         '-Input is a directory: python -m trankit [other options] --input some_dir_path\n' + \
                         '-Input is a file: python -m trankit [other options] --input some_file_path\n'
                    )
parser.add_argument('--input_format', type=str, default='plaindoc',
                    choices=['plaindoc', 'plainsen', 'pretok'],
                    help='Indicating that the input format. Sample input can be found here:\n' + \
                         'https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs\n'
                    )
parser.add_argument('--output_dir', default='trankit_output', type=str,
                    help='Location of the output directory to store the processed files.' +
                         'Processed files will be in json format, which is described in our documentation page: https://trankit.readthedocs.io/en/latest/commandline.html\n' + \
                         'Example use:\n' + \
                         'python -m trankit [other options] --output_dir some_dir_path\n'
                    )
parser.add_argument('--task', type=str, default='dall',
                    choices=['ssplit', 'stokenize', 'dtokenize', 'sposdep', 'dposdep', 'pposdep', 'sner', 'dner',
                             'pner', 'slemmatize', 'dlemmatize', 'plemmatize', 'dall', 'sall', 'pall'],
                    help='Task to be performed for the provided input.\n' + \
                         'Use cases:\n' + \
                         '-Sentence segmentation, assuming input is a single DOCUMENT string.\n' + \
                         '   python -m trankit [other options] --task ssplit\n' + \
                         ' Sample input for ssplit: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt\n'+\
                         '\n' + \
                         '-Sentence segmentation + Tokenization, assuming input is a single DOCUMENT string.\n' + \
                         '   python -m trankit [other options] --task dtokenize\n' + \
                         ' Sample input for dtokenize: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt\n' + \
                         '-Tokenization only, assuming input contains multiple raw SENTENCE strings in each line.\n' + \
                         '   python -m trankit [other options] --task stokenize\n' + \
                         ' Sample input for stokenize: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt\n' + \
                         '\n' + \
                         '-Sentence segmentation, Tokenization, Part-of-speech tagging, Morphological tagging, Dependency parsing.\n' + \
                         ' Assuming input is a single DOCUMENT string.\n' + \
                         '   python -m trankit [other options] --task dposdep\n' + \
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt\n'+\
                         '-Tokenization only, Part-of-speech tagging, Morphological tagging, Dependency parsing.\n' + \
                         ' Assuming input contains multiple raw SENTENCE strings in each line.\n' + \
                         '   python -m trankit [other options] --task sposdep\n' + \
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt\n'+\
                         '-Part-of-speech tagging, Morphological tagging, Dependency parsing.\n' + \
                         ' Assuming input contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word.\n' + \
                         '   python -m trankit [other options] --task pposdep\n' + \
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt\n' + \
                         '\n' + \
                         '-Sentence segmentation, Tokenization, Lemmatization\n' + \
                         ' Assuming input is a single DOCUMENT string.\n' + \
                         '   python -m trankit [other options] --task dlemmatize'+\
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt\n' + \
                         '-Tokenization only, Lemmatization\n' + \
                         ' Assuming input contains multiple raw SENTENCE strings in each line.\n' + \
                         '   python -m trankit [other options] --task slemmatize\n' + \
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt\n'+\
                         '-Lemmatization\n' + \
                         ' Assuming input contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word.\n' + \
                         '   python -m trankit [other options] --task plemmatize\n' + \
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt\n' + \
                         '\n' + \
                         '-Sentence segmentation, Tokenization, Named Entity Recognition.\n' + \
                         ' Assuming input is a single DOCUMENT string.\n' + \
                         '   python -m trankit [other options] --task dner\n' + \
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt\n'+\
                         '-Tokenization only, Named Entity Recognition.\n' + \
                         ' Assuming input contains multiple raw SENTENCE strings in each line.\n' + \
                         '   python -m trankit [other options] --task sner\n' + \
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt\n'
                         '-Named Entity Recognition.\n' + \
                         ' Assuming input contains pretokenized SENTENCES separated by "\n\n", each sentence is organized into multiple lines, each line contains only a single word.\n' + \
                         '   python -m trankit [other options] --task pner\n' + \
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt\n' + \
                         '\n' + \
                         '-Sentence segmentation, Tokenization, Part-of-speech tagging, Morphological tagging, Dependency parsing, Named Entity Recognition.\n' + \
                         'Assuming input is a single DOCUMENT string.\n' + \
                         '   python -m trankit [other options] --task dall\n' + \
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plaindoc.txt\n' + \
                         '-Tokenization only, Part-of-speech tagging, Morphological tagging, Dependency parsing, Named Entity Recognition.\n'+\
                         ' Assuming input contains multiple raw SENTENCE strings in each line.\n'+\
                         '   python -m trankit [other options] --task sall\n'+\
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/plainsen.txt\n'+\
                         '-Part-of-speech tagging, Morphological tagging, Dependency parsing, Named Entity Recognition.\n'+\
                         '   python -m trankit [other options] --task pall\n'+\
                         ' Sample input for dposdep: https://github.com/nlp-uoregon/trankit/tree/master/examples/commandline/sample_inputs/pretok.txt\n'
                    )
config = parser.parse_args()

from .pipeline import Pipeline
from .pipeline import treebank2lang, get_ud_score, get_ud_performance_table, ensure_dir

assert os.path.exists(config.input), "{} doesn't exist.".format(config.input)
langs = config.lang.split(',')
assert len(langs) > 0

assert check_valid_format(config.input_format, config.task), 'Unexpected input format for {}.'.format(config.task)
ensure_dir(config.output_dir)
####################### Initialization #########################
if len(langs) == 1:  # Monolingual case or auto
    p = Pipeline(langs[0], gpu=False if config.cpu else True, cache_dir=config.cache_dir, embedding=config.embedding)
else:  # Multilingual case with specified languages
    p = Pipeline(langs[0], gpu=False if config.cpu else True, cache_dir=config.cache_dir, embedding=config.embedding)
    for l in langs[1:]:
        p.add(l)
    p.set_auto(True)
####################### Input reading ##########################
if os.path.isdir(config.input):
    fpaths = [os.path.join(config.input, fname) for fname in os.listdir(config.input) if
              os.path.isfile(os.path.join(config.input, fname))]
else:  # input is a file
    fpaths = [config.input]
####################### Processing #############################
progress = tqdm(total=len(fpaths), ncols=75,
                desc='Processing')
for fpath in fpaths:
    progress.update(1)
    process_a_file(
        input_fpath=fpath,
        input_format=config.input_format,
        output_dir=config.output_dir,
        pipeline=p,
        task=config.task
    )
progress.close()
print('Processing is done.\nOutput files are put at: {}'.format(config.output_dir))
