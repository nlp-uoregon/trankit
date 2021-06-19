from .pipeline import Pipeline
from .tpipeline import TPipeline
from .pipeline import supported_langs, langwithner, remove_with_path
from .utils.base_utils import download, trankit2conllu
from .utils.tbinfo import supported_embeddings, supported_langs, saved_model_version
import os
from shutil import copyfile

__version__ = "1.1.0"


def download_missing_files(category, save_dir, embedding_name, language):
    assert language in supported_langs, '{} is not a pretrained language. Current pretrained languages: {}'.format(language, supported_langs)
    assert embedding_name in supported_embeddings, '{} has not been supported. Current supported embeddings: {}'.format(embedding_name, supported_embeddings)

    import os
    assert category in {'customized', 'customized-ner', 'customized-mwt',
                        'customized-mwt-ner'}, "Pipeline category must be one of the following: 'customized', 'customized-ner', 'customized-mwt', 'customized-mwt-ner'"
    if category == 'customized':
        file_list = [
            ('{}.tokenizer.mdl', os.path.join(save_dir, embedding_name, category, '{}.tokenizer.mdl'.format(category))),
            ('{}.tagger.mdl', os.path.join(save_dir, embedding_name, category, '{}.tagger.mdl'.format(category))),
            ('{}.vocabs.json', os.path.join(save_dir, embedding_name, category, '{}.vocabs.json'.format(category))),
            ('{}_lemmatizer.pt', os.path.join(save_dir, embedding_name, category, '{}_lemmatizer.pt'.format(category)))
        ]
    elif category == 'customized-ner':
        file_list = [
            ('{}.tokenizer.mdl', os.path.join(save_dir, embedding_name, category, '{}.tokenizer.mdl'.format(category))),
            ('{}.tagger.mdl', os.path.join(save_dir, embedding_name, category, '{}.tagger.mdl'.format(category))),
            ('{}.vocabs.json', os.path.join(save_dir, embedding_name, category, '{}.vocabs.json'.format(category))),
            ('{}_lemmatizer.pt', os.path.join(save_dir, embedding_name, category, '{}_lemmatizer.pt'.format(category))),
            ('{}.ner.mdl', os.path.join(save_dir, embedding_name, category, '{}.ner.mdl'.format(category))),
            ('{}.ner-vocab.json', os.path.join(save_dir, embedding_name, category, '{}.ner-vocab.json'.format(category)))
        ]
    elif category == 'customized-mwt':
        file_list = [
            ('{}.tokenizer.mdl', os.path.join(save_dir, embedding_name, category, '{}.tokenizer.mdl'.format(category))),
            ('{}_mwt_expander.pt', os.path.join(save_dir, embedding_name, category, '{}_mwt_expander.pt'.format(category))),
            ('{}.tagger.mdl', os.path.join(save_dir, embedding_name, category, '{}.tagger.mdl'.format(category))),
            ('{}.vocabs.json', os.path.join(save_dir, embedding_name, category, '{}.vocabs.json'.format(category))),
            ('{}_lemmatizer.pt', os.path.join(save_dir, embedding_name, category, '{}_lemmatizer.pt'.format(category)))
        ]
    elif category == 'customized-mwt-ner':
        file_list = [
            ('{}.tokenizer.mdl', os.path.join(save_dir, embedding_name, category, '{}.tokenizer.mdl'.format(category))),
            ('{}_mwt_expander.pt', os.path.join(save_dir, embedding_name, category, '{}_mwt_expander.pt'.format(category))),
            ('{}.tagger.mdl', os.path.join(save_dir, embedding_name, category, '{}.tagger.mdl'.format(category))),
            ('{}.vocabs.json', os.path.join(save_dir, embedding_name, category, '{}.vocabs.json'.format(category))),
            ('{}_lemmatizer.pt', os.path.join(save_dir, embedding_name, category, '{}_lemmatizer.pt'.format(category))),
            ('{}.ner.mdl', os.path.join(save_dir, embedding_name, category, '{}.ner.mdl'.format(category))),
            ('{}.ner-vocab.json', os.path.join(save_dir, embedding_name, category, '{}.ner-vocab.json'.format(category)))
        ]
    else:
        assert 'Unknown customized lang!'
    missing_filenamess = []
    for filename, filepath in file_list:
        if not os.path.exists(filepath):
            print('Missing {}'.format(filepath))
            missing_filenamess.append(filename)

    download(
        cache_dir=save_dir,
        language=language,
        saved_model_version=saved_model_version,  # manually set this to avoid duplicated storage
        embedding_name=embedding_name
    )
    # borrow pretrained files
    src_dir = os.path.join(save_dir, embedding_name, language)
    tgt_dir = os.path.join(save_dir, embedding_name, category)
    for fname in missing_filenamess:
        copyfile(os.path.join(src_dir, fname.format(language)), os.path.join(tgt_dir, fname.format(category)))
        print('Copying {} to {}'.format(
            os.path.join(src_dir, fname.format(language)),
            os.path.join(tgt_dir, fname.format(category))
        ))
    remove_with_path(src_dir)


def verify_customized_pipeline(category, save_dir, embedding_name):
    assert embedding_name in supported_embeddings, '{} has not been supported. Current supported embeddings: {}'.format(
        embedding_name, supported_embeddings)
    assert category in {'customized', 'customized-ner', 'customized-mwt',
                        'customized-mwt-ner'}, "Pipeline category must be one of the following: 'customized', 'customized-ner', 'customized-mwt', 'customized-mwt-ner'"
    if category == 'customized':
        file_list = [
            os.path.join(save_dir, embedding_name, category, '{}.tokenizer.mdl'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.tagger.mdl'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.vocabs.json'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}_lemmatizer.pt'.format(category))
        ]
    elif category == 'customized-ner':
        file_list = [
            os.path.join(save_dir, embedding_name, category, '{}.tokenizer.mdl'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.tagger.mdl'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.vocabs.json'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}_lemmatizer.pt'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.ner.mdl'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.ner-vocab.json'.format(category))
        ]
    elif category == 'customized-mwt':
        file_list = [
            os.path.join(save_dir, embedding_name, category, '{}.tokenizer.mdl'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}_mwt_expander.pt'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.tagger.mdl'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.vocabs.json'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}_lemmatizer.pt'.format(category))
        ]
    elif category == 'customized-mwt-ner':
        file_list = [
            os.path.join(save_dir, embedding_name, category, '{}.tokenizer.mdl'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}_mwt_expander.pt'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.tagger.mdl'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.vocabs.json'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}_lemmatizer.pt'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.ner.mdl'.format(category)),
            os.path.join(save_dir, embedding_name, category, '{}.ner-vocab.json'.format(category))
        ]
    else:
        assert 'Unknown customized lang!'

    verified = True
    for filepath in file_list:
        if not os.path.exists(filepath):
            verified = False
            print('Missing {}'.format(filepath))
    if verified:
        with open(os.path.join(save_dir, embedding_name, category, '{}.downloaded'.format(category)), 'w') as f:
            f.write('')
        remove_with_path(os.path.join(save_dir, embedding_name, category, 'train.txt.character'))
        remove_with_path(os.path.join(save_dir, embedding_name, category, 'logs'))
        remove_with_path(os.path.join(save_dir, embedding_name, category, 'preds'))
        print(
            "Customized pipeline is ready to use!\nIt can be initialized as follows:\n-----------------------------------\nfrom trankit import Pipeline\np = Pipeline(lang='{}', cache_dir='{}')".format(
                category, save_dir))
    else:
        print('Customized pipeline is not ready to use!\nPlease consider the missing files above.')
