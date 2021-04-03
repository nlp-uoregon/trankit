from .pipeline import Pipeline
from .tpipeline import TPipeline
from .pipeline import supported_langs, langwithner, remove_with_path

__version__ = "1.0.1"


def verify_customized_pipeline(category, save_dir):
    import os
    assert category in {'customized', 'customized-ner', 'customized-mwt',
                    'customized-mwt-ner'}, "Pipeline category must be one of the following: 'customized', 'customized-ner', 'customized-mwt', 'customized-mwt-ner'"
    if category == 'customized':
        file_list = [
            os.path.join(save_dir, category, '{}.tokenizer.mdl'.format(category)),
            os.path.join(save_dir, category, '{}.tagger.mdl'.format(category)),
            os.path.join(save_dir, category, '{}.vocabs.json'.format(category)),
            os.path.join(save_dir, category, '{}_lemmatizer.pt'.format(category))
        ]
    elif category == 'customized-ner':
        file_list = [
            os.path.join(save_dir, category, '{}.tokenizer.mdl'.format(category)),
            os.path.join(save_dir, category, '{}.tagger.mdl'.format(category)),
            os.path.join(save_dir, category, '{}.vocabs.json'.format(category)),
            os.path.join(save_dir, category, '{}_lemmatizer.pt'.format(category)),
            os.path.join(save_dir, category, '{}.ner.mdl'.format(category)),
            os.path.join(save_dir, category, '{}.ner-vocab.json'.format(category))
        ]
    elif category == 'customized-mwt':
        file_list = [
            os.path.join(save_dir, category, '{}.tokenizer.mdl'.format(category)),
            os.path.join(save_dir, category, '{}_mwt_expander.pt'.format(category)),
            os.path.join(save_dir, category, '{}.tagger.mdl'.format(category)),
            os.path.join(save_dir, category, '{}.vocabs.json'.format(category)),
            os.path.join(save_dir, category, '{}_lemmatizer.pt'.format(category))
        ]
    elif category == 'customized-mwt-ner':
        file_list = [
            os.path.join(save_dir, category, '{}.tokenizer.mdl'.format(category)),
            os.path.join(save_dir, category, '{}_mwt_expander.pt'.format(category)),
            os.path.join(save_dir, category, '{}.tagger.mdl'.format(category)),
            os.path.join(save_dir, category, '{}.vocabs.json'.format(category)),
            os.path.join(save_dir, category, '{}_lemmatizer.pt'.format(category)),
            os.path.join(save_dir, category, '{}.ner.mdl'.format(category)),
            os.path.join(save_dir, category, '{}.ner-vocab.json'.format(category))
        ]
    else:
        assert 'Unknown customized lang!'

    verified = True
    for filepath in file_list:
        if not os.path.exists(filepath):
            verified = False
            print('Missing {}'.format(filepath))
    if verified:
        with open(os.path.join(save_dir, category, '{}.downloaded'.format(category)), 'w') as f:
            f.write('')
        remove_with_path(os.path.join(save_dir, category, 'train.txt.character'))
        remove_with_path(os.path.join(save_dir, category, 'logs'))
        remove_with_path(os.path.join(save_dir, category, 'preds'))
        remove_with_path(os.path.join(save_dir, category, 'xlm-roberta-large'))
        remove_with_path(os.path.join(save_dir, category, 'xlm-roberta-base'))
        print(
            "Customized pipeline is ready to use!\nIt can be initialized as follows:\n-----------------------------------\nfrom trankit import Pipeline\np = Pipeline(lang='{}', cache_dir='{}')".format(
                category, save_dir))
    else:
        print('Customized pipeline is not ready to use!\nPlease consider the missing files above.')
