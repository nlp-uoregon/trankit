# pip install trankit
# https://trankit.readthedocs.io/en/stable/training.html

#an example using trankit to train custom model with pre tokenized conllu file including multiword token

import trankit,re, os, sys, json
from trankit.iterators.tagger_iterators import TaggerDataset

#load pipe
from trankit import Pipeline
from trankit.utils import CoNLL
from trankit.utils.base_utils import get_ud_score, get_ud_performance_table


# res_folder = 'trankit_res'
# train_path = 'trankit/conllus/train_parser_en_gum.conllu'
# dev_path = 'trankit/conllus/test_frontend.conllu'
# dev_raw = 'trankit/conllus/dev_raw.txt'
# train_raw = 'trankit/conllus/train_raw.txt'
# pred_fpath'trankit/pred_test_lem.conllu'

# epoch = 100
# epoch_tok = 40

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

def train_deprel_feat(res_folder, epoch):
    # initialize a trainer for the task
    trainer_dep = trankit.TPipeline(
        training_config={
        'max_epoch': epoch,
        'category': 'customized', # pipeline category
        'task': 'posdep', # task name
        'save_dir': res_folder, # directory for saving trained model
        'train_conllu_fpath': train_path, # annotations file in CONLLU format  for training
        'dev_conllu_fpath': dev_path, # annotations file in CONLLU format for development
        'embedding': 'xlm-roberta-large'

        }
    )

    # start training
    trainer_dep.train()
    return trainer_dep



def train_lemma(res_folder, epoch):
    # initialize a trainer for the task
    trainer= trankit.TPipeline(
        training_config={
            'max_epoch': epoch,
            'category': 'customized',  # pipeline category
            'task': 'lemmatize', # task name
            'save_dir': res_folder, # directory for saving trained model
            'train_conllu_fpath': train_path, # annotations file in CONLLU format  for training
            'dev_conllu_fpath': dev_path, # annotations file in CONLLU format for development
            'embedding': 'xlm-roberta-large'
        }
    )
    # start training
    trainer.train()


def get_raw_file(conllu_path, raw_path):
    txt = open(conllu_path).read()
    txt_pattern = re.compile(r"# text =.+")
    res = '\n'.join([l[9:] for l in re.findall(txt_pattern, txt)])
    if raw_path:
        with open(raw_path, 'w') as f:
            f.write(res)


def train_tok(res_folder, epoch_tok):
    """tokenizer required to build a pipeline for parsing"""
    get_raw_file(train_path, train_raw)
    get_raw_file(dev_path, dev_raw)
    
    # initialize a trainer for the task
    trainer_tok = trankit.TPipeline(
        training_config={
            'max_epoch': epoch_tok,
            'category': 'customized', # pipeline category
            'task': 'tokenize', # task name
            'save_dir': res_folder, # directory for saving trained model
            'train_txt_fpath': train_raw, # raw text file
            'train_conllu_fpath': train_path, # annotations file in CONLLU format for training
            'dev_txt_fpath': dev_raw, # raw text file
            'dev_conllu_fpath': dev_path, # annotations file in CONLLU format for development
            'embedding': 'xlm-roberta-large'
        }
    )
    # start training
    trainer_tok.train()

def test_deprel(trainer, test_path, name = 'test_dep'):
    #test trainer 
    #from trankit.iterators.tagger_iterators import TaggerDataset
    #trainer should be TPipeline instance for posdep, not for lemma 

    test_set = TaggerDataset(
        config=trainer._config,
        input_conllu = test_path,
        gold_conllu= test_path,
        evaluate= False
    )

    test_set.numberize()
    test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
    result = trainer._eval_posdep(data_set=test_set, batch_num=test_batch_num,
                            name=name, epoch= -1)
    print(trankit.utils.base_utils.get_ud_performance_table(result[0]))
    return result[0]


def check_pipe(model_dir):
    #check pipe
    trankit.verify_customized_pipeline(
        category= 'customized', # pipeline category
        save_dir= model_dir, # directory used for saving models in previous steps
        embedding_name='xlm-roberta-large' # embedding version that we use for training our customized pipeline, by default, it is `xlm-roberta-base`
    )


#load pipe
# from trankit import Pipeline
# from trankit.utils import CoNLL

def get_toklist(fpath):
    conllu_ls = CoNLL.load_conll(open(fpath), ignore_gapping = True)
    res = []
    for sent in conllu_ls:
        res.append([l[1] for l in sent])
    return res



def pred_trankit( pipe , to_parse_path, parsed_path, task = 'posdep'):
    # p = Pipeline(lang='customized', cache_dir= model_dir )
    #get token list
    conll_list = CoNLL.load_conll(open(to_parse_path), ignore_gapping = True)

    tok_ls = []
    expand_end = -1 
    for sent in conll_list:
        sent_info = []
        ldict = {}
        for l in sent:
            if '-' in l[0]:
                ldict['id'] = ( int(l[0].split('-')[0]), int(l[0].split('-')[1]) )
                ldict['text'] = l[1]
                expand_end = ldict['id'][1]
                ldict['expanded'] = []
            elif expand_end > 0 and int(l[0]) <= expand_end:
                ldict['expanded'].append( { 'id': int(l[0]), 'text' : l[1] } )

                if int(l[0]) == expand_end:
                    #reset
                    expand_end = -1
                    sent_info.append(ldict)
                    ldict = {}
            else:
                sent_info.append( { 'id': int(l[0]), 'text' : l[1] }) 
        tok_ls.append(sent_info)

    #lemmiatize + pos headid tag feat
    if task == 'posdep':
        res_dict = pipe.posdep_withID(tok_ls )

        doc_conll = CoNLL.convert_dict([ s['tokens'] for s in res_dict['sentences'] ], use_expand = True)
        conll_string = CoNLL.conll_as_string(doc_conll)
        with open(parsed_path, 'w') as outfile:
            outfile.write(conll_string)

    if task == 'lemmatize':
        res_dict = pipe.lemmatize_withID(tok_ls)

        doc_conll = CoNLL.convert_dict([ s['tokens'] for s in res_dict['sentences'] ], use_expand = True)
        conll_string = CoNLL.conll_as_string(doc_conll)
        with open(parsed_path, 'w') as outfile:
            outfile.write(conll_string)



def train_trankit(res_folder, epoch, epoch_tok):
    #train_tok(res_folder, epoch_tok)
    train_deprel_feat(res_folder, epoch)
    train_lemma(res_folder, epoch)
    train_tok(res_folder, epoch_tok)
    check_pipe(res_folder)

def eval_parsed(parsed_path, gold_path):
    score = get_ud_score(parsed_path, gold_path)
    print(get_ud_performance_table(score))
    return score

def save_score(score,score_dir,  res_folder, cv_idx, name = 'test', newfile = True):
    metric_ls = [ "Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS",
                   "CLAS", "MLAS", "BLEX"]
    mode = 'w' if newfile else 'a'
    #f1 score
    with open(os.path.join(score_dir, name+'_trankit_f1score.tsv'), mode) as f:
        if mode == 'w':
            f.write('\t'.join( ["Metrics"] +  metric_ls) + '\n')
        
        f.write('\t'.join( [f"cv{cv_idx}"] + ["{}".format(score[metric].f1) for metric in metric_ls] ) + "\n" )

    #more score
    res = {}
    for metric in metric_ls:
        res[metric] = {
            'precision': score[metric].precision,
            'recall': score[metric].recall,
            'f1': score[metric].f1}
        if score[metric].aligned_accuracy is not None:
            res['aligned_accuracy'] = score[metric].aligned_accuracy

    with open(os.path.join(res_folder, name + '_score.json'), 'w') as f1:
        json.dump(res, f1, indent = 4)


def posdep(res_folder, epoch, test_path, name = 'test_dep'):
    print(name)
    trainer = train_deprel_feat(res_folder, epoch)
    score = test_deprel(trainer, test_path, name = name)
    return score


def copy_lemma_file(lemma_path, posdep_path):
    # TODO
    #better to combine in backend when we copy the upos???
    print('copy lemma:', lemma_path)
    lemma_txt  = open(lemma_path).read().strip()
    begin, tmp = lemma_txt.split("sent_id ", 1)
    lemmas= [t.split('\n') for t in ("# sent_id "+tmp).split('\n\n') if t]

    lemma_dict = {}
    for conllu in lemmas:
        # every sent begin with #sent_id 
        # TODO replace this by keyword sent_id instead of index 
        key = conllu[0].split('=')[1].strip()
        lemma_dict[key] = [line for line in conllu[1:] if line[0] != '#']

    posdep_txt = open(posdep_path).read().strip()
    begin, tmp = parsed_txt.split("sent_id ", 1)
    deprel = [t.split('\n') for t in ("# sent_id "+tmp).split('\n\n') if t]

    posdep_dict = {}
    for conllu in deprel:
        key = conllu[0].split('=')[1].strip()
        posdep_dict[key] = conllu[1:]

    for key, conll in posdep_dict.items():
        begin = 0
        for l, line in enumerate(conll):
            if(line[0]!='#'):
                info = line.split('\t')
                info_tag = lemma_dict[key][l - begin].split('\t')
                #print(info)
                info[3] = info_tag[LEMMA]
                posdep_dict[key][l] = '\t'.join(info)
            else:
                begin += 1 
        posdep_dict[key] = '\n'.join(posdep_dict[key])

    to_write = begin[:-2] + '\n\n'.join([f'# sent_id = {k}\n' + val for k, val in posdep_dict.items()]) + '\n\n'
    with open(os.path.join(os.path.dirname(posdep_path), 'combined_parsed.conllu'), 'w' ) as f:
        f.write(to_write)




if __name__ == '__main__':
    # pred_fpath = 'trankit/pred_test_lem.conllu'
    if len(sys.argv) < 8:
        print(len(sys.argv))
        print("Usage: train_pred_trankit.py project_folder data_folder score_dir to_parse_path epoch epoch_tok cv_idx", file=sys.stderr)
        sys.exit(-1)

    #python3 train_pred_trankit.py 'test_trankit' 30 10 

    #set param
    res_folder = sys.argv[1]  #os.path.join(sys.argv[1], 'trankit_res')

    train_path = os.path.join(sys.argv[2], 'train.conllu') 
    dev_path = os.path.join(sys.argv[2], 'dev.conllu') 
    dev_raw = os.path.join(sys.argv[2], 'dev_raw.txt') 
    train_raw = os.path.join(sys.argv[2], 'train_raw.txt') 

    score_dir = sys.argv[3]

    to_parse_path = sys.argv[4]  #os.path.join(sys.argv[3], 'test1000.conllu')
    epoch = int(sys.argv[5])
    epoch_tok = int(sys.argv[6])
    cv_idx =  int(sys.argv[7])


    #train & pred
    lemma = True
    if lemma:
        train_trankit(res_folder, epoch, epoch_tok)
        p = Pipeline(lang='customized', cache_dir= res_folder, embedding = 'xlm-roberta-large' )

        for task in ['posdep','lemmatize']:
            print('==== pred for task ', task)
            parsed_path = os.path.join(res_folder, f'parsed_{task}_test1000.conllu')
            print(parsed_path)
            print(to_parse_path)
            pred_trankit(p, to_parse_path, parsed_path, task = task)

            score = eval_parsed(parsed_path, to_parse_path)

            new_fscore = True if cv_idx == 0 else False
            save_score(score, score_dir, res_folder, cv_idx, name = f'test_{task}', newfile = new_fscore)

        # if False:
        #     copy_lemma_file(
        #         os.path.join(res_folder, 'parsed_lemmatize_test1000.conllu'),
        #         os.path.join(res_folder, 'parsed_posdep_test1000.conllu')
        #     )
    else:
        posdep(res_folder, epoch, to_parse_path, name = 'test') 



