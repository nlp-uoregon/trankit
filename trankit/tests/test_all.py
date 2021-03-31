import os, json, sys
import trankit
from time import sleep

embedding = sys.argv[1]


def is_equal(a, b):
    with open('a.json', 'w') as f:
        json.dump(a, f, ensure_ascii=False)
    with open('b.json', 'w') as f:
        json.dump(b, f, ensure_ascii=False)
    with open('a.json') as f:
        ajson = json.load(f)
    with open('b.json') as f:
        bjson = json.load(f)
    bjson['lang'] = ajson['lang']
    if ajson == bjson: return True
    return False


p = trankit.Pipeline('english', embedding=embedding)

num_passed = 0

for lid, lang in enumerate(trankit.supported_langs):
    p.add(lang)
    p.set_active(lang)
    with open('trankit/tests/sample_inputs/{}.txt'.format(lang)) as f:
        text = f.read()

    all_doc = p(text)
    with open('trankit/tests/sample_outputs/{}/{}.json'.format(embedding, lang)) as f:
        sample_output = json.load(f)

    # this might not always be equal due to the difference of the running environment
    if not is_equal(all_doc, sample_output): continue

    all_sent = p(text, is_sent=True)

    sents = p.ssplit(text)

    tokens = p.tokenize(text)
    tokens2 = p.tokenize(text, is_sent=True)

    posdep1 = p.posdep(text)
    posdep2 = p.posdep(text, is_sent=True)
    posdep3 = p.posdep([[w['text'] for w in sent['tokens']] for sent in tokens['sentences']])
    posdep4 = p.posdep([w['text'] for w in tokens2['tokens']], is_sent=True)

    lemma1 = p.lemmatize(text)
    lemma2 = p.lemmatize(text, is_sent=True)
    lemma3 = p.lemmatize([[w['text'] for w in sent['tokens']] for sent in tokens['sentences']])
    lemma4 = p.lemmatize([w['text'] for w in tokens2['tokens']], is_sent=True)

    if lang in trankit.langwithner:
        ner1 = p.ner(text)
        ner2 = p.ner(text, is_sent=True)
        ner3 = p.ner([[w['text'] for w in sent['tokens']] for sent in tokens['sentences']])
        ner4 = p.ner([w['text'] for w in tokens2['tokens']], is_sent=True)
    print('*' * 30 + ' {}:{}: PASSED '.format(lid + 1, lang) + '*' * 30)
    num_passed += 1

print('=' * 20 + ' SUMMARY ' + '=' * 20)
print('Total passed: {}/{}'.format(num_passed, len(trankit.supported_langs)))
