import trankit
from time import sleep

p = trankit.Pipeline('english')

num_passed = 0

for lid, lang in enumerate(trankit.supported_langs):
    p.add(lang)
    p.set_active(lang)
    with open('trankit/tests/sample_inputs/{}.txt'.format(lang)) as f:
        text = f.read()

    all_doc = p(text)
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
    print('*' * 30 + ' {}:{}: PASSED '.format(lid, lang) + '*' * 30)
    num_passed += 1
    sleep(1)

print('=' * 20 + ' SUMMARY ' + '=' * 20)
print('Total passed: {}'.format(num_passed))
