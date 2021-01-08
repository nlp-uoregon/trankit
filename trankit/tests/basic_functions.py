import trankit

p = trankit.Pipeline('english')

p.add('arabic')
p.add('chinese')
p.add('vietnamese')

text = '''I doubt the very few who actually read my blog have not come across this yet,
but I figured I would put it out there anyways. John Donovan from Argghhh! has
put out a excellent slide show on what was actually found and fought for in
Fallujah.'''

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

ner1 = p.ner(text)
ner2 = p.ner(text, is_sent=True)
ner3 = p.ner([[w['text'] for w in sent['tokens']] for sent in tokens['sentences']])
ner4 = p.ner([w['text'] for w in tokens2['tokens']], is_sent=True)