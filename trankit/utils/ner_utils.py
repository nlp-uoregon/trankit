import json


def convert_to_bio2(ori_tags):
    bio2_tags = []
    for i, tag in enumerate(ori_tags):
        if tag == 'O':
            bio2_tags.append(tag)
        elif tag[0] == 'I':
            if i == 0 or ori_tags[i - 1] == 'O' or ori_tags[i - 1][1:] != tag[1:]:
                bio2_tags.append('B' + tag[1:])
            else:
                bio2_tags.append(tag)
        else:
            bio2_tags.append(tag)
    return bio2_tags


def convert_to_bioes(tags):
    bioes_tags = []
    for i, tag in enumerate(tags):
        tag = tag.replace('B--', 'B-').replace('I--', 'I-')
        if tag == 'O':
            bioes_tags.append(tag)
        else:
            assert len(tag) >= 2
            if tag[:2] == 'I-':
                if i + 1 < len(tags) and tags[i + 1][:2] == 'I-':
                    bioes_tags.append(tag)
                else:
                    bioes_tags.append('E-' + tag[2:])
            elif tag[:2] == 'B-':
                if i + 1 < len(tags) and tags[i + 1][:2] == 'I-':
                    bioes_tags.append(tag)
                else:
                    bioes_tags.append('S-' + tag[2:])
            else:
                bioes_tags.append('O')
    return bioes_tags


def get_example_from_lines(sent_lines):
    tokens = []
    ner_tags = []
    for line in sent_lines:
        array = line.split()
        assert len(array) >= 2
        tokens.append(array[0])
        ner_tags.append(array[-1])
    ner_tags = convert_to_bioes(convert_to_bio2(ner_tags))
    return {'words': tokens, 'entity-labels': ner_tags}


def get_examples_from_bio_fpath(config, bio_fpath, evaluate):
    sent_lines = []
    bioes_examples = []
    with open(bio_fpath) as infile:
        for line in infile:
            line = line.strip()
            if '-DOCSTART-' in line:
                continue
            if len(line) > 0:
                array = line.split()
                if len(array) < 2:
                    continue
                else:
                    sent_lines.append(line)
            elif len(sent_lines) > 0:
                example = get_example_from_lines(sent_lines)
                bioes_examples.append(example)
                sent_lines = []
        if len(sent_lines) > 0:
            bioes_examples.append(get_example_from_lines(sent_lines))

    if not evaluate:
        tagset = set()
        for ex in bioes_examples:
            tagset.update(set(ex['entity-labels']))
        taglist = list(tagset)
        vocab = {'O': 0}
        taglist = [t for t in taglist if t != 'O']
        taglist.sort()
        for t in taglist:
            vocab[t] = vocab.get(t, len(vocab))

        with open(config.vocab_fpath,
                  'w') as f:
            json.dump(vocab, f)

    return bioes_examples


def read_bio_format(bio_fpath):
    sent_lines = []
    bio_examples = []
    with open(bio_fpath) as infile:
        for line in infile:
            line = line.strip()
            if '-DOCSTART-' in line:
                continue
            if len(line) > 0:
                array = line.split()
                if len(array) < 2:
                    continue
                else:
                    sent_lines.append(line)
            elif len(sent_lines) > 0:
                example = get_example_from_lines(sent_lines)
                bio_examples.append(example)
                sent_lines = []
        if len(sent_lines) > 0:
            bio_examples.append(get_example_from_lines(sent_lines))
    return bio_examples
