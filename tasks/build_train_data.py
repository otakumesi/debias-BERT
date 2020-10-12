import json
import inflect

p = inflect.engine()
ATTR_NOUN_FILE = 'data/occupations.txt'

SUBJECTS = [('he', 'she')]

TEMPLATE_SENTENCES = [
    '{target} is {occupation}.',
    '{target} works as {occupation}.',
    '{target} is interested in {occupation}.'
]

def add_article(noun):
    if noun in ['he', 'she', 'they']:
        return noun

    if not p.singular_noun(noun):
        return f'the {noun}'
    else:
        return f'the {p.singular_noun(noun)}'

def should_add_article(noun):
    if noun in ['he', 'she', 'they']:
        return False
    return True


def main():
    with open(ATTR_NOUN_FILE) as f:
        attrs = f.readlines()
    subjects = SUBJECTS

    attrs = [add_article(attr.rstrip('\n').lower()) for attr in attrs]

    results = []

    for sbjs in subjects:
        for attr in attrs:
            for sent in TEMPLATE_SENTENCES:
                if should_add_article(sbjs[0]):
                    mask = 'the [MASK]'
                else:
                    mask = '[MASK]'
                biased_sent = sent.format(target = mask, occupation = attr)
                standard_sent = sent.format(target = mask, occupation = 'the [MASK]')
                results.append({'biased_sentence': biased_sent,
                                'base_sentence': standard_sent,
                                'targets': sbjs})

    print(json.dumps({'data': results}))

if __name__ == '__main__':
    main()
