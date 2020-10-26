def extract_kv_by_prefix(target_dict, prefix):
    return {k.lstrip(prefix):v for k, v in target_dict if k.startswith(prefix)}

def prepare_gap(dataset, tokenizer):
    def make_label(example):
        if exmaple['A-coref'] == 1:
            return {'labels': 1}
        if example['B-coref'] == 1:
            return {'labels': 2}
        return {'labels': 0}

    dataset = dataset.map(make_label)
    dataset = dataset.map(lambda example: tokenizer(example['text'].lower(), return_tensors='pt'))
    dataset = dataset.map(lambda example: {'offsets': (example['A-offset'], example['B-offset'])})

    return dataset
