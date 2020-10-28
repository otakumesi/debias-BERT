def extract_kv_by_prefix(target_dict, prefix):
    return {k.lstrip(prefix): v for k, v in target_dict if k.startswith(prefix)}


def extract_spans_from_offset_maps(offset_maps, start, end):
    offsets = [
        i
        for i, pair in enumerate(offset_maps)
        if pair[0] == start or pair[1] == end
    ]

    span_start = offsets[0]
    span_end = offsets[-1]

    if span_start == 0:
        span_start += 1

    return (span_start, span_end)


def prepare_gap(datasets, tokenizer, max_token_len=500):
    def make_label(example):
        if example["A-coref"] == 1:
            return {"labels": 1}
        if example["B-coref"] == 1:
            return {"labels": 2}
        return {"labels": 0}

    def make_spans(example):
        p_start = example["Pronoun-offset"]
        a_start = example["A-offset"]
        b_start = example["B-offset"]

        p_end = p_start + len(example["Pronoun"])
        a_end = a_start + len(example["A"])
        b_end = b_start + len(example["B"])

        offset_maps = example["offset_mapping"]

        p_indeces = extract_spans_from_offset_maps(offset_maps, p_start, p_end)
        a_indeces = extract_spans_from_offset_maps(offset_maps, a_start, a_end)
        b_indeces = extract_spans_from_offset_maps(offset_maps, b_start, b_end)

        return {"a_span_indeces": [a_indeces, p_indeces],
                "b_span_indeces": [b_indeces, p_indeces]}

    datasets = datasets.map(make_label, remove_columns=['A-coref', 'B-coref'])
    datasets = datasets.map(lambda example: tokenizer([t.lower() for t in example["Text"]],
                                                      return_offsets_mapping=True,
                                                      max_length=max_token_len,
                                                      padding='max_length'),
                            batched=True,
                            remove_columns=['Text'])
    datasets = datasets.map(make_spans, remove_columns=[
                            'Pronoun-offset', 'A-offset', 'B-offset', 'Pronoun', 'A', 'B'])

    return datasets
