def extract_kv_by_prefix(target_dict, prefix):
    return {k.lstrip(prefix): v for k, v in target_dict if k.startswith(prefix)}


def prepare_gap(dataset, tokenizer):
    def make_label(example):
        if example["A-coref"] == 1:
            return {"labels": 1}
        if example["B-coref"] == 1:
            return {"labels": 2}
        return {"labels": 0}

    def make_spans(example):
        a_start = example["A-offset"]
        b_start = example["B-offset"]

        a_end = a_start + len(example["A"])
        b_end = b_start + len(example["B"])

        example["offset_mapping"]
        import ipdb

        ipdb.set_trace()

        return {"a_span_indeces": [], "b_span_indeces": []}

    dataset = dataset.map(make_label)
    dataset = dataset.map(
        lambda example: tokenizer(
            example["Text"].lower(), return_tensors="pt", return_offsets_mapping=True
        )
    )
    dataset = dataset.map(make_spans)

    return dataset
