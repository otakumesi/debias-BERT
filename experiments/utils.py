def extract_kv_by_prefix(target_dict, prefix):
    return {k.lstrip(prefix):v for k, v in target_dict if k.startswith(prefix)}
