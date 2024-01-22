def get_configuration(name):
    if "local" in name:
        method = "local"
    elif "pairwise" in name:
        method = "pairwise"

    if "mean" in name:
        reduction = "mean"
    elif "min" in name:
        reduction = "min"

    return method, reduction