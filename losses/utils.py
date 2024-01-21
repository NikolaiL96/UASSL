

def get_configuration(name):
    if "simple" in name:
        method = "simple"
    elif "pairwise" in name:
        method = "pairwise"

    if "mean" in name:
        reduction = "mean"
    elif "min" in name:
        reduction = "min"

    return method, reduction