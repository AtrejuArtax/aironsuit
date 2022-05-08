def to_sum(var):
    if isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, dict):
        var = [loss_ for _, loss_ in var.items()]
    if isinstance(var, list):
        var = sum(var)
    return var
