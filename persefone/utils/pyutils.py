def get_arg(kwargs, name, default=None):
    if name in kwargs:
        return kwargs[name]
    return default
