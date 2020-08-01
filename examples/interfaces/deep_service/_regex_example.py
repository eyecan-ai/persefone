from schema import Schema, Regex, Or


d = {
    '_endpoint': '/dataset/name',
    '_method': 'POST',
    '_query': {
        'data': bytes(),
        'data_encoding': 'jpg'
    }
}

schemas = {
    'get_dataset': Schema({
        '_endpoint': Or(
            Regex('\\/dataset?\\/[A-Za-z0-9]+$'),
            Regex('\\/dataset?\\/[A-Za-z0-9]+\\/$')
        ),
        '_method': 'GET',
        '_query': dict
    }),
    'new_dataset': Schema({
        '_endpoint': Or(
            Regex('\\/dataset?\\/[A-Za-z0-9]+$'),
            Regex('\\/dataset?\\/[A-Za-z0-9]+\\/$')
        ),
        '_method': 'POST',
        '_query': dict
    }),
    'get_sample': Schema({
        '_endpoint': Or(Regex('\\/dataset?\\/[A-Za-z0-9]+\\/[A-Za-z0-9]+$')),
        '_method': 'GET',
        '_query': dict
    }),
}


for k, v in schemas.items():
    print(k, v.is_valid(d))
