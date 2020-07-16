from typing import List


class MongoQueryParser(object):
    OPERATORS_MAPPING = {
        '>': 'gt',
        '>=': 'gte',
        '=': '',
        '<': 'lt',
        '<=': 'lte',
        '!=': 'ne',
        'not_in': 'nin',
    }

    @classmethod
    def parse_single_order(cls, order_string: str) -> str:
        return order_string.replace('.', '__')

    @classmethod
    def parse_orders_list(cls, orders_strings: List[str]) -> List[str]:
        return [cls.parse_single_order(x) for x in orders_strings]

    @classmethod
    def parse_single_query(cls, query_string: str) -> str:
        chunks = query_string.split(' ')
        assert len(chunks) == 3, f'Invalid query string: [{query_string}]'

        label: str = chunks[0]
        label = label.replace('.', '__')

        operator: str = chunks[1]
        if operator in cls.OPERATORS_MAPPING:
            operator = cls.OPERATORS_MAPPING[operator]

        value = eval(chunks[2])

        query_label = f'{label}__{operator}' if len(operator) > 0 else f'{label}'
        return query_label, value

    @classmethod
    def parse_queries_list(cls, queries_list: List[str]) -> List[str]:
        query_dict = {}
        for query_string in queries_list:
            label, value = cls.parse_single_query(query_string)
            query_dict[label] = value
        return query_dict
