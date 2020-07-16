from typing import List, Tuple, Any


class MongoQueryParser(object):
    OPERATORS_MAPPING = {
        '>': 'gt',
        '>=': 'gte',
        '=': '',
        '==': '',
        '<': 'lt',
        '<=': 'lte',
        '!=': 'ne',
        'not_in': 'nin',
    }

    @classmethod
    def parse_single_order(cls, order_string: str) -> str:
        """ Parses a single orderby string like "+field.subfield" in order to convert
        it in a PyMongo/Mongoengine format lik "+field__subfield"

        :param order_string: input orderby string
        :type order_string: str
        :return: converted orderby string
        :rtype: str
        """

        return order_string.replace('.', '__')

    @classmethod
    def parse_orders_list(cls, orders_strings: List[str]) -> List[str]:
        """ Parses a list of orderby strings

        :param orders_strings: list of orderby strings
        :type orders_strings: List[str]
        :return: converted list of orderby strings
        :rtype: List[str]
        """

        return [cls.parse_single_order(x) for x in orders_strings]

    @classmethod
    def parse_single_query(cls, query_string: str) -> Tuple[str, Any]:
        """ Parses a single query string like "field.subfield >= VALUE" in order to convert
        it in a tuple as PyMongo/Mongoengine format ('field__subfield_gtw', VALUE)

        :param query_string: input query string
        :type query_string: str
        :return: parsed query string
        :rtype: Tuple[str, Any]
        """

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
    def parse_queries_list(cls, queries_list: List[str]) -> dict:
        """ Parses a list of query strings

        :param queries_list: list of query strings
        :type queries_list: List[str]
        :return: dict representing kwargs used in Mongoengine queries
        :rtype: dict
        """

        query_dict = {}
        for query_string in queries_list:
            label, value = cls.parse_single_query(query_string)
            query_dict[label] = value
        return query_dict
