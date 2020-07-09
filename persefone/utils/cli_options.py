import click
from typing import Callable


def cli_base_options(function):
    """ Encapsulate following options: ['--debug']

    :param function: decorated function
    :type function: Callable
    :return: decorated function
    :rtype: Callable
    """

    function = click.option('--debug', default=False, help="Debug mode", is_flag=True)(function)
    return function


def cli_host_options(function: Callable) -> Callable:
    """ Encapuslate following options: ['--host', '--port']

    :param function: decorated function
    :type function: Callable
    :return: decorated function
    :rtype: Callable
    """

    function = click.option("--host", default='0.0.0.0', type=str, help="Sever host")(function)
    function = click.option("--port", default=50051, type=int, help="Sever port")(function)
    return function
