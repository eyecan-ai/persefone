"""Main module."""
import torch


class FooClass(object):

    def __init__(self, nothing):
        self.__nothing = nothing

    def do_nothing(self, x):
        """ summary

        :param x: input parameter
        :type x: int
        :return: return new FooClass object
        :rtype: torch.Tensor
        """

        print("Do nothing with:", x)
        return torch.Tensor()
