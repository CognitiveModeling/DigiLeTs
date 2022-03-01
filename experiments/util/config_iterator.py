from abc import ABC, abstractmethod


class ConfigIterator(ABC):
    """
    Class to describe configs that can be iterated on programatically,
    e.g. to test the effects of different batch sizes. Recommend to use yield-syntax.
    """
    @abstractmethod
    def __iter__(self):
        pass
