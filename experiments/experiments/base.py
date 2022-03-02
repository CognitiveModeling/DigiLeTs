__author__ = "Julius Wührer"

from abc import ABC, abstractmethod

class BaseExperiment(ABC):

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def run(self):
        pass
