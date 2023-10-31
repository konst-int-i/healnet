"""
Base classes to load different data modalities
"""
from abc import abstractmethod


class Dataset(object):

    def __init__(self, name: str):
        self.name=name

    @abstractmethod
    def load_tabular(self):
        pass

    @abstractmethod
    def load_image(self):
        pass

    @abstractmethod
    def load_text(self):
        pass

