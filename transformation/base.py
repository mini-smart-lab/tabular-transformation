import abc

class Transformation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, data):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass