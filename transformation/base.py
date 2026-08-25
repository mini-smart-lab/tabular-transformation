import abc

class Transformation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, data):
        pass