from abc import ABC, abstractmethod


class Module(ABC):
    """
    Base class for every neural network component.
    """

    def __init__(self):
        self.training = True

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Every child class must implement a forward pass.
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        """
        Returns a list of trainable parameters.
        Child classes override this if they own Parameters.
        """
        return []
    def children(self):
        """
        Returns child modules.
        Child classes override this if they contain Modules.
        """
        return []

    def train(self):
        self.training = True

        for child in self.children():
            child.train()

    def eval(self):
        self.training = False

        for child in self.children():
            child.eval()