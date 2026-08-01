from core.module import Module
from core.linear import Linear
from core.activation import ReLU, Sigmoid
from core.reshape import Reshape

class Decoder(Module):

    def __init__(self,
                 latent_size=128,
                 image_size=64):

        super().__init__()

        self.fc1 = Linear(
            latent_size,
            512
        )

        self.relu1 = ReLU()

        self.fc2 = Linear(
            512,
            2048
        )

        self.relu2 = ReLU()

        self.fc3 = Linear(
            2048,
            image_size * image_size
        )

        self.sigmoid = Sigmoid()

        self.reshape = Reshape(
            -1,
            1,
            image_size,
            image_size
        )

    def forward(self, latent):

        x = self.fc1(latent)

        x = self.relu1(x)

        x = self.fc2(x)

        x = self.relu2(x)

        x = self.fc3(x)

        x = self.sigmoid(x)

        x = self.reshape(x)

        return x

    def parameters(self):

        params = []

        params.extend(
            self.fc1.parameters()
        )

        params.extend(
            self.fc2.parameters()
        )

        params.extend(
            self.fc3.parameters()
        )

        return params
    def children(self):

        return [

            self.fc1,

            self.relu1,

            self.fc2,

            self.relu2,

            self.fc3,

            self.sigmoid,

            self.reshape

        ]