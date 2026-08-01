import torch
import numpy as np

from core.module import Module
from core.encoder import Encoder
from core.dynamics import Dynamics
from core.decoder import Decoder

class WorldModel(Module):

    def __init__(self,
                 latent_size=128,
                 hidden_size=128,
                 image_size=64):

        super().__init__()

        self.encoder = Encoder(
            latent_size=latent_size
        )

        self.dynamics = Dynamics(
            latent_size=latent_size,
            hidden_size=hidden_size
        )

        self.decoder = Decoder(
            latent_size=latent_size,
            image_size=image_size
        )

    def forward(self,
                frame,
                action,
                hidden):

        #  Encode the current frame
        latent = self.encoder(frame)

        # Predict the next latent state
        hidden = self.dynamics(
            latent,
            action,
            hidden
        )

        # Decode the predicted latent state
        prediction = self.decoder(hidden)

        return prediction, hidden

    def init_hidden(self,
                batch_size):

        return self.dynamics.init_hidden(
            batch_size
        )

    def parameters(self):

        params = []

        params.extend(
            self.encoder.parameters()
        )

        params.extend(
            self.dynamics.parameters()
        )

        params.extend(
            self.decoder.parameters()
        )

        return params

    def state_dict(self):

        state = {}

        for index, parameter in enumerate(self.parameters()):

            state[f"param_{index}"] = (
                parameter.data.detach().cpu().numpy()
            )

        return state   

    def load(self, filepath):

        weights = np.load(filepath)

        parameters = self.parameters()

        for index, parameter in enumerate(parameters):

            parameter.data.copy_(

                torch.tensor(
                    weights[f"param_{index}"]
                )

            )