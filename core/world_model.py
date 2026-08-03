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
                batch_size,
                device = None,
                dtype = torch.float32):

        return self.dynamics.init_hidden(
            batch_size = batch_size,
            device=device,
            dtype=dtype
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
        
        with torch.no_grad():

            for index, parameter in enumerate(parameters):

                parameter.data.copy_(

                    torch.tensor(
                        weights[f"param_{index}"],
                        dtype=parameter.data.dtype
                    )

                )

        epoch = int(weights["epoch"]) if "epoch" in weights else 0
        best_loss = float(weights["best_loss"]) if "best_loss" in weights else float("inf")

        return epoch, best_loss
            
    def save(self, filepath,epoch = 0,best_loss=float("inf")):
        state = self.state_dict()

        state["epoch"] = epoch
        state["best_loss"] = best_loss
        np.savez(
            filepath,
            **state
        )
    def to(self, device):

        for parameter in self.parameters():

            parameter.data = (
                parameter.data
                .detach()
                .to(device)
                .requires_grad_(parameter.requires_grad)
            )

        return self