import torch
from core.parameter import Parameter
from core.module import Module

class GRUCell(Module):


    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        std = (2.0 / input_size)**0.5
        #update the three gates
        self.W_z = Parameter(
            torch.randn(input_size,hidden_size) * std
        )
        self.U_z = Parameter(
            torch.randn(hidden_size,hidden_size) * std
        )
        self.b_z = Parameter(
            torch.zeros(hidden_size)
        )
        #reset the gate
        self.W_r = Parameter(
            torch.randn(input_size,hidden_size)* std
        )
        self.U_r = Parameter(
            torch.randn(hidden_size,hidden_size) * std
        )
        self.b_r = Parameter(
            torch.zeros(hidden_size)
        )

        #candidate state
        self.W_h = Parameter(
            torch.randn(input_size, hidden_size) * std
        )

        self.U_h = Parameter(
            torch.randn(hidden_size, hidden_size) * std
        )

        self.b_h = Parameter(
            torch.zeros(hidden_size)
        )

    def parameters(self):
        return [self.W_z,
                self.U_z,
                self.b_z,

                self.W_r,
                self.U_r,
                self.b_r,

                self.W_h,
                self.U_h,
                self.b_h
                ]
    def forward(self, x, h_prev):

        #update gate
        z = torch.sigmoid(

        x @ self.W_z.data +

        h_prev @ self.U_z.data +

        self.b_z.data

        )

        #reset gate
        r = torch.sigmoid(

        x @ self.W_r.data +

        h_prev @ self.U_r.data +

        self.b_r.data

        )
        #candidate hiiden state
        h_candidate = torch.tanh(

            x @ self.W_h.data +

            (r * h_prev) @ self.U_h.data +

            self.b_h.data

        )

        h = (
            (1-z) * h_prev + 
            z * h_candidate
        )
        return h