
import torch

class Parameter:
    
    def __init__(self,data: torch.Tensor,requires_grad: bool = True):

        if not isinstance(data,torch.Tensor):
            raise TypeError("Parameter data must be a torch.Tensor")
        
        self.data = (data.float().clone().detach().requires_grad_(requires_grad))

        self.requires_grad = requires_grad

    #utility functions

    def zero_grad(self):
        if self.data.grad is not None:
            self.data.grad.zero_()

    def clone(self):
        
        copied = Parameter(
            self.data.clone(),
            self.requires_grad
        )

        return copied

    #properties

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    #debug priting

    def __repr__(self):
        return(
            "Parameter("
            f"shape ={tuple(self.shape)}, "
            f"requires_grad ={self.requires_grad}) "
        )