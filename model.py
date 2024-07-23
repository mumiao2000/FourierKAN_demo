import torch
import torch.nn as nn

class FourierKANLayer(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, grid=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid= grid
        self.coefficient = nn.Parameter(torch.zeros(in_dim * 2 * (grid + 1), out_dim))
        nn.init.xavier_normal_(self.coefficient, 0.1)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        hidden_size = self.in_dim * (self.grid + 1) * 2
        X = torch.stack([torch.cos(i * X) for i in range(self.grid + 1)] + [torch.sin(i * X) for i in range(self.grid + 1)], -1)
        X = X.reshape(-1, hidden_size)
        Y = X @ self.coefficient
        return Y

class FourierKAN(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:list[int], out_dim:int, grid=3):
        super().__init__()
        dim = [in_dim] + hidden_dim + [out_dim]
        self.kan = nn.Sequential(*[FourierKANLayer(dim[i], dim[i + 1], grid) for i in range(len(dim) - 1)])

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        Y = self.kan(X)
        return Y