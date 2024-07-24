import torch
import torch.nn as nn

class PolyModel(nn.Module):
    def __init__(self, power:int):
        super().__init__()
        self.power = power
        self.a = nn.Parameter(torch.tensor(1.0))
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.a * X ** self.power

class ExpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.a * torch.exp(self.b * X)

class LnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.a * torch.log(torch.abs(self.b * X))

class SqrtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.a * torch.sqrt(torch.abs(self.b * X))

class TanhModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.a * torch.tanh(self.b * X)

class SinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.a * torch.sin(self.b * X)

class AbsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.a * torch.abs(X)