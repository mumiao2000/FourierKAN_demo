import torch
import torch.nn as nn
import torch.optim as optim
import basic_model

def _FourierX(X:torch.Tensor, hidden_dim:int, grid=5) -> torch.Tensor:
    i_values = torch.arange(grid, dtype=X.dtype, device=X.device).reshape(grid, 1, 1)
    cos_values = torch.cos(i_values * X).permute(1, 2, 0)
    sin_values = torch.sin(i_values * X).permute(1, 2, 0)
    X = torch.cat([cos_values, sin_values], dim=-1).reshape(-1, hidden_dim)
    return X

class FourierKANLayer(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, grid=5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid= grid
        self.coefficient = nn.Parameter(torch.zeros(in_dim * 2 * grid, out_dim))
        nn.init.xavier_normal_(self.coefficient, 0.1)
        self.min_range = torch.ones(in_dim) * 1e9
        self.max_range = torch.ones(in_dim) * -1e9

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        self.min_range = torch.min(torch.min(X, dim=0).values, self.min_range.to(X.device)).detach()
        self.max_range = torch.max(torch.max(X, dim=0).values, self.max_range.to(X.device)).detach()
        Y = _FourierX(X, self.in_dim * self.grid * 2, self.grid) @ self.coefficient
        return Y

class FourierKAN(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:list[int], out_dim:int, grid=5):
        super().__init__()
        dim = [in_dim] + hidden_dim + [out_dim]
        self.grid = grid
        self.KAN_layer = [FourierKANLayer(dim[i], dim[i + 1], grid) for i in range(len(dim) - 1)]
        self.KAN = nn.Sequential(*self.KAN_layer)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        Y = self.KAN(X)
        return Y

    def formula(self) -> str:
        coefficient = list()
        min_range = list()
        max_range = list()
        for layer in self.KAN_layer:
            coefficient = [layer.coefficient.detach().reshape(layer.in_dim, 2 * layer.grid, layer.out_dim)] + coefficient
            min_range = [layer.min_range.to(layer.coefficient.device)] + min_range
            max_range = [layer.max_range.to(layer.coefficient.device)] + max_range
        return self._dfs_formula(coefficient, min_range, max_range, 0)

    def _dfs_formula(
            self, coefficient:list[torch.Tensor],
            min_range:list[torch.Tensor], max_range:list[torch.Tensor],
            layer_idx:int) -> str:
        if layer_idx == len(coefficient): return ''
        coefficient_layer = coefficient[layer_idx]
        min_range_layer = min_range[layer_idx]
        max_range_layer = max_range[layer_idx]

        formula = str()
        for i in range(coefficient_layer.shape[-1]):
            if i != 0: formula += '+'
            for j in range(coefficient_layer.shape[0]):
                if j != 0: formula += '+'
                operator, params = self._fit(coefficient_layer[j, :, i], min_range_layer[j], max_range_layer[j])
                # prune
                if operator in ['']: break
                # exp, log, sqrt, tanh, sin
                if operator in ['exp','tanh','sin']:
                    formula += str(round(params[0], 3)) + '*' + operator + '(' + str(round(params[1], 3)) + '*'
                    formula += self._dfs_formula(coefficient, min_range, max_range, layer_idx + 1)
                    if formula[-1] == '*': formula += 'x' + str(j + 1)
                    formula += ')'
                # exp, log, sqrt, tanh, sin
                if operator in ['ln','sqrt']:
                    formula += str(round(params[0], 3)) + '*' + operator + '|' + str(round(params[1], 3)) + '*'
                    formula += self._dfs_formula(coefficient, min_range, max_range, layer_idx + 1)
                    if formula[-1] == '*': formula += 'x' + str(j + 1)
                    formula += '|'
                # abs
                if operator in ['abs']:
                    formula += str(round(params[0], 3)) + '*' + '|'
                    formula += self._dfs_formula(coefficient, min_range, max_range, layer_idx + 1)
                    if formula[-1] == '|': formula += 'x' + str(j + 1)
                    formula += '|'
                # x^0, x^1, x^2, x^3, x^4
                if operator in ['^0', '^1', '^2', '^3', '^4']:
                    formula += str(round(params[0], 3)) + '*' + '('
                    formula += self._dfs_formula(coefficient, min_range, max_range, layer_idx + 1)
                    if formula[-1] == '(': formula += 'x' + str(j + 1)
                    formula += ')' + operator
        return formula

    def _fit(self, coefficient:torch.Tensor, min_range:int, max_range:int, threshold=5e-2) -> tuple[str, list[float]]:
        if max_range - min_range < threshold: return '', []
        X = min_range + (max_range - min_range) * torch.rand((int(1e4), 1)).to(coefficient.device)
        Y = (_FourierX(X, self.grid * 2, self.grid) @ coefficient).reshape(-1, 1)
        def train_model(model, X, Y, criterion, optimizer):
            def closure():
                optimizer.zero_grad()
                Y_pred = model(X)
                loss = criterion(Y_pred, Y)
                loss.backward()
                return loss
            optimizer.step(closure)
            return closure().item()
        def get_params(operator, operator_type):
            if operator_type in ['^0', '^1', '^2', '^3', '^4', 'abs']:
                return [operator.a.item()]
            elif operator_type in ['exp', 'ln', 'sqrt', 'tanh', 'sin']:
                return [operator.a.item(), operator.b.item()]
        operators = {
            '^0': basic_model.PolyModel(0),
            '^1': basic_model.PolyModel(1),
            '^2': basic_model.PolyModel(2),
            '^3': basic_model.PolyModel(3),
            '^4': basic_model.PolyModel(4),
            'exp': basic_model.ExpModel(),
            'ln': basic_model.LnModel(),
            'sqrt': basic_model.SqrtModel(),
            'tanh': basic_model.TanhModel(),
            'sin': basic_model.SinModel(),
            'abs': basic_model.AbsModel()}
        loss_dict = dict()
        for operator_type, operator in operators.items():
            operator.to(X.device)
            creterion = nn.MSELoss()
            optimizer = optim.LBFGS(operator.parameters())
            loss = train_model(operator, X, Y, creterion, optimizer)
            loss_dict[operator_type] = loss
        best_operator_type = min(loss_dict, key=loss_dict.get)
        print(loss_dict[best_operator_type])
        best_operator = operators[best_operator_type]
        best_params = get_params(best_operator, best_operator_type)
        return best_operator_type, best_params

if __name__ == '__main__':
    X = torch.rand((2048, 2), dtype=torch.float32)
    net = FourierKAN(2, [1], 1, 2)
    net.train()
    net(X)
    formula = net.formula()
    print(formula)