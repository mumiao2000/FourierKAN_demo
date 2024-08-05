from kan import *
torch.set_default_dtype(torch.float64)
model = KAN(width=[2,5,1], grid=3, k=3, seed=42)

from kan.utils import create_dataset
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)
dataset['train_input'].shape, dataset['train_label'].shape

# plot KAN at initialization
model(dataset['train_input'])
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
model = model.prune()
model.fit(dataset, opt="LBFGS", steps=50)
model = model.refine(10)
model.fit(dataset, opt="LBFGS", steps=50)
LIB = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
model.auto_symbolic(lib=LIB)
model.fit(dataset, opt="LBFGS", steps=50)


from kan.utils import ex_round
print('\n', ex_round(model.symbolic_formula()[0][0],4))