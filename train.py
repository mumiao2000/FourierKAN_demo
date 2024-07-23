import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import model

epochs = 25
batch_size = 128
lr = 1.2e-3
wd = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# generate data
def get_dataloader(n, is_train=False):
    eps = 1e-5
    X = torch.rand((n, 2), dtype=torch.float32) * 2 - 1
    Y = (torch.sin(torch.exp(X[:, 0] ** 4 * X[:, 1] ** 3) * torch.sin(torch.pi * X[:, 0] + torch.exp(X[:, 1])) + 1.2 * torch.abs(0.2 + X[:, 1])) / (1 + torch.sqrt(torch.abs(X[:, 1])))).unsqueeze(1)
    Y = Y + torch.randn_like(Y, dtype=torch.float32) * 0.01
    # generate X_min, X_max, Y_min, Y_max
    if is_train:
        global X_min, X_max, Y_min, Y_max
        X_min = torch.min(X, dim=0, keepdim=True).values
        X_max = torch.max(X, dim=0, keepdim=True).values
        Y_min = torch.min(Y, dim=0, keepdim=True).values
        Y_max = torch.max(Y, dim=0, keepdim=True).values
    # normalization
    X = (X - X_min) / (X_max - X_min + eps)
    Y = (Y - Y_min) / (Y_max - Y_min + eps)
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    return dataloader

train_dataloader = get_dataloader(int(5e4), True)
valid_dataloader = get_dataloader(int(1e3))
test_dataloader = get_dataloader(int(1e3))

# net, optimizer and criterion
net = model.FourierKAN(2, [5, 11], 1).to(device)
param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Model Param Num: ', param_num)
criterion = nn.MSELoss().to(device)
optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

# one epoch
def one_epoch(data_loader, is_train=False):
    tmp_rmse_loss_list = list()
    for data in data_loader:
        X, Y = data[0].to(device), data[1].to(device)
        if is_train: optimizer.zero_grad()
        output = net(X)
        rmse_loss = torch.sqrt(criterion(output, Y))
        back_loss = rmse_loss
        if is_train: back_loss.backward()
        if is_train: optimizer.step()
        with torch.no_grad():
            tmp_rmse_loss_list.append(rmse_loss.item())
    return np.mean(tmp_rmse_loss_list)

# train & valid
print('\nTraining...')
for i in tqdm.tqdm(range(epochs)):
    # train
    net.train()
    train_rmse_loss = one_epoch(train_dataloader, True)
    scheduler.step()
    # validate
    net.eval()
    with torch.no_grad():
        valid_rmse_loss = one_epoch(valid_dataloader, False)
    print('Train RMSE Loss: ', train_rmse_loss)
    print('Valid RMSE Loss: ', valid_rmse_loss)
print('End Training')

# test
eps = 1e-5
print('\nTesting...')
net.eval()
with torch.no_grad():
    test_rmse_loss = one_epoch(test_dataloader, False)
print('Test  RMSE Loss: ', test_rmse_loss)
print('End Testing')
# plot error plot
with torch.no_grad():
    real = list()
    pred = list()
    for data in test_dataloader:
        X, Y = data[0].to(device), data[1].to(device)
        output = net(X)
        tmp_real = (Y.cpu() * (Y_max - Y_min + eps) + Y_min).reshape(-1).tolist()
        tmp_pred = (output.cpu() * (Y_max - Y_min + eps) + Y_min).reshape(-1).tolist()
        real = real + tmp_real
        pred = pred + tmp_pred
    real = np.array(real)
    pred = np.array(pred)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    # plot the main scatter plot
    ax.scatter(real, pred, alpha=0.1, s=10)
    ax.scatter(real, real, alpha=0.1, s=1)
    ax.set_xlabel('Real')
    ax.set_ylabel('Pred')
    # show the plot
    plt.tight_layout()
    plt.savefig('./figure/error.png', dpi=500)