import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tqdm
import model

epochs = 225
batch_size = 128
lr = 1.2e-3
wd = 1e-2
l1_lambda = 1e-2
n_components = 2

seq_len = 175

need_load = False
need_train = True
need_test = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data preprocessing
pca = PCA(n_components=n_components)
def get_data(csv, is_train=False):
    df = pd.read_csv(csv, encoding='utf-8')
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32).reshape(-1, 2 * seq_len)
    Y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)
    if is_train:
        global X_min, X_max, Y_min, Y_max
        X_min = torch.min(X, dim=0, keepdim=True).values
        X_max = torch.max(X, dim=0, keepdim=True).values
        Y_min = torch.min(Y, dim=0, keepdim=True).values
        Y_max = torch.max(Y, dim=0, keepdim=True).values
    eps = 1e-5
    X = (X - X_min) / (X_max - X_min + eps)
    # Y = (Y - Y_min) / (Y_max - Y_min + eps)
    X = pca.fit_transform(X.numpy()) if is_train else pca.transform(X.numpy())
    X = torch.tensor(X, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    return dataloader

train_dataloader = get_data('./data/train_data', True)
valid_dataloader = get_data('./data/valid_data')

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print(f'Explained variance ratio: {explained_variance_ratio}')
print(f'Cumulative explained variance: {cumulative_explained_variance}')


# net, optimizer and criterion
net = model.FourierKAN(n_components, [n_components * 2 + 1], 1, grid=12).to(device)
# net = model.MLP(n_components, [64, 512, 64], 1).to(device)
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
        l1_norm = sum(p.abs().sum() for p in net.parameters())
        back_loss = rmse_loss + l1_lambda * l1_norm
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

# print(net.formula())


# plot error plot
with torch.no_grad():
    real = list()
    pred = list()
    for data in valid_dataloader:
        X, Y = data[0].to(device), data[1].to(device)
        output = net(X)
        tmp_real = Y.cpu().reshape(-1).tolist()
        tmp_pred = output.cpu().reshape(-1).tolist()
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