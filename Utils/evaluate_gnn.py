import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import math
def model_train(model, loader, optimizer, criterion,device):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    y_true = []  # 收集真实值
    y_pred = []  # 收集预测值
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()  # 清空梯度
        out = model(batch).squeeze()  # 前向传播
        loss = criterion(out, batch.y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()
        y_true.append(batch.y.detach().cpu().numpy())
        y_pred.append(out.detach().cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    avg_loss = total_loss / len(loader)

    # print(
    #         "Train RMSE : %.3f | MAE : %.3f | R2: %.3f"
    #         % (rmse, mae, r2)
    #     )
    return avg_loss, r2, mae, rmse

def model_val(model, loader, criterion,device):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    y_true = []  # 收集真实值
    y_pred = []  # 收集预测值
    with torch.no_grad():  # 禁用梯度计算
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze()
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    avg_loss = total_loss / len(loader)
    # print("Val RMSE : %.3f | MAE : %.3f | R2: %.3f" % (rmse, mae, r2))
    return avg_loss, r2, mae, rmse

def model_test(model, loader,device):
    model.eval()  # 设置模型为评估模式
    y_true = []  # 真实值
    y_pred = []  # 预测值
    with torch.no_grad():  # 禁用梯度计算
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze()
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mae, rmse, y_true, y_pred


class EarlyStopping:
  

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=50, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_mse_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_mse, model):

        if self.best_score is None:
            self.best_score = val_mse
            self.save_checkpoint(val_mse, model)
        elif val_mse > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_mse
            self.save_checkpoint(val_mse, model)
            self.counter = 0

    def save_checkpoint(self, val_mse, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_mse_min:.6f} --> {val_mse:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_mse_min = val_mse

def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()