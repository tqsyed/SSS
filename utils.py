
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
import torch

def normalize(X_train, X_test):

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test

def get_data_loaders(X_train, y_train, X_test, y_test,device):

    train_target = torch.from_numpy(y_train).long().to(device)#torch.tensor(y_train.astype(np.float32))
    train_x = torch.from_numpy(X_train).float().to(device)

    test_target = torch.from_numpy(y_test).long().to(device)#torch.tensor(y_test.astype(np.float32))
    test_x = torch.from_numpy(X_test).float().to(device)

    train_tensor =  torch.utils.data.TensorDataset(train_x, train_target)
    test_tensor =   torch.utils.data.TensorDataset(test_x, test_target)

    train_loader =  torch.utils.data.DataLoader(dataset = train_tensor, batch_size = 100, shuffle = True)
    test_loader =  torch.utils.data.DataLoader(dataset = test_tensor, batch_size = 100, shuffle = True)

    return train_loader, test_loader


