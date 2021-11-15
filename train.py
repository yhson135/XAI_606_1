import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import argparse


torch.manual_seed(314)
torch.cuda.manual_seed(314)
torch.cuda.manual_seed_all(314)
np.random.seed(314)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(314)


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, out_features)

        self.relu = nn.ReLU()

        self.drop = nn.Dropout(p=args.drop)
        # self.bn1 = nn.BatchNorm1d(hidden_features)
        # self.bn2 = nn.BatchNorm1d(hidden_features)
        # self.bn3 = nn.BatchNorm1d(hidden_features)
        #
        # self.lin = nn.Sequential(self.fc1, self.bn1, self.relu,
        #                          self.fc2, self.bn2, self.relu,
        #                          self.fc3, self.bn3, self.relu,
        #                          self.fc4
        #                          )

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        # x = self.drop(F.relu(self.fc2(x)))
        x = self.drop(F.relu(self.fc3(x)))
        x = self.fc4(x)
        # out = self.lin(x)
        return x


class Dataset(Dataset):
    def __init__(self, dataset, device):
        self.data = pd.read_csv('data/' + dataset + '.csv')

        encoder = LabelEncoder()

        self.data['model'] = encoder.fit_transform(self.data['model'])
        self.data['transmission'] = encoder.fit_transform(self.data['transmission'])
        self.data['fuelType'] = encoder.fit_transform(self.data['fuelType'])

        self.x_train = self.data.drop('price', axis=1)
        self.y_train = self.data['price']

        scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        self.x_train = scaler.fit_transform(self.x_train)
        self.y_train = self.y_train.values

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = torch.Tensor(self.x_train[idx]).to(device)
        y = torch.Tensor(self.y_train)[idx].to(device)

        return x, y


parser = argparse.ArgumentParser()
# timestamp = datetime.today().strftime("_%Y%m%d%H%M%S")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--features", type=list, default=[8, 128, 1])
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-7)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--drop", type=float, default=0.2)
args = parser.parse_args()




batch_size = args.batch_size
in_features = args.features[0]
hidden_features = args.features[1]
out_features = args.features[2]
lr = args.lr
weight_decay = args.weight_decay
n_epochs = args.n_epochs

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':

    train_data = Dataset('train', device)
    val_data = Dataset('val', device)
    test_data = Dataset('test', device)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model = MultivariateLinearRegressionModel(in_features, hidden_features, out_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    min_val_loss = 1e08
    best_val_test_loss = 1e08

    for epoch in range(n_epochs+1):
        train_loss = 0.0
        model.train()
        for batch_idx, (x_train, y_train) in enumerate(trainloader):
            pred = model(x_train)
            loss = F.mse_loss(pred, y_train.unsqueeze(-1))
            # loss = F.mse_loss(pred.squeeze(-1), y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += np.sqrt(loss.item())

        model.eval()
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            for x_val, y_val in validloader:
                pred = model(x_val)
                loss = F.mse_loss(pred, y_val.unsqueeze(-1))
                # loss = F.mse_loss(pred.squeeze(-1), y_val)
                valid_loss += np.sqrt(loss.item())

            model.eval()
            test_loss = 0.0
            for x_test, y_test in testloader:
                pred = model(x_test)
                loss = F.mse_loss(pred, y_test.unsqueeze(-1))
                # loss = F.mse_loss(pred.squeeze(-1), y_test)
                test_loss += np.sqrt(loss.item())
            if valid_loss < min_val_loss:
                min_val_loss = valid_loss
                best_val_test_loss = test_loss

        if epoch % 10 == 0:
            print('Epoch {:4d}/{} Train_Loss: {:.6f} Val_Loss: {:.6f} Test_Loss: {:.6f}'.format(
                epoch, n_epochs, train_loss / len(trainloader), valid_loss / len(validloader), test_loss / len(testloader)
            ))

    print(f"best test loss using min val loss VAL: {min_val_loss:.6f}, TEST: {best_val_test_loss:.6f}")
