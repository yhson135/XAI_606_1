import torch
import torch.nn as nn
import torch.nn.functional as F
from train import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn.metrics import accuracy_score
import argparse




torch.manual_seed(314)
torch.cuda.manual_seed(314)
torch.cuda.manual_seed_all(314)
np.random.seed(314)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(314)

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
train_numpy = Dataset("train", device).x_train
label_numpy = Dataset("train", device).y_train
valid_numpy = Dataset("val", device).x_train
valid_label_numpy = Dataset("val", device).y_train
test_numpy = Dataset("test", device).x_train
test_label_numpy = Dataset("test", device).y_train

# train_numpy = np.concatenate((train_numpy, valid_numpy), axis=0)
# label_numpy = np.concatenate((label_numpy, valid_label_numpy), axis=0)




reg = LinearRegression(normalize=False, fit_intercept=True).fit(train_numpy, label_numpy)

out = reg.predict(test_numpy)
test_loss = ((out - test_label_numpy)**2).mean()
test_loss = np.sqrt(test_loss)

print(f"Linear regression test_loss: {test_loss:.6f}")


reg = ElasticNet().fit(train_numpy, label_numpy)
out = reg.predict(test_numpy)
test_loss = ((out - test_label_numpy)**2).mean()
test_loss = np.sqrt(test_loss)

print(f"Elastic regression test_loss: {test_loss:.6f}")


reg = Lasso(max_iter=1000, tol=1e-4).fit(train_numpy, label_numpy)
out = reg.predict(test_numpy)
test_loss = ((out - test_label_numpy)**2).mean()
test_loss = np.sqrt(test_loss)

print(f"Lasso regression test_loss: {test_loss:.6f}")

reg = Lasso(max_iter=1000, tol=1e-3, warm_start=True).fit(train_numpy, label_numpy)
out = reg.predict(test_numpy)
test_loss = ((out - test_label_numpy)**2).mean()
test_loss = np.sqrt(test_loss)

print(f"Lasso regression test_loss: {test_loss:.6f}")

reg = Lasso(max_iter=1000, tol=1e-2).fit(train_numpy, label_numpy)
out = reg.predict(test_numpy)
test_loss = ((out - test_label_numpy)**2).mean()
test_loss = np.sqrt(test_loss)

print(f"Lasso regression test_loss: {test_loss:.6f}")

min_val_loss = 1e08
best_val_test_loss = test_loss
max_alpha = 0
for alpha in range(30, 250, 5):
    reg = Ridge(max_iter=1000, tol=1e-4, alpha=alpha).fit(train_numpy, label_numpy)
    val_out = reg.predict(valid_numpy)
    val_loss = ((val_out - valid_label_numpy) ** 2).mean()
    val_loss = np.sqrt(val_loss)
    test_out = reg.predict(test_numpy)
    test_loss = ((test_out - test_label_numpy) ** 2).mean()
    test_loss = np.sqrt(test_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        max_alpha = alpha
        best_val_test_loss = test_loss

    print(f"Ridge regression {alpha} val_loss: {val_loss:.6f} test_loss: {test_loss:.6f}")

print(f"best test loss using min val loss alpha:{max_alpha} VAL: {min_val_loss:.6f}, TEST: {best_val_test_loss:.6f}")

min_val_loss = 1e08
best_val_test_loss = test_loss
max_c = 0


for c in range(1, 10):
    SVM = svm.SVC(C=c)  # 10, 20, 100, /
    SVM.fit(train_numpy, label_numpy)
    val_out = SVM.predict(valid_numpy)
    val_loss = ((val_out - valid_label_numpy) ** 2).mean()
    val_loss = np.sqrt(val_loss)
    test_out = SVM.predict(test_numpy)
    test_loss = ((test_out - test_label_numpy) ** 2).mean()
    test_loss = np.sqrt(test_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        max_c = c
        best_val_test_loss = test_loss
    print(f"SVM regression max_c {c} val_loss: {val_loss:.6f} test_loss: {test_loss:.6f}")

print(f"best test loss using min val loss alpha:{max_c} VAL: {min_val_loss:.6f}, TEST: {best_val_test_loss:.6f}")

for c in range(1, 10):
    SVM = svm.SVR(C=c)  # 10, 20, 100, /
    SVM.fit(train_numpy, label_numpy)
    val_out = SVM.predict(valid_numpy)
    val_loss = ((val_out - valid_label_numpy) ** 2).mean()
    val_loss = np.sqrt(val_loss)
    test_out = SVM.predict(test_numpy)
    test_loss = ((test_out - test_label_numpy) ** 2).mean()
    test_loss = np.sqrt(test_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        max_c = c
        best_val_test_loss = test_loss
    print(f"SVR regression max_c {c} val_loss: {val_loss:.6f} test_loss: {test_loss:.6f}")

print(f"best test loss using min val loss alpha:{max_c} VAL: {min_val_loss:.6f}, TEST: {best_val_test_loss:.6f}")


