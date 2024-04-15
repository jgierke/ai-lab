from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


n_epochs = 100
batch_size_train = 16
batch_size_test = 64
learning_rate = 0.01
momentum = 0.5
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
device = torch.device("cpu")
#device = torch.device("cuda")
#device = torch.device("mps") # for GPU usage on Apple Silicon

columns = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked", "Survived"]
csv_path = '../data/titanic.csv'


def impute_NaNs(df, drop=False):
  if drop:
    dfc = df.dropna()
    return dfc
  dfc = df.copy()
  categorical_columns = dfc.select_dtypes(exclude=np.number).columns
  imp_freq = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
  dfc.loc[:, categorical_columns] = imp_freq.fit_transform(dfc[categorical_columns])

  numeric_columns = dfc.select_dtypes(include=np.number).columns
  imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
  dfc.loc[:, numeric_columns] = imp_mean.fit_transform(dfc[numeric_columns])
  return dfc


def scale(X_train, X_test):
  scaler = MinMaxScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  return X_train_scaled, X_test_scaled, scaler


def split_data(df):
  X = df.drop(columns=["Survived"])
  y = df["Survived"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
  X_train = X_train.reset_index(drop=True)
  X_test = X_test.reset_index(drop=True)
  y_train = y_train.reset_index(drop=True)
  y_test = y_test.reset_index(drop=True)

  return X_train, X_test, y_train, y_test


def prepare_data(df):
  df = impute_NaNs(df, drop=True)
  df = pd.get_dummies(df)
  X_train, X_test, y_train, y_test = split_data(df)
  X_train_scaled, X_test_scaled, _ = scale(X_train, X_test)

  return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns


titanic_df = pd.read_csv(csv_path)[columns]
X_train_scaled, _, _, _, _ = prepare_data(titanic_df)


class CustomTitanic(Dataset):
  def __init__(self, df_path, train=True):
    columns = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked", "Survived"]
    titanic_df = pd.read_csv(df_path)[columns]
    X_train_scaled, X_test_scaled, y_train, y_test, new_columns = prepare_data(titanic_df)

    X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    self.X = X_train_scaled if train else X_test_scaled
    self.y = y_train if train else y_test

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]


titanic_train = CustomTitanic(csv_path, train=True)
titanic_val = CustomTitanic(csv_path, train=False)

train_loader = DataLoader(titanic_train, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(titanic_val, batch_size=batch_size_test, shuffle=False)


class NN(torch.nn.Module):
  def __init__(self, D_in, H, D_out, dropout=0.5):
    super(NN, self).__init__()
    self.hidden = nn.ModuleList()
    self.hidden.append(nn.Linear(D_in, H[0]))
    if dropout > 0:
      self.hidden.append(nn.Dropout(dropout))
    self.hidden.append(nn.ReLU())

    for i in range(1, len(H)):
      self.hidden.append(nn.Linear(H[i-1], H[i]))
      if dropout > 0:
        self.hidden.append(nn.Dropout(dropout))
      self.hidden.append(nn.ReLU())

    self.output = nn.Linear(H[-1], D_out)

  def forward(self, x):
    for layer in self.hidden:
      x = layer(x)
    x = self.output(x)
    return F.sigmoid(x).squeeze()


in_dim = X_train_scaled.shape[1]
out_dim = 1
models_to_train = {
  "model_1": NN(in_dim, [3], out_dim, dropout=0.0).to(device),
  "model_2": NN(in_dim, [16], out_dim, dropout=0.0).to(device),
  "model_3": NN(in_dim, [64, 64], out_dim, dropout=0.0).to(device),
  "model_4": NN(in_dim, [64, 64], out_dim, dropout=0.5).to(device),
  "model_5": NN(in_dim, [512, 256, 128], out_dim, dropout=0.0).to(device),
  "model_6": NN(in_dim, [512, 256, 128], out_dim, dropout=0.5).to(device),
  "model_7": NN(in_dim, [512, 256, 128], out_dim, dropout=0.8).to(device),
  "model_8": NN(in_dim, [1024, 512, 256, 128], out_dim, dropout=0.0).to(device),
  "model_9": NN(in_dim, [1024, 512, 256, 128], out_dim, dropout=0.5).to(device),
  "model_10": NN(in_dim, [1024, 512, 256, 128], out_dim, dropout=0.8).to(device),
}

loss_fn = nn.BCELoss()


def train(n_epochs, model, device, train_loader, learning_rate, momentum, loss_fn):
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
  train_data = defaultdict(list)

  for ep in range(n_epochs):
    model.train()
    epoch_losses = list()
    epoch_acc = list()
    for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      y_pred = model(data)
      loss = loss_fn(y_pred, target)
      train_data["loss"].append(loss.item())
      epoch_losses.append(loss.item())
      acc = ((y_pred > 0.5).float() == target).float().mean().item()
      epoch_acc.append(acc)
      train_data["acc"].append(acc)
      loss.backward()
      optimizer.step()

    train_data["epoch_loss"].append(sum(epoch_losses) / len(epoch_losses))
    train_data["epoch_acc"].append(sum(epoch_acc) / len(epoch_acc))
    test_results = test(model, device, val_loader, loss_fn)
    train_data["test_loss"].append(sum(test_results["loss"]) / len(test_results["loss"]))
    train_data["test_acc"].append(sum(test_results["acc"]) / len(test_results["acc"]))

  return train_data


def test(model, device, test_loader, loss_fn):
  model.eval()
  test_data = defaultdict(list)

  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      y_pred = model(data)
      loss = loss_fn(y_pred, target)
      test_data["loss"].append(loss.item())
      acc = ((y_pred > 0.5).float() == target).float().mean().item()
      test_data["acc"].append(acc)
  return test_data


train_data = dict()
test_data = dict()

for model_name, model in models_to_train.items():
  train_data[model_name] = train(n_epochs, model, device, train_loader, learning_rate, momentum, loss_fn)
  test_data[model_name] = test(model, device, val_loader, loss_fn)
  train_mean_loss = sum(train_data[model_name]["loss"]) / len(train_data[model_name]["loss"])
  train_mean_acc = sum(train_data[model_name]["acc"]) / len(train_data[model_name]["acc"]) * 100
  test_mean_loss = sum(test_data[model_name]["loss"]) / len(test_data[model_name]["loss"])
  test_mean_acc = sum(test_data[model_name]["acc"]) / len(test_data[model_name]["acc"]) * 100

  print(f"#### {model_name} ####")
  print(f"Train mean loss: {train_mean_loss:.4f}")
  print(f"Train mean accuracy: {train_mean_acc:.2f}%")
  print(f"Test mean loss: {test_mean_loss:.4f}")
  print(f"Test mean accuracy: {test_mean_acc:.2f}%")
  print()


def print_metrics(train_data, test_data, title):
  print(title)
  print(f"Training: loss: {train_data['epoch_loss'][-1]:.2f}, acc: {train_data['epoch_acc'][-1]:.2f}")
  print(f"Testing: loss: {sum(test_data['loss']) / len(test_data['loss']):.2f}, acc: {sum(test_data['acc']) / len(test_data['acc']):.2f}")


def plot_metrics(train_data, title):
  fig, axs = plt.subplots(1, 2, figsize=(12, 4))
  axs[0].plot(train_data["epoch_loss"], label="train")
  axs[0].plot(train_data["test_loss"], label="test")
  axs[0].set_title("Loss")
  axs[0].legend()
  axs[1].plot(train_data["epoch_acc"], label="train")
  axs[1].plot(train_data["test_acc"], label="test")
  axs[1].set_title("Accuracy")
  axs[1].legend()

  plt.suptitle(title)
  plt.tight_layout()
  plt.show()


for model_name in models_to_train.keys():
  print_metrics(train_data[model_name], test_data[model_name], model_name)
  plot_metrics(train_data[model_name], model_name)
