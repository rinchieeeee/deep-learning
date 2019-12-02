import torch
import torch.nn as nn
import torch.nn.functional as F
class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()
        # define network
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        output = self.fc5(x)
        return output

from torch import optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

digits = load_digits()     # loading data

X = digits.data
Y = digits.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

X_train = torch.tensor(X_train, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)
Y_train = torch.tensor(Y_train, dtype = torch.int64)
Y_test = torch.tensor(Y_test, dtype = torch.int64)

# 訓練用データでDataLoaderを作成. tensor型に変換し、データセット化する
dataset = TensorDataset(X_train, Y_train)
loader = DataLoader(dataset, batch_size = 32, shuffle = True)

train_losses = []
test_losses = []
def trainer(model, dataloader, X_test, Y_test):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("use", device)

    model.to(device)
    # ネットワークをある程度固定できれば、高速化する
    torch.backends.cudnn.benchmark = True
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    #train_losses = []
    #test_losses = []
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    
    for epoch in range(100):
        print("epoch {}/{}".format(epoch + 1, 100))
        print("===================================")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0
            for i, (xx, yy) in enumerate(dataloader):
                xx = xx.to(device)
                yy = yy.to(device)
                optimizer.zero_grad()
                if phase == "train":
                    y_pred = model(xx)
                    loss = loss_fn(y_pred, yy)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
            if phase == "train":
                #print(i)
                train_losses.append(running_loss/(i))
                
            if phase == "val":

                # validation
                y_pred = model(X_test)
                test_loss = loss_fn(y_pred, Y_test)
                test_losses.append(test_loss.item())


model = net()
trainer(model, loader, X_test = X_test, Y_test = Y_test)

plt.figure(figsize = (10, 8))
plt.plot(train_losses, label = "train")
plt.plot(test_losses, label = "test")
plt.legend()
plt.show()