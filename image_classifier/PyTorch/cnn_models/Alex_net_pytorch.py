import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm

class AlexNet(nn.Module):

    def __init__(self):

        super(AlexNet, self).__init__()

        # define the network's layers
        self.conv1 = nn.Conv2d(3, 96, kernel_size = 11, stride = 4)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 0)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 0)
        self.fc1 = nn.Linear(256*5*5, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc_out = nn.Linear(4096, 2)

        self.Local_normalizer = nn.LocalResponseNorm(size = 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.Local_normalizer(x)
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.Local_normalizer(x)
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = nn.Dropout()(x)
        x = F.relu(self.fc2(x))
        x = nn.Dropout()(x)
        x = self.fc_out(x)
        x = F.softmax(x, dim = 1)
        
        return x



def trainer(model, train_data_loader, test_data_loader, epoch_num):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We use", device)

    model.to(device)

    torch.backends.cudnn.benchmark = True
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())  # モデルの全パラメータを最適化する
    train_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(epoch_num):
        
        print("epoch{}/{}".format(epoch + 1, epoch_num))
        print("==================================")

        # 訓練モードと検証モードを交互に行う
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                running_loss = 0.0

                n = 0
                n_acc = 0 # 正解数

                for i, (xx, yy) in tqdm.tqdm(enumerate(train_data_loader), total = len(train_data_loader)):
                    
                    xx = xx.to(device)
                    yy = yy.to(device)

                    output = model(xx)
                    loss = loss_func(output, yy)
                    # 以前の誤差逆伝播の勾配の初期化
                    optimizer.zero_grad()
                    
                    # 勾配伝播
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    n += len(xx)
                    _, y_pred = output.max(1)  # ラベルの予想
                    n_acc += (yy == y_pred).float().sum().item()  # 正解数
                    
                train_losses.append(running_loss)

                epoch_acc = n_acc/n
                train_acc.append(epoch_acc)

            
            if phase == "val":
                model.eval()
                ys = []
                y_preds = []

                for x, y in test_data_loader:
                    x = x.to(device)
                    y = y.to(device)
                    # 確率最大のクラスを予測
                    with torch.no_grad():
                        _, y_pred = model(x).max(1)

                    ys.append(y)
                    y_preds.append(y_pred)

                # ミニバッチごとの予測結果を一つにまとめる
                ys = torch.cat(ys, dim = 0)  # dim = 0で縦方向にysを結合させる
                y_preds = torch.cat(y_preds)

                acc = (ys == y_preds).float().sum().item()/len(y_preds)  # 1 epochでの正解率

                val_acc.append(acc)
            
                # このepochでの結果を表示
                print("訓練loss: {} 訓練精度: {} 検証精度{}".format(train_losses[-1], train_acc[-1], val_acc[-1]), flush = True)

    return train_acc, val_acc