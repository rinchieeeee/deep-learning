from torchvision.datasets import FashionMNIST
from torchvision import transforms

# get the training data
train_data = FashionMNIST("data/FashionMNIST", train = True, 
                            download = True, 
                            transform = transforms.ToTensor()    # このままだとPIL形式なので、これをRGBの順の形式にかえる
                        )  

# make the test data
test_data = FashionMNIST("data/FashionMNIST",
                            train = False, 
                            download = True, 
                            transform = transforms.ToTensor()
                            )


from torch.utils.data import TensorDataset, DataLoader
# make the data loader 
batch_size = 128
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size = batch_size, shuffle=False)


# CNN の構築

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class Fashion_CNN(nn.Module):

    def __init__(self):

        super(Fashion_CNN, self).__init__()

        # define the network's layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.batchnormal1 = nn.BatchNorm2d(32)   # 32 is output number
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2d = nn.Conv2d(32, 64, 5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.batchnormal2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(1024, 200)
        self.batchnormal3 = nn.BatchNorm1d(200)
        self.dropout3 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = self.batchnormal1(x)
        x = self.dropout1(x)
        x = self.conv2d(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = self.batchnormal2(x)
        x = self.dropout2(x)
        size = x.size()
        x = x.view(size[0], -1)     # Flatten layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.batchnormal3(x)
        x = self.dropout3(x)
        output = self.fc2(x)
        return output


train_losses = []
train_acc = []
y_preds = []
val_acc = []

import tqdm

def trainer(model, train_data_loader, test_data_loader, epoch_num):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("use", device)
    
    model.to(device)
    # ネットワークをある程度固定化できれば高速化する
    torch.backends.cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    

    for epoch in range(epoch_num):
        
        print("epoch{}/{}".format(epoch + 1, epoch_num))
        print("====================================")

        # 訓練モードと検証モードを交互に行う
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()

            else:
                model.eval()
                ys = []
                y_preds = []

            running_loss = 0.0

            n = 0        # 訓練に使った画像の数
            n_acc = 0    # 正解数
            for i, (xx, yy) in tqdm.tqdm(enumerate(train_data_loader), total = len(train_data_loader)):
                xx = xx.to(device)
                yy = yy.to(device)
                optimizer.zero_grad()

                if phase == "train":
                    output = model(xx)
                    loss = criterion(output, yy)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    n += len(xx)
                    _, y_pred = output.max(1)     # ラベルの予想
                    n_acc += (yy == y_pred).float().sum().item()   # 正解数

            if phase == "val":
                for x, y in test_data_loader:
                    x = x.to(device)
                    y = y.to(device)
                    # 確率が最大のクラスを予測.また、推論の計算だけなので、自動微分はいらないので明言する
                    with torch.no_grad():
                        _, y_pred = model(x).max(1)
                            
                        
                    #print(ys)
                    ys.append(y)
                    y_preds.append(y_pred)

                # ミニバッチごとの予測結果を一つにまとめる
                ys = torch.cat(ys, dim = 0)     # dim = 0で縦方向にysを結合させていく
                y_preds = torch.cat(y_preds)
                # 予測精度を計算
                acc = (ys == y_preds).float().sum()/len(ys)



            if phase == "train":
                train_losses.append(running_loss/i)
                train_acc.append(n_acc/n)     # 訓練データの予測精度

            # 検証データの予測精度
            if phase == "val":
                val_acc.append(acc.item())

                # このepochでの結果を表示
                print("訓練loss: {} 訓練精度: {} 検証精度{}".format(train_losses[-1], train_acc[-1], val_acc[-1]), flush = True)



