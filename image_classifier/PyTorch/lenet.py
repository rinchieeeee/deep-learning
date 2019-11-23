import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# 乱数の固定
torch.manual_seed(1)
np.random.seed(1)

class LeNet(nn.Module):

    def __init__(self, num_class):
        super(LeNet, self).__init__()
        # ネットワーク層の定義
        self.num_class = num_class
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, self.num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = F.sigmoid(self.maxpool1(x))
        x = self.conv2(x)
        x = F.sigmoid(self.maxpool2(x))
        x = x.view(-1, 5*5*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.softmax(x, dim = 1)
        return output



def trainer(model, dataloader_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかの確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使うのは:",  device)

    model.to(device)

    # ネットワークがある程度固定化できれば高速化する
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        print("epoch {}/{}".format(epoch + 1, num_epochs))
        print("====================")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()

            else:
                model.eval()

            epoch_loss = 0.0   # epochの損失和
            epoch_corrects = 0   # epochの正解数

            # データからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloader_dict[phase]):
                # GPUが使えるならデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizerを初期化
                optimizer.zero_grad()

                # forward計算
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(torch.log(outputs), labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はback-proba
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # loss.item()にはミニバッチでの平均損失が格納されているので
                                                                # inputs.size(0) = mini_batch_sizeをかけて合計損失にする
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)


                # epochごとのlossと正解率を表示
                epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
                epoch_acc = epoch_corrects.double()/len(dataloader_dict[phase].dataset)

                print("{} loss : {:.4f} Acc : {:.4f}".format(phase, epoch_loss, epoch_acc))


