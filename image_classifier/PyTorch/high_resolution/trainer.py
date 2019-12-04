import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm
import math

# 画像や音声などの信号の複合問題でよく使われる評価関数を指標として用いる
def psnr(mse, max_value = 1.0):
    return 10 * math.log10(max_value**2/mse)

def trainer(model, train_loader, test_loader, epoch_num):

    # GPUが使えるなら使う
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We use", device)

    train_losses = []
    train_acc = []
    val_losses = []

    model.to(device)
    # ネットワークがある程度固定化できれば高速化する
    torch.backends.cudnn.benchmark = True
    loss_func = nn.MSELoss()    # 損失関数
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epoch_num):

        print("epoch{}/{}".format(epoch + 1, epoch_num))
        print("=======================================")

        # 訓練モードと検証モードを交互に行う
        for phase in ["train", "val"]:

            if phase == "train":
                # 訓練モードにする
                model.train()
                running_loss = 0
                n = 0
                score = 0

                for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):

                    xx = xx.to(device)
                    yy = yy.to(device)

                    output = model(xx)
                    loss = loss_func(output, yy)
                    # 以前の誤差逆伝播の勾配の初期化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    n += len(xx)

                train_losses.append(running_loss/len(train_loader))

            if phase == "val":

                # 検証モードにする. こうすることで、DropoutやBatchNormalを無効化する
                model.eval()
                ys = []
                y_preds = []

                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    with torch.no_grad():
                        y_pred = model(x)

                    ys.append(y)
                    y_preds.append(y_pred)

                # ミニバッチごとの計算を一つにまとめる
                ys = torch.cat(ys, dim = 0)   # ysを縦方向に結合する
                y_preds = torch.cat(y_preds, dim = 0)

                # MSEを計算
                score = F.mse_loss(y_preds, ys).item()
                
                val_losses.append(score)

                # print(, flush = True)にすることにより, 重い処理があってもこのprint()文を表示させるというoption
                print("{}回目のepochでの訓練Loss: {} 検証Loss: {}".format(epoch + 1, train_losses[-1], val_losses[-1]), flush = True)


    return train_losses, val_losses
