import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm

def trainer(model, train_loader, test_loader, epoch_num, fc_params_only = True):

    train_losses = []
    train_acc = []
    y_preds = []
    val_acc = []
    # もし、GPUが使えるのなら使う
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("use: ", device)

    model.to(device)
    # ネットワークがある程度固定化できれば、高速化する
    torch.backends.cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()     # 損失関数

    # 最後の全結合層のみを学習させるかどうか(デフォルトはTrue)
    if fc_params_only:
        optimizer = optim.Adam(model.fc.parameters())
    else:
        optimizer = optim.Adam(model.parameters())

    for epoch in range(epoch_num):

        print("epoch{}/{}".format(epoch + 1, epoch_num))
        print("============================================")

        # 訓練モードと検証モードを交互に行う

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()

            else:
                model.eval()
                ys = []
                y_preds = []

            running_loss = 0

            n = 0      # ミニバッチの数みたいなの
            n_acc = 0  # 正解数

            if phase == "train":

                for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader)):
                    xx = xx.to(device)
                    yy = yy.to(device)
                    # 以前の誤差逆伝播の勾配の初期化
                    optimizer.zero_grad()

                    output = model(xx)    # ラベルの予測
                    loss = criterion(output, yy)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    n += len(xx)
                    _, y_pred = output.max(1)   # y_predにラベルの予測を入れる
                    n_acc += (yy == y_pred).float().sum().item()      # 正解数

                train_losses.append(running_loss/i)
                train_acc.append(n_acc/n)

            if phase == "val":

                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    # 確率が最大のクラスを返す。また，推論の計算だけなので自動微分はいらないから明記する
                    with torch.no_grad():
                        _, y_pred = model(x).max(1)

                    ys.append(y)
                    y_preds.append(y_pred)
                
                # ミニバッチごとの予測を計算を一つにまとめる
                ys = torch.cat(ys, dim = 0)   # ysを縦方向に結合させる
                y_preds = torch.cat(y_preds, dim = 0)

                # 予測精度を計算
                acc = (ys == y_preds).float().sum()/len(ys)
                val_acc.append(acc.item())
                
                # print(, flush = True)にすることにより、重い処理があってもこのprint()文を表示させるというoption
                print("{}回目のepochでの訓練loss: {} 訓練精度: {} 検証精度 : {}".format(epoch+1, train_losses[-1], train_acc[-1], val_acc[-1]), flush = True)

            
    return train_acc, val_acc
