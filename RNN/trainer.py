import torch
import torch.nn as nn
from torch import optim
import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from early_stopping import EarlyStopping 
from sklearn.utils import shuffle

def trainer(model, train_data, target_data, epoch_num):

    # GPUがあるなら使う
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We use", device)

    train_losses = []
    val_losses = []
    x_train, x_val, t_train, t_val = train_test_split(train_data, target_data, test_size = 0.2, shuffle = False)

    model.to(device)
    # ネットワークがある程度固定化できれば固定化する
    torch.backends.cudnn.benchmark = True
    loss_func = nn.MSELoss(reduction = "mean")
    # amsgrad オプションは, 2019年に発表された「On the convergence of Adam and Beyond」というモデルを使用するかどうか
    optimizer = optim.Adam(model.parameters(), amsgrad = True)
    early_stopping = EarlyStopping(gaman_count = 10, verbose = 1)

    batch_size = 100
    train_batch_size = x_train.shape[0] // batch_size + 1
    val_batch_size = x_val.shape[0] // batch_size + 1

    for epoch in range(epoch_num):
        
        print("=============================================")
        print("epoch{}/{}".format(epoch + 1, epoch_num))

        x_, t_ = shuffle(x_train, t_train)

        # 訓練モードと検証モードを交互に行う
        for phase in ["train", "val"]:

            if phase == "train":
                # モデルを訓練モードにする
                model.train()
                running_loss = 0
                mse_score = 0

                for batch in range(train_batch_size):
                    start = batch * batch_size
                    end = start + batch_size

                    # tensor に変換してからdeviceに送る
                    xx = torch.Tensor(x_[start:end]).to(device)
                    tt = torch.Tensor(t_[start:end]).to(device)
                    output = model(xx)
                    loss = loss_func(output, tt)
                    # 誤差逆伝播. 
                    # 以前の誤差逆伝播の勾配の初期化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                train_losses.append(running_loss/train_batch_size)

            if phase == "val":

                # 検証モードに変更
                model.eval()
                ys = []
                y_preds = []
                
                for batch in range(val_batch_size):
                    start = batch * batch_size
                    end = start + batch_size

                    xx = torch.Tensor(x_[start:end]).to(device)
                    tt = torch.Tensor(t_[start:end]).to(device)

                    with torch.no_grad():
                        y_pred = model(xx)

                    ys.append(tt)
                    y_preds.append(y_pred)

                # ミニバッチの計算を一つにまとめる
                ys = torch.cat(ys, dim = 0)   # ys を縦方向に結合する
                y_preds = torch.cat(y_preds, dim = 0) # y_preds を縦方向に結合

                # MSEを計算
                mse_score = F.mse_loss(y_preds, ys).item()

                val_losses.append(mse_score)

                print("{}回目のepochでの訓練Loss: {} 検証Loss: {}".format(epoch + 1, train_losses[-1], val_losses[-1]), flush = True)

    return train_losses, val_losses
        
