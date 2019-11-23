import os
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms

# 乱数の固定
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# 入力画像の前処理
class Imagetransform():

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(resize, scale = (0.5, 1.0)), # augumentation
                transforms.RandomHorizontalFlip(), # 確率1/2で左右反転
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  
            ]), 
            "val" : transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize), # 画像中央をresizeの正方形で切り取る
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase = "train"):
        '''
        phase: "train" or "val"
            前処理のモードを指定する
        '''

        return self.data_transform[phase](img)

    

# 画像へのファイルパスリストを作成する

def make_filepath_list(phase = "train"):

    '''
    parameters 
    ==========
    phase : 'train' or 'val'
        train data か validation データを指定
    
    return 
    ===========
    path_list : list

    '''
    rootpath = "./image/"
    target_path = os.path.join(rootpath + phase + '/**/*.jpg')
    print(target_path)

    path_list = []    # 格納するよ

    for path in glob.glob(target_path):   # globでtarget_path以下のpathを全て取得する
        path_list.append(path)

    return path_list


# 訓練データと検証データについて，それぞれのインスタンスを作成する

class beautifulDataset(data.Dataset):
    '''
    transformとしては先ほど実装したImagetransform()を用いる

    '''

    def __init__(self, file_list, transform = None, phase = "train"):
        self.file_list = file_list
        self.transform = transform    # 前処理クラスのインスタンス
        self.phase = phase

    def __len__(self):
        '画像の枚数を返す'

        return len(self.file_list)

    def __getitem__(self, index):
        '''
        前処理した画像のtensor形式のデータとラベルを取得
        '''

        # index番目の画像を取得
        img_path = self.file_list[index]
        img = Image.open(img_path)   # [h][w][RGB]

        # 画像の前処理
        transformed_img = self.transform(img, self.phase)

        # ラベルづけ
        if "史帆" in img_path:
            label = 0

        elif "郡司" in img_path:
            label = 1

        elif "小坂" in img_path:
            label = 2

        return transformed_img, label


