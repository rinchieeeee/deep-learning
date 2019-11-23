import os
import urllib.request
import zipfile
import tarfile

import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
import numpy as np

# フォルダ「data」が存在しない場合は作成する
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# MNIStをダウンロードして読み込む
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version = 1, data_home = "./data")

data_dir_path = "./data/img_78/"
if not os.path.exists(data_dir_path):
    os.mkdir(data_dir_path)

# MNIST1から数字の7, 8の画像だけフォルダ img_78に保存するよ

count_7 = 0
count_8 = 0
N = 200   # 200枚ずつ作成

X = mnist.data
y = mnist.target


for i in range(len(X)):

    # generate image of 7
    if (y[i] is "7") and (count_7 < N):
        file_path = "./data/img_78/img_7_" + str(count_7) + ".jpg"
        im_f = (X[i].reshape(28, 28))
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
        pil_img_f.save(file_path)  # 保存
        count7+=1 
    
    # 画像8の作成
    if (y[i] is "8") and (count8<max_num):
        file_path="./data/img_78/img_8_"+str(count_8)+".jpg"
        im_f=(X[i].reshape(28, 28))  # 画像を28*28の形に変形
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
        pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
        pil_img_f.save(file_path)  # 保存
        count8+=1 