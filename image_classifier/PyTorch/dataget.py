from icrawler.builtin import GoogleImageCrawler
import os
import sys
import glob
import random
import shutil


def get_image(name, file_path, data_count, sample_filter = None):
    crawler = GoogleImageCrawler(storage = {"root_dir" : file_path + "/train" +"/" + name})
    
    filters = dict(
                size = "large", 
                type = "photo"
    )

    # クローリングの実行
    crawler.crawl(keyword=name, filters = sample_filter, max_num = data_count)
    
    # valディレクトリの作成
    if os.path.isdir(file_path + "/val" + "/" + name):
        shutil.rmtree(file_path + "/val"+ "/" +name)

    os.makedirs(file_path + "/val" +"/" + name)


    # ダウンロードファイルのリストを作成
    filelist = glob.glob(file_path + "/train" + "/" + name + "/*")
    # 訓練データの2割りをvalデータとして保存
    ration = 0.2
    val_files = random.sample(filelist, int(len(filelist)*ration))

    for line in val_files:
        shutil.move(line, file_path + "/val" + "/" + name)
