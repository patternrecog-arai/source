#!/usr/bin/env python3
#coding:utf-8
#
# 数字パターンの一括表示
#
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--type', type=str, default='L', help='data type (L)earn or (T)est')
    args = parser.parse_args()    

    if args.type == "L":
        # 学習データ
        df_pattern = pd.read_csv(filepath_or_buffer='./Pattern/pattern2learn.kinds.dat', encoding='ms932', sep=',', header=None)
    else:
        # テストデータ
        df_pattern = pd.read_csv(filepath_or_buffer='./Pattern/pattern2recog.kinds.dat', encoding='ms932', sep=',', header=None)
    pattern_values = df_pattern.values
    label_pattern  = pattern_values[:,0]      # ラベルはレコードの先頭に入っている
    pattern_data   = pattern_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_pattern    = len(pattern_data)        # データ数

    print('パターン数 = ', num_pattern)

    # パターンの表示
    fig = plt.figure(figsize=(12,8))

    sp = []
    for no in range(128):
        sp.append(fig.add_subplot(8, 16, no+1))

    no = 0
    for vecX, label in zip(pattern_data, label_pattern):
        img = vecX.reshape(7,5)
        sp[no%128].cla()
        sp[no%128].axis('off')
        sp[no%128].imshow(img)

        no += 1
        plt.pause(0.01)
        print("[No. %d] %d" % (no, label))

    print('異なるパターンは全てで', no, '個あります')

    plt.show()
    sys.exit()
