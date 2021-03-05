#!/usr/bin/env python3
#coding: utf-8
#------------------------------------------------
#
# 2 class識別 (Genderの学習) 
#    (身長, 体重)データから性別を推定
#
#------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
import time
import neuralnet as nn


if __name__ == '__main__':
    """ 
    Main function
        -l (--loop)     <loop count>                  (default: 1000)
        -r (--rho)      <learning rate>               (default: 0.1)
        -u (--num_unit) <unit number of middle layer> (default: 3)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--loop', type=int, default=100, help='loop number')
    parser.add_argument('-r', '--rho', type=float, default=0.001, help='learning rate rho')
    parser.add_argument('-u', '--num_unit', type=int, default=3, help='unit number of middle layer')
    args = parser.parse_args()    


    # class1の学習データ
    male         = pd.read_csv(filepath_or_buffer='./data.gender/genderPattern2learn.male.dat', encoding='ms932', sep=',', header=None)
    male_values  = male.values
    label_male   = male_values[:,0]      # ラベルはレコードの先頭に入っている
    male_data    = male_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_male     = male.values.shape[0]  # 学習データ数(男)
    x0_male      = np.ones(num_male)     # x0は常に1

    # class2の学習データ
    female         = pd.read_csv(filepath_or_buffer='./data.gender/genderPattern2learn.female.dat', encoding='ms932', sep=',', header=None)
    female_values  = female.values
    label_female   = female_values[:,0]      # ラベルはレコードの先頭に入っている
    female_data    = female_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_female     = female.values.shape[0]  # 学習データ数(女)
    x0_female      = np.ones(num_female)     # x0は常に1

    train_num = num_male + num_female
    print('学習パターン数 = ', train_num)

    # データの正規化
    all_data = np.r_[male_data, female_data]
    mean = all_data.mean(axis=0)
    std  = all_data.std(axis=0)
    print("平均= ", mean, "標準偏差= ", std)
    male_data   = (male_data-mean)/std
    female_data = (female_data-mean)/std


    # 拡張特徴ベクトルを並べた行列
    x1 = np.c_[x0_male, male_data]
    x2 = np.c_[x0_female, female_data]

    # 全学習データを統合
    matX   = np.r_[x1, x2] 
    labels = np.r_[label_male, label_female]

    # Neural Network の生成 (入力層数, 中間層数, 出力層数, 中間層活性化関数, 出力層活性化関数)
    mlp = nn.MultiLayerPerceptron(2, args.num_unit, 1, "tanh", "sigmoid")


    # Neural Network の学習
    ts = time.time() # 開始時刻
    for loop in range(args.loop): # loop回数
        mlp.fit_local(matX, labels, args.rho)                # 1 episode分のデータでWを更新
        g2, g3 = mlp.predict(matX)                           # 1 episode分のデータに対する g2, g3を予測
        entropy = mlp.error_2class_cross_entropy(g3, labels) # 1 episode分のcross entropyを計算
        print("[%d] cross entropy=%7.3f" % (loop, entropy))
    te = time.time() # 終了時刻
    print ("\ncompute : %.3f sec" % (te-ts))

    # 学習データの識別
    print("======== 学習データに対する評価 ========")
    mlp.infer_2class(matX, labels)

    # 未知テストデータの識別
    test        = pd.read_csv(filepath_or_buffer='./data.gender/genderPattern2recog.dat', encoding='ms932', sep=',', header=None)
    test_values = test.values
    labels_test = test_values[:,0]      # ラベルはレコードの先頭に入っている
    test_data   = test_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_test    = test.values.shape[0]  # テストデータ数
    x0_test      = np.ones(num_test)    # x0は常に1

    # データの正規化
    test_dataN = (test_data-mean)/std
    matTest    = np.c_[x0_test, test_dataN]

    print("======== テストデータに対する評価 ========")
    result_infer, cMat = mlp.infer_2class(matTest, labels_test)

    g2, g3 = mlp.predict(matTest)

    print("cross entropy = %7.3f" % (mlp.error_2class_cross_entropy(g3, labels_test)))

    # 誤識別データの表示
    for i in range(len(result_infer)):
        if result_infer[i] != labels_test[i]:
            print("data:",test_data[i], "推定値=", result_infer[i], "正解=", labels_test[i])

